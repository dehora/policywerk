"""Level 3: Deep Q-Network (DQN).

Mnih et al. (2013), 'Playing Atari with Deep Reinforcement Learning.'

Q-learning (L04) stores one value per state-action pair in a table.
That works when the state space is small and discrete — 48 cells on
the cliff world. But when the agent sees pixels, the table approach
breaks: each observation is effectively unique, and the table cannot
generalize across similar-looking states.

DQN replaces the table with a neural network. The network takes
state features as input and outputs Q-values for every action.
Instead of looking up Q(s, a) in a dict, the agent runs a forward
pass and reads off the action values.

Three ideas make this work:

  1. Experience replay: store transitions in a buffer and train on
     random samples. This breaks the temporal correlation that would
     otherwise make training unstable (consecutive frames are nearly
     identical).

  2. Target network: a frozen copy of the online network, updated
     periodically. The TD target uses this copy so the target doesn't
     shift with every gradient step.

  3. Epsilon decay: start with high exploration (random actions) and
     gradually shift to exploitation (network-chosen actions) as the
     network improves.
"""

from policywerk.primitives import scalar, vector, matrix
from policywerk.primitives.progress import progress_bar, progress_done
from policywerk.primitives.activations import relu, identity
from policywerk.primitives.losses import huber, huber_derivative
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.dense import DenseLayer
from policywerk.building_blocks.network import Network, create_network, network_forward
from policywerk.building_blocks.grad import backward, LayerGradients
from policywerk.building_blocks.optimizers import create_adam_state, adam_update
from policywerk.building_blocks.replay_buffer import ReplayBuffer
from policywerk.building_blocks.mdp import Transition
from policywerk.building_blocks.policies import epsilon_greedy

Vector = list[float]
Matrix = list[list[float]]


def _copy_network(source: Network) -> Network:
    """Deep copy a network's weights and biases.

    The target network needs to be an independent copy — modifying
    the online network's weights during training must not change
    the target network's predictions.
    """
    layers = []
    for layer in source.layers:
        new_weights = [list(row) for row in layer.weights]
        new_biases = list(layer.biases)
        layers.append(DenseLayer(weights=new_weights, biases=new_biases))
    return Network(layers=layers, activation_fns=list(source.activation_fns))


def _linear_epsilon(episode: int, start: float, end: float, decay_episodes: int) -> float:
    """Linearly anneal epsilon from start to end over decay_episodes."""
    if decay_episodes <= 0:
        return end
    fraction = min(episode / decay_episodes, 1.0)
    return start + fraction * (end - start)


def dqn(
    env,
    num_episodes: int = 200,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 150,
    learning_rate: float = 0.0005,
    batch_size: int = 16,
    replay_capacity: int = 5000,
    min_replay_size: int = 200,
    target_update_freq: int = 10,
    train_every: int = 4,
    hidden_size: int = 32,
    seed: int = 42,
) -> tuple[Network, list[dict]]:
    """Train a DQN agent on a pixel-observation environment.

    Replaces Q-learning's table with a neural network:
      Table:   Q[(label, action)] -> float
      Network: forward(state.features) -> [Q(s,0), Q(s,1), ..., Q(s,n)]

    The network generalizes across states — similar pixel patterns
    produce similar Q-values, unlike a table where each state is
    independent.

    Returns (trained_network, history) where each history entry has:
      episode, total_reward, steps, epsilon, avg_loss, avg_q
    """
    rng = create_rng(seed)

    # Determine input/output dimensions from the environment
    state = env.reset()
    input_dim = len(state.features)
    num_actions = env.num_actions()

    # Online network: state features → hidden (relu) → Q-values (identity)
    online_net = create_network(
        rng, [input_dim, hidden_size, num_actions], [relu, identity]
    )
    target_net = _copy_network(online_net)
    adam_states = create_adam_state(online_net)

    replay = ReplayBuffer(replay_capacity)
    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        epsilon = _linear_epsilon(ep, epsilon_start, epsilon_end, epsilon_decay_episodes)
        total_reward = 0.0
        step_count = 0
        episode_losses: list[float] = []
        episode_q_vals: list[float] = []

        done = False
        while not done:
            # Select action: epsilon-greedy over online network's Q-values
            q_values, _ = network_forward(online_net, state.features)
            action = epsilon_greedy(rng, q_values, epsilon)
            episode_q_vals.append(max(q_values))

            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            replay.add(Transition(state, action, reward, next_state, done))

            # Train every N steps once the buffer has enough experience.
            # Accumulate gradients across the batch, average them, then
            # do a single Adam update. This is much less noisy than
            # per-sample updates.
            if step_count % train_every == 0 and len(replay) >= min_replay_size:
                batch = replay.sample(rng, batch_size)

                # Initialize accumulated gradients to zero
                acc_grads: list[LayerGradients] = []
                for layer in online_net.layers:
                    rows = len(layer.weights)
                    cols = len(layer.weights[0])
                    acc_grads.append(LayerGradients(
                        weight_grads=matrix.zeros(rows, cols),
                        bias_grads=vector.zeros(len(layer.biases)),
                    ))

                batch_loss = 0.0
                for t in batch:
                    # Compute TD target using the target network
                    if t.done:
                        target_q = t.reward
                    else:
                        target_q_vals, _ = network_forward(target_net, t.next_state.features)
                        target_q = scalar.add(
                            t.reward,
                            scalar.multiply(gamma, max(target_q_vals)),
                        )

                    # Forward pass on online network
                    online_q_vals, cache = network_forward(online_net, t.state.features)

                    # Build target vector: copy Q-values, replace the taken
                    # action's value with the TD target. Non-action outputs
                    # have zero gradient because predicted == target.
                    targets = list(online_q_vals)
                    targets[t.action] = target_q

                    # Backprop
                    loss_grad = huber_derivative(online_q_vals, targets)
                    gradients = backward(online_net, cache, loss_grad)

                    # Accumulate gradients
                    for i, grads in enumerate(gradients):
                        acc_grads[i].weight_grads = matrix.add(
                            acc_grads[i].weight_grads, grads.weight_grads)
                        acc_grads[i].bias_grads = vector.add(
                            acc_grads[i].bias_grads, grads.bias_grads)

                    # Track loss (Huber, matching the gradient function)
                    batch_loss += huber(online_q_vals, targets)

                # Average gradients over batch
                inv_bs = scalar.inverse(float(batch_size))
                for ag in acc_grads:
                    ag.weight_grads = matrix.scale(inv_bs, ag.weight_grads)
                    ag.bias_grads = vector.scale(inv_bs, ag.bias_grads)

                # Single Adam update with averaged gradients
                adam_update(online_net, acc_grads, adam_states, learning_rate)
                episode_losses.append(batch_loss / batch_size)

            state = next_state

        # Update target network periodically
        if (ep + 1) % target_update_freq == 0:
            target_net = _copy_network(online_net)

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
        avg_q = sum(episode_q_vals) / len(episode_q_vals) if episode_q_vals else 0.0

        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "steps": step_count,
            "epsilon": epsilon,
            "avg_loss": avg_loss,
            "avg_q": avg_q,
        })

        progress_bar(ep + 1, num_episodes, f"reward={total_reward:+6.2f}")

    progress_done()
    return online_net, history


def greedy_poster_frame(net: Network, env, max_steps: int = 200,
                        min_score: int = 2) -> list[list[list[float]]]:
    """Run a greedy rollout and return a mid-game color frame.

    Picks the first frame where at least min_score bricks have been
    destroyed and the game is still going. Falls back to the final
    frame if the agent clears or dies before reaching min_score.
    """
    state = env.reset()
    frame = env.render_color_frame()  # fallback
    for _ in range(max_steps):
        q_vals, _ = network_forward(net, state.features)
        action = q_vals.index(max(q_vals))
        state, _, done = env.step(action)
        if env._score >= min_score and not done:
            return env.render_color_frame()
        if done:
            return env.render_color_frame()
    return frame
