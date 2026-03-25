"""Level 3: Proximal Policy Optimization (PPO).

Schulman et al. (2017), 'Proximal Policy Optimization Algorithms.'

DQN (L05) learned Q-values and derived actions by taking the argmax.
That works for discrete actions (left, stay, right) but breaks when
actions are continuous—there is no finite set to take the max over.

PPO takes a different approach: instead of learning values and
deriving a policy, it learns the policy directly. The actor network
outputs a probability distribution over actions (a Gaussian bell
curve for continuous control), and the agent samples from it. The
critic network estimates state values for computing advantages.

The key insight is the clipped surrogate objective. Naive policy
gradient can make catastrophically large updates—one bad step
can destroy a good policy. PPO constrains the update: the ratio
of new to old action probability cannot deviate too far from 1.
If the ratio tries to move outside [1 - epsilon, 1 + epsilon],
the gradient is clipped to zero, preventing the update.

Three ideas make PPO work:

  1. Clipped surrogate: bounds the policy update to a trust region
     without the complexity of TRPO's second-order optimization.

  2. Generalized Advantage Estimation (GAE): blends short-term
     and long-term credit assignment via lambda, same tradeoff
     as TD(lambda) from L03 but applied to advantages.

  3. Multiple epochs: reuses the same trajectory data for K passes
     of gradient descent, extracting more learning from each batch
     of experience. On-policy methods are data-hungry; this helps.
"""

import math

from policywerk.primitives import scalar, vector, matrix
from policywerk.primitives.progress import progress_bar, progress_done
from policywerk.primitives.activations import tanh_, identity
from policywerk.primitives.losses import mse_derivative
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.network import Network, create_network, network_forward
from policywerk.building_blocks.grad import backward, LayerGradients
from policywerk.building_blocks.optimizers import create_adam_state, adam_update
from policywerk.building_blocks.distributions import Gaussian

Vector = list[float]


def _compute_gae_with_resets(
    rewards: Vector,
    values: Vector,
    dones: list[bool],
    next_value: float,
    gamma: float,
    lam: float,
) -> Vector:
    """GAE that handles episode boundaries within a trajectory.

    When collecting T steps, episodes can end and restart mid-trajectory.
    Standard GAE assumes one continuous episode. This version zeros the
    advantage carry at episode boundaries so credit does not leak across
    unrelated episodes.

    At each step, the TD residual is:
      delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)

    The (1 - done_t) factor zeroes the bootstrap when the episode ended,
    and the carry is also zeroed so advantages don't propagate backward
    across the boundary.
    """
    t = len(rewards)
    advantages = [0.0] * t
    gae_val = 0.0
    for i in range(t - 1, -1, -1):
        if i == t - 1:
            next_val = next_value
            next_nonterminal = 1.0 - float(dones[i])
        else:
            next_val = values[i + 1]
            next_nonterminal = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * next_val * next_nonterminal - values[i]
        gae_val = delta + gamma * lam * next_nonterminal * gae_val
        advantages[i] = gae_val
    return advantages


def _policy_gradient(
    mean: float,
    log_std: float,
    action: float,
    advantage: float,
    log_prob_old: float,
    clip_epsilon: float,
    entropy_coeff: float,
) -> list[float]:
    """Compute dL/d[mean, log_std] for the PPO clipped surrogate.

    The loss is:
      L = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A) - c * entropy

    where ratio = exp(log_prob_new - log_prob_old).

    Returns [dL/d_mean, dL/d_log_std]—the gradient of the loss
    with respect to the actor network's two outputs. This vector
    feeds into backward() as the loss_grad argument.
    """
    # Current distribution parameters.
    # Clamp log_std for numerical stability; track whether the clamp
    # is active so we can zero the gradient when saturated.
    clamped_log_std = scalar.clamp(log_std, -2.0, 2.0)
    log_std_active = -2.0 < log_std < 2.0  # gradient flows only when inside
    std = scalar.exp(clamped_log_std)
    inv_std = scalar.inverse(std)

    # Gaussian log probability: -0.5 * ((a - mu)/sigma)^2 - log(sigma) - 0.5*log(2*pi)
    diff = scalar.subtract(action, mean)
    z = scalar.multiply(diff, inv_std)
    log_prob_new = scalar.subtract(
        scalar.negate(scalar.multiply(0.5, scalar.multiply(z, z))),
        scalar.add(scalar.log(std), 0.5 * math.log(2.0 * math.pi)),
    )

    # PPO ratio
    ratio = scalar.exp(scalar.subtract(log_prob_new, log_prob_old))

    # Clipped surrogate
    surr1 = scalar.multiply(ratio, advantage)
    clamped_ratio = scalar.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surr2 = scalar.multiply(clamped_ratio, advantage)
    # min(surr1, surr2): when the clipped term is the min, its gradient
    # is zero (the clip kills it). This is correct for both positive and
    # negative advantages—no special casing needed.
    use_clipped = surr2 < surr1

    # Gradient of log_prob w.r.t. mean and log_std
    # d_log_prob / d_mean = (action - mean) / std^2
    dlp_dmean = scalar.multiply(diff, scalar.multiply(inv_std, inv_std))
    # d_log_prob / d_log_std = ((action - mean)^2 / std^2 - 1)
    # (chain rule: d/d_log_std of log_prob, where std = exp(log_std))
    dlp_dlogstd = scalar.subtract(scalar.multiply(z, z), 1.0)

    # Gradient of ratio w.r.t. log_prob: d_ratio / d_log_prob = ratio
    # So d_surr1 / d_param = advantage * ratio * d_log_prob / d_param
    if use_clipped:
        # Clipped branch: gradient is zero (the clip kills it)
        dsurr_dmean = 0.0
        dsurr_dlogstd = 0.0
    else:
        # Unclipped branch: gradient flows through
        dsurr_dmean = scalar.multiply(advantage, scalar.multiply(ratio, dlp_dmean))
        dsurr_dlogstd = scalar.multiply(advantage, scalar.multiply(ratio, dlp_dlogstd))

    # Entropy gradient: entropy = 0.5 * (1 + log(2*pi) + 2*log_std)
    # d_entropy / d_log_std = 1.0
    # d_entropy / d_mean = 0.0
    dent_dlogstd = 1.0

    # Total loss = -surrogate - entropy_coeff * entropy
    # dL/d_param = -dsurr/d_param - entropy_coeff * dent/d_param
    dl_dmean = scalar.negate(dsurr_dmean)
    dl_dlogstd = scalar.subtract(
        scalar.negate(dsurr_dlogstd),
        scalar.multiply(entropy_coeff, dent_dlogstd),
    )

    # When log_std is clamped (saturated at bounds), the derivative
    # through the clamp is zero—no gradient should flow.
    if not log_std_active:
        dl_dlogstd = 0.0

    return [dl_dmean, dl_dlogstd]


def ppo(
    env,
    num_iterations: int = 250,
    steps_per_iter: int = 500,
    num_epochs: int = 3,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_epsilon: float = 0.2,
    learning_rate_actor: float = 0.001,
    learning_rate_critic: float = 0.003,
    entropy_coeff: float = 0.005,
    hidden_size: int = 64,
    seed: int = 42,
) -> tuple[Network, Network, list[dict]]:
    """Train a PPO agent with continuous actions on a balance task.

    The actor network outputs [mean, log_std] for a 1D Gaussian policy.
    The critic network outputs a scalar value estimate. Both are trained
    with Adam.

    Returns (actor_net, critic_net, history) where each history entry has:
      iteration, avg_reward, episodes_completed, value_loss, entropy, mean_std
    """
    rng = create_rng(seed)

    # Determine input dimension from the environment
    state = env.reset()
    state_dim = len(state.features)

    # Actor: state -> [mean, log_std] for 1D Gaussian policy
    # tanh hidden layer keeps activations bounded, identity output is unconstrained
    actor_net = create_network(
        rng, [state_dim, hidden_size, 2], [tanh_, identity]
    )
    # Critic: state -> scalar value estimate
    critic_net = create_network(
        rng, [state_dim, hidden_size, 1], [tanh_, identity]
    )

    actor_adam = create_adam_state(actor_net)
    critic_adam = create_adam_state(critic_net)

    history: list[dict] = []

    for iteration in range(num_iterations):
        # ---- Phase 1: Collect trajectory of T steps ----
        states: list[Vector] = []
        raw_actions: list[float] = []  # unclamped samples (for log_prob in update)
        rewards: list[float] = []
        log_probs_old: list[float] = []
        values: list[float] = []
        dones: list[bool] = []

        ep_rewards: list[float] = []
        ep_reward = 0.0

        for _t in range(steps_per_iter):
            # Actor forward: get policy distribution
            actor_out, _ = network_forward(actor_net, state.features)
            mean = actor_out[0]
            log_std = scalar.clamp(actor_out[1], -2.0, 2.0)
            std = scalar.exp(log_std)

            # Sample from the Gaussian, then clamp for the environment.
            # The log_prob is computed on the raw sample (not the clamped
            # action) so the PPO ratio is consistent: it compares old and
            # new probability of the same sample. The environment executes
            # the clamped version, which is what generates the reward.
            dist = Gaussian([mean], [std])
            raw_action = dist.sample(rng)[0]
            lp = dist.log_prob([raw_action])
            action = scalar.clamp(raw_action, -1.0, 1.0)

            # Critic forward: get value estimate
            value_out, _ = network_forward(critic_net, state.features)
            value = value_out[0]

            # Step the environment
            next_state, reward, done = env.step_continuous(action)
            ep_reward += reward

            # Store transition data
            states.append(list(state.features))
            raw_actions.append(raw_action)
            rewards.append(reward)
            log_probs_old.append(lp)
            values.append(value)
            dones.append(done)

            state = next_state
            if done:
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                state = env.reset()

        # Bootstrap value for the last state (used by GAE)
        last_val_out, _ = network_forward(critic_net, state.features)
        next_value = last_val_out[0]

        # ---- Phase 2: Compute advantages via GAE ----
        advantages = _compute_gae_with_resets(
            rewards, values, dones, next_value, gamma, lam
        )
        # Returns = advantages + values (the target for the critic)
        returns = [scalar.add(a, v) for a, v in zip(advantages, values)]

        # Normalize advantages to zero mean, unit variance
        adv_mean = sum(advantages) / len(advantages)
        adv_var = sum((a - adv_mean) ** 2 for a in advantages) / len(advantages)
        adv_std = math.sqrt(adv_var) + 1e-8
        advantages = [(a - adv_mean) / adv_std for a in advantages]

        # ---- Phase 3: K epochs of PPO updates ----
        iter_value_loss = 0.0
        iter_entropy = 0.0
        iter_std_sum = 0.0
        update_count = 0

        for _epoch in range(num_epochs):
            # Accumulate actor gradients
            actor_acc: list[LayerGradients] = []
            for layer in actor_net.layers:
                rows = len(layer.weights)
                cols = len(layer.weights[0])
                actor_acc.append(LayerGradients(
                    weight_grads=matrix.zeros(rows, cols),
                    bias_grads=vector.zeros(len(layer.biases)),
                ))

            # Accumulate critic gradients
            critic_acc: list[LayerGradients] = []
            for layer in critic_net.layers:
                rows = len(layer.weights)
                cols = len(layer.weights[0])
                critic_acc.append(LayerGradients(
                    weight_grads=matrix.zeros(rows, cols),
                    bias_grads=vector.zeros(len(layer.biases)),
                ))

            for t in range(steps_per_iter):
                # ---- Actor update ----
                actor_out, actor_cache = network_forward(actor_net, states[t])
                a_mean = actor_out[0]
                a_log_std = scalar.clamp(actor_out[1], -2.0, 2.0)
                a_std = scalar.exp(a_log_std)

                # Policy gradient uses raw (unclamped) action to match
                # the log_prob_old that was computed on the raw sample.
                loss_grad = _policy_gradient(
                    a_mean, a_log_std, raw_actions[t], advantages[t],
                    log_probs_old[t], clip_epsilon, entropy_coeff,
                )
                actor_grads = backward(actor_net, actor_cache, loss_grad)

                for i, grads in enumerate(actor_grads):
                    actor_acc[i].weight_grads = matrix.add(
                        actor_acc[i].weight_grads, grads.weight_grads)
                    actor_acc[i].bias_grads = vector.add(
                        actor_acc[i].bias_grads, grads.bias_grads)

                # ---- Critic update ----
                value_out, critic_cache = network_forward(critic_net, states[t])
                value_grad = mse_derivative(value_out, [returns[t]])
                critic_grads = backward(critic_net, critic_cache, value_grad)

                for i, grads in enumerate(critic_grads):
                    critic_acc[i].weight_grads = matrix.add(
                        critic_acc[i].weight_grads, grads.weight_grads)
                    critic_acc[i].bias_grads = vector.add(
                        critic_acc[i].bias_grads, grads.bias_grads)

                # Track metrics
                ent = 0.5 * (1.0 + math.log(2.0 * math.pi) + 2.0 * scalar.log(a_std))
                iter_entropy += ent
                iter_std_sum += a_std

                # Value loss: squared error between critic prediction and GAE return.
                # Both are based on the rewards the environment produced from the
                # clamped action, so this diagnostic is internally consistent.
                # Policy loss is omitted because the ratio uses the raw (unclamped)
                # action's log_prob while the advantage reflects the clamped action's
                # reward—logging it would be misleading.
                v_err = value_out[0] - returns[t]
                iter_value_loss += v_err * v_err

                update_count += 1

            # Average gradients over trajectory length
            inv_t = scalar.inverse(float(steps_per_iter))
            for ag in actor_acc:
                ag.weight_grads = matrix.scale(inv_t, ag.weight_grads)
                ag.bias_grads = vector.scale(inv_t, ag.bias_grads)
            for cg in critic_acc:
                cg.weight_grads = matrix.scale(inv_t, cg.weight_grads)
                cg.bias_grads = vector.scale(inv_t, cg.bias_grads)

            # Adam updates
            adam_update(actor_net, actor_acc, actor_adam, learning_rate_actor)
            adam_update(critic_net, critic_acc, critic_adam, learning_rate_critic)

        # Record history
        avg_reward = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0
        avg_entropy = iter_entropy / update_count if update_count else 0.0
        avg_std = iter_std_sum / update_count if update_count else 1.0

        avg_vloss = iter_value_loss / update_count if update_count else 0.0

        history.append({
            "iteration": iteration,
            "avg_reward": avg_reward,
            "episodes_completed": len(ep_rewards),
            "value_loss": avg_vloss,
            "entropy": avg_entropy,
            "mean_std": avg_std,
        })

        progress_bar(
            iteration + 1, num_iterations,
            f"reward={avg_reward:+7.1f}  std={avg_std:.3f}",
        )

    progress_done()
    return actor_net, critic_net, history


def balance_outcome(steps: int, final_reward: float, max_steps: int
                    ) -> tuple[bool, str]:
    """Determine whether a Balance episode ended in success or failure.

    Returns (survived, label). Uses the final reward to distinguish
    timeout success (reward=1.0) from a terminal fall (reward=0.0)
    at the same step count.
    """
    survived = (final_reward == 1.0) and steps >= max_steps
    if survived:
        label = f"Balanced for {steps} steps!"
    else:
        label = f"Fell after {steps} steps"
    return survived, label
