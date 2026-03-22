"""Level 3: Barto/Sutton/Anderson actor-critic (1983).

The first actor-critic architecture — two single neurons that learn
to balance a pole through trial and error alone. Unlike Bellman's
value iteration (L01), the agent has no access to the environment's
rules. It must discover good behavior by acting, observing rewards,
and adjusting.

The architecture has two components:

  ACE (Adaptive Critic Element): predicts how good the current state
    is (the value function). It produces a TD error signal — "this
    turned out better or worse than I predicted" — which drives
    learning in both components.

  ASE (Adaptive Search Element): selects actions. It outputs a
    weighted sum of the state features plus noise, and thresholds
    to choose left or right. The TD error from the ACE tells it
    whether the chosen action was good or bad.

Both components use eligibility traces to solve credit assignment:
when the pole finally falls, which of the last 50 actions was
responsible? Traces maintain a fading memory of recent state-action
pairs, so the blame is spread proportionally to recency.

The state is discretized into boxes (bins of angle and velocity).
Each box gets one weight in the ACE and one in the ASE. The input
is a one-hot vector: all zeros except a 1.0 at the current box.
"""

from dataclasses import dataclass, field
import random as _random

from policywerk.primitives import scalar, vector
from policywerk.primitives.random import create_rng, normal
from policywerk.building_blocks.mdp import State, Transition, Episode

Vector = list[float]


@dataclass
class ACE:
    """Adaptive Critic Element — predicts state value.

    weights: one weight per state box, v[i]. The prediction for a
        state is simply v[box_index] (since input is one-hot).
    traces: eligibility traces, one per box. Recently visited boxes
        have high traces and receive larger weight updates.
    prev_prediction: the prediction from the previous time step,
        needed to compute the TD error.
    """
    weights: Vector
    traces: Vector
    prev_prediction: float = 0.0


@dataclass
class ASE:
    """Adaptive Search Element — selects actions.

    weights: one weight per state box, w[i]. Positive weights favor
        action 1 (right), negative favor action 0 (left).
    traces: eligibility traces encoding which box-action pairs were
        recently active.
    """
    weights: Vector
    traces: Vector


def create_ace_ase(num_boxes: int) -> tuple[ACE, ASE]:
    """Create ACE and ASE with zero-initialized weights and traces."""
    ace = ACE(
        weights=vector.zeros(num_boxes),
        traces=vector.zeros(num_boxes),
    )
    ase = ASE(
        weights=vector.zeros(num_boxes),
        traces=vector.zeros(num_boxes),
    )
    return ace, ase


def state_to_box(state: State, num_boxes: int) -> int:
    """Convert a discretized state label "a_bin,v_bin" to a box index.

    The balance environment labels states as "a_bin,v_bin" where each
    bin is 0-5. This maps to a flat index: a_bin * num_vel_bins + v_bin.
    """
    parts = state.label.split(",")
    a_bin = int(parts[0])
    v_bin = int(parts[1])
    # Infer number of velocity bins from total boxes and max angle bin
    # For 36 boxes with 6 angle bins: num_vel_bins = 6
    num_vel_bins = 6  # matches balance.py's _VEL_BINS (5 boundaries -> 6 bins)
    return a_bin * num_vel_bins + v_bin


def state_to_input(state: State, num_boxes: int) -> Vector:
    """One-hot encode the discretized state.

    Returns a vector of num_boxes floats, all zeros except a 1.0
    at the current box index. This is the input to both ACE and ASE.
    """
    x = vector.zeros(num_boxes)
    box = state_to_box(state, num_boxes)
    if 0 <= box < num_boxes:
        x[box] = 1.0
    return x


def select_action(ase: ASE, x: Vector, rng: _random.Random,
                  noise_std: float = 1.0) -> int:
    """Choose an action using the ASE with exploration noise.

    Computes the weighted sum of inputs, adds Gaussian noise for
    exploration, and thresholds: >= 0 means action 1 (right),
    < 0 means action 0 (left).

    The noise is deliberate — without it, the ASE would always pick
    the same action for the same state and could never discover
    better alternatives.
    """
    # weighted_sum = w . x (with one-hot input, this is just w[box])
    weighted_sum = vector.dot(ase.weights, x)
    # Add exploration noise
    noisy = scalar.add(weighted_sum, normal(rng, 0.0, noise_std))
    # Threshold to binary action
    return 1 if noisy >= 0.0 else 0


def compute_td_error(ace: ACE, x: Vector, x_prev: Vector,
                     reward: float, gamma: float, done: bool) -> float:
    """Compute the TD error — the critic's surprise signal.

    TD error = reward + gamma * prediction(current) - prediction(previous)

    Positive: things turned out better than predicted.
    Negative: things turned out worse than predicted.

    On terminal states (done=True), there is no future prediction,
    so we use only the reward minus the previous prediction.
    """
    prediction = vector.dot(ace.weights, x)
    if done:
        # Terminal: no future value
        # td_error = reward - prev_prediction
        td_error = scalar.subtract(reward, ace.prev_prediction)
    else:
        # td_error = reward + gamma * prediction - prev_prediction
        td_error = scalar.subtract(
            scalar.add(reward, scalar.multiply(gamma, prediction)),
            ace.prev_prediction,
        )
    # Update stored prediction for next step
    ace.prev_prediction = prediction
    return td_error


def update_ace(ace: ACE, td_error: float, x: Vector,
               beta: float, trace_decay: float) -> None:
    """Update the critic's weights using TD error and eligibility traces.

    The trace for each box decays each step and increases when that
    box is visited. Weights are adjusted in proportion to both the
    TD error (how surprising the outcome was) and the trace (how
    recently that box was visited).
    """
    for i in range(len(ace.weights)):
        # Decay trace, then add current state's contribution
        # e[i] = decay * e[i] + x[i]
        ace.traces[i] = scalar.add(
            scalar.multiply(trace_decay, ace.traces[i]),
            x[i],
        )
        # w[i] += beta * td_error * e[i]
        ace.weights[i] = scalar.add(
            ace.weights[i],
            scalar.multiply(beta, scalar.multiply(td_error, ace.traces[i])),
        )


def update_ase(ase: ASE, td_error: float, x: Vector, action: int,
               alpha: float, trace_decay: float) -> None:
    """Update the actor's weights using TD error and eligibility traces.

    The ASE trace encodes which box-action pair was recently active.
    The direction factor (action - 0.5) is +0.5 for action 1 (right)
    and -0.5 for action 0 (left), so the trace remembers not just
    which box was active but which direction was chosen.
    """
    # Direction encoding: +0.5 for right, -0.5 for left
    direction = scalar.subtract(float(action), 0.5)
    for i in range(len(ase.weights)):
        # Decay trace, then add current state-action contribution
        # e[i] = decay * e[i] + direction * x[i]
        ase.traces[i] = scalar.add(
            scalar.multiply(trace_decay, ase.traces[i]),
            scalar.multiply(direction, x[i]),
        )
        # w[i] += alpha * td_error * e[i]
        ase.weights[i] = scalar.add(
            ase.weights[i],
            scalar.multiply(alpha, scalar.multiply(td_error, ase.traces[i])),
        )


def train_episode(env, ace: ACE, ase: ASE, rng: _random.Random,
                  num_boxes: int, gamma: float = 0.95,
                  alpha: float = 10.0, beta: float = 0.1,
                  trace_decay: float = 0.5,
                  noise_std: float = 0.1) -> tuple[Episode, list[float]]:
    """Run one training episode.

    The agent interacts with the environment step by step:
      1. Observe state, convert to one-hot input
      2. ASE selects an action (with noise)
      3. Environment returns reward and next state
      4. ACE computes TD error
      5. Both ACE and ASE update weights via traces

    Returns the episode and a list of angles (for animation).
    """
    episode = Episode()
    angles = []

    state = env.reset()
    x = state_to_input(state, num_boxes)
    # Reset critic's previous prediction for the new episode
    ace.prev_prediction = vector.dot(ace.weights, x)
    # Reset traces at the start of each episode
    for i in range(num_boxes):
        ace.traces[i] = 0.0
        ase.traces[i] = 0.0

    for step in range(env._max_steps):
        angles.append(state.features[0])  # record angle

        action = select_action(ase, x, rng, noise_std)
        next_state, env_reward, done = env.step(action)
        x_next = state_to_input(next_state, num_boxes)

        # Paper's reward convention: 0 during balancing, -1 on failure.
        # The balance env gives +1 per step and 0 on failure, so we
        # transform: failure (env_reward=0, done=True) -> -1, else -> 0
        reward = -1.0 if (done and env_reward == 0.0) else 0.0

        episode.add(Transition(
            state=state, action=action, reward=env_reward,
            next_state=next_state, done=done,
        ))

        # Compute TD error from the critic
        td_error = compute_td_error(ace, x_next, x, reward, gamma, done)

        # Update both components using the TD error
        update_ace(ace, td_error, x, beta, trace_decay)
        update_ase(ase, td_error, x, action, alpha, trace_decay)

        if done:
            break

        state = next_state
        x = x_next

    return episode, angles


def train(env, num_episodes: int = 200, seed: int = 42,
          gamma: float = 0.95, alpha: float = 10.0,
          beta: float = 0.1, trace_decay: float = 0.5,
          noise_std: float = 0.1) -> tuple[ACE, ASE, list[int], list[list[float]]]:
    """Train ACE/ASE on the balance environment.

    Returns:
        ace: trained critic
        ase: trained actor
        episode_lengths: list of episode lengths (for plotting)
        all_angles: list of angle trajectories per episode (for animation)
    """
    num_boxes = 36  # 6 angle bins x 6 velocity bins
    rng = create_rng(seed)
    ace, ase = create_ace_ase(num_boxes)

    episode_lengths: list[int] = []
    all_angles: list[list[float]] = []

    for ep in range(num_episodes):
        episode, angles = train_episode(
            env, ace, ase, rng, num_boxes,
            gamma=gamma, alpha=alpha, beta=beta,
            trace_decay=trace_decay, noise_std=noise_std,
        )
        episode_lengths.append(len(episode))
        all_angles.append(angles)

    return ace, ase, episode_lengths, all_angles
