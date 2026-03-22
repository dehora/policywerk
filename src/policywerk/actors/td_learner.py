"""Level 3: Temporal-difference learning (Sutton, 1988).

TD learning is the idea that you can learn to predict without waiting
for the final outcome. Instead of asking "what actually happened?"
(Monte Carlo), TD asks "what do I now predict will happen?" and
updates the old prediction toward the new one.

This is prediction only — the agent follows a fixed random policy
(equal probability of left and right) and learns the value of each
state. The random walk environment has known true values, so we can
measure exactly how accurate the predictions are.

Three methods are implemented:

  TD(0): after each step, update V(s) toward r + gamma * V(s').
    Bootstraps from the next state's estimate. Low variance because
    it uses only one step of actual experience, but biased because
    it trusts its own (possibly wrong) estimate of V(s').

  TD(lambda): after each step, update all recently visited states
    using eligibility traces. lambda=0 gives TD(0). lambda=1 gives
    Monte Carlo-like behavior. Values in between trade off bias
    against variance.

  Monte Carlo: after each complete episode, update every visited
    state toward the actual return G. Unbiased (uses the real
    outcome) but high variance (different episodes from the same
    state can give very different returns).

The TD error — r + gamma * V(s') - V(s) — is the same "surprise
signal" that drove the ACE critic in Lesson 02. This lesson
isolates it and shows why it works.
"""

import math
import random as _random

from policywerk.primitives import scalar
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.mdp import State, Transition, Episode
from policywerk.building_blocks.value_functions import TabularV
from policywerk.building_blocks.traces import EligibilityTrace
from policywerk.building_blocks.returns import discount_return
from policywerk.world.random_walk import RandomWalk

Vector = list[float]


def rms_error(V: TabularV, true_values: Vector, labels: list[str]) -> float:
    """Root mean squared error between estimated and true values.

    Measures how far off the predictions are on average.
    """
    total = 0.0
    for i, label in enumerate(labels):
        diff = scalar.subtract(V.get(label), true_values[i])
        total = scalar.add(total, scalar.multiply(diff, diff))
    mean = scalar.multiply(total, scalar.inverse(len(labels)))
    return math.sqrt(mean)


def td_zero(
    env: RandomWalk,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0,
    seed: int = 42,
    init_value: float = 0.5,
) -> tuple[TabularV, list[dict]]:
    """Learn state values using TD(0) — single-step bootstrapping.

    After each step (s, r, s'), update:
      V(s) += alpha * [r + gamma * V(s') - V(s)]

    The agent follows a random policy (equal probability left/right).
    """
    rng = create_rng(seed)
    V = TabularV(default=init_value)
    # Initialize all states to init_value
    for label in RandomWalk.LABELS:
        V.set(label, init_value)

    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()

        while True:
            # Random policy: equal probability left/right
            action = rng.randint(0, 1)
            next_state, reward, done = env.step(action)

            if done:
                # Terminal state has value 0 (no future reward)
                # td_error = reward + gamma * 0 - V(s) = reward - V(s)
                td_error = scalar.subtract(reward, V.get(state.label))
            else:
                # td_error = reward + gamma * V(s') - V(s)
                td_error = scalar.subtract(
                    scalar.add(reward, scalar.multiply(gamma, V.get(next_state.label))),
                    V.get(state.label),
                )

            # Update: V(s) += alpha * td_error
            V.update(state.label, scalar.multiply(alpha, td_error))

            if done:
                break
            state = next_state

        # Record history for this episode
        values = [V.get(label) for label in RandomWalk.LABELS]
        history.append({
            "episode": ep,
            "values": list(values),
            "rms": rms_error(V, RandomWalk.TRUE_VALUES, RandomWalk.LABELS),
        })

    return V, history


def td_lambda(
    env: RandomWalk,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.5,
    seed: int = 42,
    init_value: float = 0.5,
) -> tuple[TabularV, list[dict]]:
    """Learn state values using TD(lambda) with eligibility traces.

    After each step:
      delta = r + gamma * V(s') - V(s)
      e(s) += 1  (accumulating trace for visited state)
      For all states with nonzero trace:
        V(s) += alpha * delta * e(s)
        e(s) *= gamma * lambda  (decay trace)
    """
    rng = create_rng(seed)
    V = TabularV(default=init_value)
    for label in RandomWalk.LABELS:
        V.set(label, init_value)

    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        trace = EligibilityTrace(gamma=gamma, lam=lam)

        while True:
            action = rng.randint(0, 1)
            next_state, reward, done = env.step(action)

            if done:
                td_error = scalar.subtract(reward, V.get(state.label))
            else:
                td_error = scalar.subtract(
                    scalar.add(reward, scalar.multiply(gamma, V.get(next_state.label))),
                    V.get(state.label),
                )

            # Update trace for current state
            trace.visit(state.label)

            # Update all states with nonzero traces
            for label, e in trace.all_traces().items():
                if abs(e) > 1e-10:
                    V.update(label, scalar.multiply(alpha, scalar.multiply(td_error, e)))

            # Decay all traces
            trace.decay()

            if done:
                break
            state = next_state

        values = [V.get(label) for label in RandomWalk.LABELS]
        history.append({
            "episode": ep,
            "values": list(values),
            "rms": rms_error(V, RandomWalk.TRUE_VALUES, RandomWalk.LABELS),
        })

    return V, history


def monte_carlo(
    env: RandomWalk,
    num_episodes: int = 100,
    alpha: float = 0.1,
    gamma: float = 1.0,
    seed: int = 42,
    init_value: float = 0.5,
) -> tuple[TabularV, list[dict]]:
    """Learn state values using first-visit Monte Carlo.

    After each episode, compute the actual return G for each visited
    state, then update:
      V(s) += alpha * [G - V(s)]

    Waits for the episode to end before updating — no bootstrapping.
    """
    rng = create_rng(seed)
    V = TabularV(default=init_value)
    for label in RandomWalk.LABELS:
        V.set(label, init_value)

    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        # Collect the episode
        states_visited: list[str] = []
        rewards: list[float] = []

        while True:
            states_visited.append(state.label)
            action = rng.randint(0, 1)
            next_state, reward, done = env.step(action)
            rewards.append(reward)

            if done:
                break
            state = next_state

        # Compute returns from each visited state (first-visit)
        G = 0.0
        visited_this_episode: set[str] = set()
        for t in range(len(states_visited) - 1, -1, -1):
            G = scalar.add(rewards[t], scalar.multiply(gamma, G))
            label = states_visited[t]
            if label not in visited_this_episode:
                visited_this_episode.add(label)
                # V(s) += alpha * [G - V(s)]
                error = scalar.subtract(G, V.get(label))
                V.update(label, scalar.multiply(alpha, error))

        values = [V.get(label) for label in RandomWalk.LABELS]
        history.append({
            "episode": ep,
            "values": list(values),
            "rms": rms_error(V, RandomWalk.TRUE_VALUES, RandomWalk.LABELS),
        })

    return V, history
