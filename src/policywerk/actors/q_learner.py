"""Level 3: Q-learning and SARSA (Watkins, 1989).

Q-learning extends temporal-difference learning from prediction
(Lesson 03) to control. Instead of learning how good each STATE
is, it learns how good each ACTION is in each state. This lets
the agent decide what to do, not just predict what will happen.

The key idea is off-policy learning: the agent can explore
randomly (epsilon-greedy) while learning the OPTIMAL policy.
The update rule uses the best possible next action (max), not
the action actually taken:

  Q(s, a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

Compare with SARSA (on-policy), which uses the action actually
taken next:

  Q(s, a) += alpha * [r + gamma * Q(s', a_next) - Q(s, a)]

The difference is one word — "max" vs "actual" — but it changes
what the agent learns. On the cliff walking environment:

  Q-learning finds the risky optimal path along the cliff edge
    (shortest route, but epsilon-greedy exploration occasionally
    steps off the cliff during training).

  SARSA finds the safer path away from the cliff (because it
    accounts for its own exploratory behavior when learning).
"""

import random as _random

from policywerk.primitives import scalar
from policywerk.primitives.random import create_rng
from policywerk.building_blocks.mdp import State
from policywerk.building_blocks.value_functions import TabularQ
from policywerk.building_blocks.policies import epsilon_greedy

Vector = list[float]


def q_learning(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42,
) -> tuple[TabularQ, list[dict]]:
    """Learn action-values using Q-learning (off-policy TD control).

    After each step (s, a, r, s'):
      Q(s, a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    The "max" makes this off-policy: the update assumes the agent
    will act optimally from s' onward, even though it actually
    explores with epsilon-greedy.

    Returns (Q, history) where Q is the learned action-value table
    and history contains per-episode records.
    """
    rng = create_rng(seed)
    Q = TabularQ(default=0.0)
    num_actions = env.num_actions()
    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        path: list[tuple[int, int]] = []
        path.append(_label_to_pos(state.label))

        step_count = 0
        while True:
            # Epsilon-greedy action selection
            q_vals = [Q.get(state.label, a) for a in range(num_actions)]
            action = epsilon_greedy(rng, q_vals, epsilon)

            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            if done:
                # Terminal: no future value
                # Q(s,a) += alpha * [r - Q(s,a)]
                td_error = scalar.subtract(reward, Q.get(state.label, action))
            else:
                # Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
                max_next = Q.max_value(next_state.label, num_actions)
                td_error = scalar.subtract(
                    scalar.add(reward, scalar.multiply(gamma, max_next)),
                    Q.get(state.label, action),
                )

            Q.update(state.label, action, scalar.multiply(alpha, td_error))
            path.append(_label_to_pos(next_state.label))

            if done:
                break
            state = next_state

        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "path": path,
            "steps": step_count,
        })

    return Q, history


def sarsa(
    env,
    num_episodes: int = 500,
    alpha: float = 0.5,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    seed: int = 42,
) -> tuple[TabularQ, list[dict]]:
    """Learn action-values using SARSA (on-policy TD control).

    After each step (s, a, r, s', a'):
      Q(s, a) += alpha * [r + gamma * Q(s', a') - Q(s, a)]

    Unlike Q-learning, SARSA uses the action ACTUALLY taken next,
    not the best possible one. This makes it on-policy: what the
    agent learns depends on how it explores.

    On cliff walking, SARSA learns a safer path because it accounts
    for the possibility that epsilon-greedy will step off the cliff.
    """
    rng = create_rng(seed)
    Q = TabularQ(default=0.0)
    num_actions = env.num_actions()
    history: list[dict] = []

    for ep in range(num_episodes):
        state = env.reset()
        q_vals = [Q.get(state.label, a) for a in range(num_actions)]
        action = epsilon_greedy(rng, q_vals, epsilon)
        total_reward = 0.0
        path: list[tuple[int, int]] = []
        path.append(_label_to_pos(state.label))

        step_count = 0
        while True:
            next_state, reward, done = env.step(action)
            total_reward += reward
            step_count += 1

            if done:
                td_error = scalar.subtract(reward, Q.get(state.label, action))
                Q.update(state.label, action, scalar.multiply(alpha, td_error))
                path.append(_label_to_pos(next_state.label))
                break

            # Choose next action BEFORE updating (SARSA needs a')
            q_vals_next = [Q.get(next_state.label, a) for a in range(num_actions)]
            next_action = epsilon_greedy(rng, q_vals_next, epsilon)

            # Q(s,a) += alpha * [r + gamma * Q(s',a') - Q(s,a)]
            td_error = scalar.subtract(
                scalar.add(reward, scalar.multiply(gamma, Q.get(next_state.label, next_action))),
                Q.get(state.label, action),
            )
            Q.update(state.label, action, scalar.multiply(alpha, td_error))
            path.append(_label_to_pos(next_state.label))

            state = next_state
            action = next_action

        history.append({
            "episode": ep,
            "total_reward": total_reward,
            "path": path,
            "steps": step_count,
        })

    return Q, history


def extract_greedy_policy(Q: TabularQ, rows: int, cols: int,
                          num_actions: int) -> dict[str, int]:
    """Derive greedy policy from Q-values for a grid environment.

    Returns a dict mapping "r,c" labels to the best action.
    """
    policy: dict[str, int] = {}
    for r in range(rows):
        for c in range(cols):
            label = f"{r},{c}"
            policy[label] = Q.best_action(label, num_actions)
    return policy


def _label_to_pos(label: str) -> tuple[int, int]:
    """Convert "r,c" label to (row, col) tuple."""
    parts = label.split(",")
    return int(parts[0]), int(parts[1])
