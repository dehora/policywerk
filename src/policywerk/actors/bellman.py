"""Level 3: Bellman's dynamic programming algorithms (1957).

Value iteration and policy iteration — two ways to find the optimal
policy when the environment's rules are fully known. Both require
a StochasticMDP (an environment that can answer "what would happen
if I took action A from state S?") and both converge to the same
answer: the optimal value function and the optimal policy.

Value iteration repeatedly asks, for every state: "what's the best
action I could take, considering all possible outcomes?" Each sweep
propagates reward information one step further through the state
space. After enough sweeps, every state knows how good it is.

Policy iteration takes a different approach: start with any policy
(even a random one), figure out exactly how good it is (policy
evaluation), then improve it by switching each state to its best
action (policy improvement). Repeat until the policy stops changing.

The Bellman equation — the recursive relationship between a state's
value and its successors' values — is the mathematical foundation
of both algorithms:

    V(s) = max_a Σ_s' P(s'|s,a) × [R(s,a,s') + γ × V(s')]

"The value of a state is the best action's expected reward plus the
discounted value of wherever that action leads."
"""

from policywerk.primitives import scalar
from policywerk.building_blocks.mdp import StochasticMDP, State
from policywerk.building_blocks.value_functions import TabularV


def value_iteration(
    env: StochasticMDP,
    gamma: float = 0.9,
    theta: float = 0.001,
) -> tuple[TabularV, list[dict]]:
    """Find the optimal value function by repeated Bellman backups.

    Sweeps through every state, updating each to the best action's
    expected value. Converges when the largest change in any state's
    value drops below theta.

    Args:
        env: environment with known transition dynamics.
        gamma: discount factor — how much to value future rewards.
        theta: convergence threshold — stop when changes are this small.

    Returns:
        (V, history) where V is the converged value function and
        history is a list of per-sweep records for visualization.
    """
    V = TabularV(default=0.0)
    all_states = env.states()
    num_actions = env.num_actions()
    history: list[dict] = []

    sweep = 0
    while True:
        sweep += 1
        max_change = 0.0

        # Synchronous update: snapshot current values so every state
        # in this sweep reads from the previous sweep's values.
        old_V = TabularV(default=0.0)
        for k, v in V.all_values().items():
            old_V.set(k, v)

        for state in all_states:
            if env.is_terminal(state):
                continue

            # Bellman optimality backup: try all actions, keep the best
            best_value = float("-inf")
            for action in range(num_actions):
                action_value = 0.0
                for next_state, prob, reward in env.transition_probs(state, action):
                    # action_value += prob * (reward + gamma * old_V[next_state])
                    action_value = scalar.add(
                        action_value,
                        scalar.multiply(prob,
                                        scalar.add(reward,
                                                   scalar.multiply(gamma, old_V.get(next_state.label)))),
                    )
                if action_value > best_value:
                    best_value = action_value

            V.set(state.label, best_value)

            change = scalar.abs_val(scalar.subtract(V.get(state.label), old_V.get(state.label)))
            if change > max_change:
                max_change = change

        history.append({
            "sweep": sweep,
            "max_change": max_change,
            "values": {k: v for k, v in V.all_values().items()},
        })

        if max_change < theta:
            break

    return V, history


def policy_iteration(
    env: StochasticMDP,
    gamma: float = 0.9,
    theta: float = 0.001,
) -> tuple[TabularV, dict[str, int], int]:
    """Find the optimal policy by alternating evaluation and improvement.

    1. Policy evaluation: compute V(s) for the current policy until stable.
    2. Policy improvement: set each state's action to the greedy best.
    3. Repeat until the policy stops changing.

    Args:
        env: environment with known transition dynamics.
        gamma: discount factor.
        theta: convergence threshold for policy evaluation.

    Returns:
        (V, policy, iterations) where V is the value function under the
        optimal policy, policy maps state labels to actions, and iterations
        is how many evaluate/improve cycles were needed.
    """
    all_states = env.states()
    num_actions = env.num_actions()

    # Start with action 0 everywhere
    policy: dict[str, int] = {}
    for state in all_states:
        if not env.is_terminal(state):
            policy[state.label] = 0

    V = TabularV(default=0.0)
    iterations = 0

    while True:
        iterations += 1

        # --- Policy evaluation ---
        # Compute V(s) under the fixed current policy until convergence.
        # Unlike value iteration, we don't take the max over actions —
        # we use the single action prescribed by the current policy.
        while True:
            max_change = 0.0
            for state in all_states:
                if env.is_terminal(state):
                    continue

                old_value = V.get(state.label)
                action = policy[state.label]

                # new_value += prob * (reward + gamma * V[next_state])
                # Unlike value_iteration, we don't try all actions here —
                # we use the single action prescribed by the current policy.
                new_value = 0.0
                for next_state, prob, reward in env.transition_probs(state, action):
                    new_value = scalar.add(
                        new_value,
                        scalar.multiply(prob,
                                        scalar.add(reward,
                                                   scalar.multiply(gamma, V.get(next_state.label)))),
                    )

                V.set(state.label, new_value)
                change = scalar.abs_val(scalar.subtract(new_value, old_value))
                if change > max_change:
                    max_change = change

            if max_change < theta:
                break

        # --- Policy improvement ---
        # For each state, check if there's a better action than the current one.
        policy_stable = True
        for state in all_states:
            if env.is_terminal(state):
                continue

            old_action = policy[state.label]
            best_action = _best_action(env, state, V, gamma, num_actions)

            if best_action != old_action:
                policy[state.label] = best_action
                policy_stable = False

        if policy_stable:
            break

    return V, policy, iterations


def extract_policy(
    env: StochasticMDP,
    V: TabularV,
    gamma: float = 0.9,
) -> dict[str, int]:
    """Derive the greedy policy from a value function.

    For each non-terminal state, pick the action that leads to the
    highest expected value. Returns a dict mapping state labels to
    action indices (0=N, 1=E, 2=S, 3=W for gridworlds).
    """
    num_actions = env.num_actions()
    policy: dict[str, int] = {}

    for state in env.states():
        if env.is_terminal(state):
            continue
        policy[state.label] = _best_action(env, state, V, gamma, num_actions)

    return policy


def _best_action(
    env: StochasticMDP,
    state: State,
    V: TabularV,
    gamma: float,
    num_actions: int,
) -> int:
    """Find the action with highest expected value from this state.

    This action evaluation logic appears in three places:
      - value_iteration: tries all actions, keeps the best value
      - policy_evaluation: uses a single fixed action from the current policy
      - extract_policy / _best_action: finds the best action without modifying V
    Same core calculation, different usage.
    """
    best_a = 0
    best_val = float("-inf")

    for action in range(num_actions):
        action_value = 0.0
        for next_state, prob, reward in env.transition_probs(state, action):
            # action_value += prob * (reward + gamma * V[next_state])
            action_value = scalar.add(
                action_value,
                scalar.multiply(prob,
                                scalar.add(reward,
                                           scalar.multiply(gamma, V.get(next_state.label)))),
            )
        if action_value > best_val:
            best_val = action_value
            best_a = action

    return best_a
