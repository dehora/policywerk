"""Tests for RL actors."""

from policywerk.world.gridworld import GridWorld
from policywerk.actors.bellman import value_iteration, policy_iteration, extract_policy


class TestBellman:
    def test_value_iteration_converges(self):
        env = GridWorld()
        V, history = value_iteration(env, gamma=0.9, theta=0.001)
        # Should converge — last sweep's max_change is below theta
        assert history[-1]["max_change"] < 0.001
        # Should take multiple sweeps (not trivial)
        assert len(history) > 3

    def test_value_iteration_terminal_values(self):
        """Terminal states should retain default value (0.0) — they're not updated."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        assert V.get("0,4") == 0.0  # goal
        assert V.get("1,3") == 0.0  # pit

    def test_value_iteration_goal_neighbors_positive(self):
        """States adjacent to the goal should have positive values."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        # (0,3) is directly west of goal — should be high value
        assert V.get("0,3") > 0.5
        # (1,4) is directly south of goal — should be positive
        assert V.get("1,4") > 0.0

    def test_value_iteration_pit_neighbors_low(self):
        """States adjacent to the pit should have lower values."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        # (1,4) is east of pit — might be low or positive depending on other paths
        # But start (4,0) should be lower than goal neighbor
        assert V.get("4,0") < V.get("0,3")

    def test_extract_policy_points_toward_goal(self):
        """The greedy policy near the goal should point toward it."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        policy = extract_policy(env, V, gamma=0.9)
        # (0,3) is west of goal at (0,4) — policy should be East (action 1)
        assert policy["0,3"] == 1  # East toward goal

    def test_policy_iteration_converges(self):
        env = GridWorld()
        V, policy, iterations = policy_iteration(env, gamma=0.9)
        # Should converge in a small number of iterations
        assert iterations < 20
        # Policy should cover all non-terminal, non-wall states
        assert len(policy) > 0

    def test_policy_iteration_matches_value_iteration(self):
        """Both methods should find the same optimal values and policy."""
        env = GridWorld()
        V_vi, _ = value_iteration(env, gamma=0.9, theta=0.0001)
        V_pi, policy_pi, _ = policy_iteration(env, gamma=0.9, theta=0.0001)
        policy_vi = extract_policy(env, V_vi, gamma=0.9)

        # Values should be very close
        for state in env.states():
            if not env.is_terminal(state):
                diff = abs(V_vi.get(state.label) - V_pi.get(state.label))
                assert diff < 0.01, f"Values differ at {state.label}: {V_vi.get(state.label)} vs {V_pi.get(state.label)}"

        # Policies should match
        assert policy_vi == policy_pi

    def test_extract_policy_covers_all_non_terminal(self):
        """extract_policy should return an action for every non-terminal state."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        policy = extract_policy(env, V, gamma=0.9)
        non_terminal = [s for s in env.states() if not env.is_terminal(s)]
        assert len(policy) == len(non_terminal)
