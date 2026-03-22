"""Tests for RL actors."""

from policywerk.world.gridworld import GridWorld
from policywerk.world.balance import Balance
from policywerk.actors.bellman import value_iteration, policy_iteration, extract_policy
from policywerk.actors.barto_sutton import (
    create_ace_ase, state_to_input, state_to_box, select_action,
    compute_td_error, update_ace, update_ase, train,
)
from policywerk.primitives.random import create_rng
from policywerk.primitives import vector


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


class TestBartoSutton:
    def test_ace_ase_creation(self):
        ace, ase = create_ace_ase(36)
        assert len(ace.weights) == 36
        assert len(ace.traces) == 36
        assert len(ase.weights) == 36
        assert len(ase.traces) == 36
        assert ace.prev_prediction == 0.0

    def test_state_to_input(self):
        """One-hot encoding should have a single 1.0 at the correct index."""
        from policywerk.building_blocks.mdp import State
        state = State(features=[0.05, 0.1], label="2,3")
        x = state_to_input(state, 36)
        assert len(x) == 36
        assert sum(x) == 1.0
        # box = 2*6 + 3 = 15
        assert x[15] == 1.0
        assert x[0] == 0.0

    def test_select_action_returns_binary(self):
        rng = create_rng(42)
        _, ase = create_ace_ase(36)
        x = vector.zeros(36)
        x[0] = 1.0
        for _ in range(20):
            action = select_action(ase, x, rng, noise_std=1.0)
            assert action in (0, 1)

    def test_compute_td_error(self):
        """With known weights, TD error should match hand calculation."""
        ace, _ = create_ace_ase(36)
        # Set weight for box 0 to 5.0, box 1 to 3.0
        ace.weights[0] = 5.0
        ace.weights[1] = 3.0
        ace.prev_prediction = 5.0  # p(t-1)

        x_current = vector.zeros(36)
        x_current[1] = 1.0  # current state is box 1, prediction = 3.0
        x_prev = vector.zeros(36)
        x_prev[0] = 1.0

        # TD error = reward + gamma * p(t) - p(t-1) = 0 + 0.95 * 3.0 - 5.0 = -2.15
        td = compute_td_error(ace, x_current, x_prev, reward=0.0, gamma=0.95, done=False)
        assert abs(td - (-2.15)) < 1e-10

    def test_compute_td_error_terminal(self):
        """On failure (done=True), TD error = reward - prev_prediction."""
        ace, _ = create_ace_ase(36)
        ace.prev_prediction = 2.0
        x = vector.zeros(36)
        # TD error = -1 - 2.0 = -3.0
        td = compute_td_error(ace, x, x, reward=-1.0, gamma=0.95, done=True)
        assert abs(td - (-3.0)) < 1e-10

    def test_train_improves(self):
        """After training, later episodes should be longer than early ones."""
        env = Balance()
        ace, ase, lengths, _ = train(env, num_episodes=150, seed=42)
        # Average of first 20 episodes should be shorter than last 20
        early_avg = sum(lengths[:20]) / 20
        late_avg = sum(lengths[-20:]) / 20
        assert late_avg > early_avg, f"Training did not improve: early={early_avg:.1f}, late={late_avg:.1f}"
