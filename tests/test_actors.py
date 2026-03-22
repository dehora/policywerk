"""Tests for RL actors."""

from policywerk.world.gridworld import GridWorld
from policywerk.world.balance import Balance
from policywerk.world.random_walk import RandomWalk
from policywerk.actors.bellman import value_iteration, policy_iteration, extract_policy
from policywerk.actors.barto_sutton import (
    create_ace_ase, state_to_input, state_to_box, select_action,
    compute_td_error, update_ace, update_ase, train, NUM_BOXES,
)
from policywerk.actors.td_learner import td_zero, td_lambda, monte_carlo, rms_error
from policywerk.actors.q_learner import q_learning, sarsa, extract_greedy_policy
from policywerk.world.cliffworld import CliffWorld
from policywerk.building_blocks.value_functions import TabularV
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
        td = compute_td_error(ace, x_current, reward=0.0, gamma=0.95, done=False)
        assert abs(td - (-2.15)) < 1e-10

    def test_compute_td_error_terminal(self):
        """On failure (done=True), TD error = reward - prev_prediction."""
        ace, _ = create_ace_ase(36)
        ace.prev_prediction = 2.0
        x = vector.zeros(36)
        # TD error = -1 - 2.0 = -3.0
        td = compute_td_error(ace, x, reward=-1.0, gamma=0.95, done=True)
        assert abs(td - (-3.0)) < 1e-10

    def test_train_improves(self):
        """After training, later episodes should be longer than early ones."""
        env = Balance()
        ace, ase, lengths, _, _ = train(env, num_episodes=150, seed=42)
        # Average of first 20 episodes should be shorter than last 20
        early_avg = sum(lengths[:20]) / 20
        late_avg = sum(lengths[-20:]) / 20
        assert late_avg > early_avg, f"Training did not improve: early={early_avg:.1f}, late={late_avg:.1f}"

    def test_timeout_does_not_penalize(self):
        """A 500-step success should not produce a negative TD error."""
        ace, _ = create_ace_ase(NUM_BOXES)
        ace.weights[15] = 1.0  # some learned value for box 15
        ace.prev_prediction = 1.0

        x = vector.zeros(NUM_BOXES)
        x[15] = 1.0

        # Simulate a timeout success: done=True, env_reward=1.0
        # The train_episode code maps this to learn_done=False (truncation),
        # so compute_td_error should bootstrap, not use terminal path.
        # With reward=0 (paper convention) and done=False:
        # td_error = 0 + gamma * prediction - prev_prediction
        td = compute_td_error(ace, x, reward=0.0, gamma=0.95, done=False)
        # prediction = 1.0, prev = 1.0: td = 0 + 0.95*1.0 - 1.0 = -0.05
        assert abs(td - (-0.05)) < 1e-10
        # This is a small adjustment, NOT the large negative from terminal path

    def test_state_to_box_rejects_wrong_num_boxes(self):
        """state_to_box only works with the 36-box balance discretization."""
        from policywerk.building_blocks.mdp import State
        import pytest
        state = State(features=[0.0, 0.0], label="3,3")
        with pytest.raises(ValueError, match="36-box"):
            state_to_box(state, 100)


class TestTDLearner:
    def test_rms_error(self):
        V = TabularV(default=0.5)
        for label in RandomWalk.LABELS:
            V.set(label, 0.5)
        # All estimates = 0.5, true values = [1/6, 2/6, 3/6, 4/6, 5/6]
        err = rms_error(V, RandomWalk.TRUE_VALUES, RandomWalk.LABELS)
        assert err > 0.0
        # Perfect values should give 0
        for i, label in enumerate(RandomWalk.LABELS):
            V.set(label, RandomWalk.TRUE_VALUES[i])
        assert rms_error(V, RandomWalk.TRUE_VALUES, RandomWalk.LABELS) < 1e-10

    def test_td_zero_converges(self):
        env = RandomWalk()
        V, history = td_zero(env, num_episodes=100, alpha=0.1, seed=42)
        # RMS error should decrease
        early_rms = sum(h["rms"] for h in history[:10]) / 10
        late_rms = sum(h["rms"] for h in history[-10:]) / 10
        assert late_rms < early_rms

    def test_td_zero_values_near_true(self):
        env = RandomWalk()
        V, _ = td_zero(env, num_episodes=200, alpha=0.1, seed=42)
        for i, label in enumerate(RandomWalk.LABELS):
            diff = abs(V.get(label) - RandomWalk.TRUE_VALUES[i])
            assert diff < 0.15, f"V({label})={V.get(label):.3f}, true={RandomWalk.TRUE_VALUES[i]:.3f}"

    def test_td_lambda_converges(self):
        env = RandomWalk()
        V, history = td_lambda(env, num_episodes=100, alpha=0.1, lam=0.5, seed=42)
        early_rms = sum(h["rms"] for h in history[:10]) / 10
        late_rms = sum(h["rms"] for h in history[-10:]) / 10
        assert late_rms < early_rms

    def test_monte_carlo_converges(self):
        env = RandomWalk()
        V, history = monte_carlo(env, num_episodes=100, alpha=0.1, seed=42)
        early_rms = sum(h["rms"] for h in history[:10]) / 10
        late_rms = sum(h["rms"] for h in history[-10:]) / 10
        assert late_rms < early_rms


class TestQLearner:
    def test_q_learning_converges(self):
        """Total reward should improve over training."""
        env = CliffWorld()
        Q, history = q_learning(env, num_episodes=500, seed=42)
        early_reward = sum(h["total_reward"] for h in history[:50]) / 50
        late_reward = sum(h["total_reward"] for h in history[-50:]) / 50
        assert late_reward > early_reward

    def test_q_learning_finds_goal(self):
        """The greedy policy should reach the goal."""
        env = CliffWorld()
        Q, _ = q_learning(env, num_episodes=500, seed=42)
        policy = extract_greedy_policy(Q, env)
        # Follow the greedy policy from start
        state = env.reset()
        for _ in range(100):
            action = policy.get(state.label, 0)
            state, _, done = env.step(action)
            if done:
                assert state.label == "3,11", "Should reach the goal"
                break
        else:
            assert False, "Greedy policy did not reach goal in 100 steps"

    def test_sarsa_converges(self):
        """SARSA total reward should also improve."""
        env = CliffWorld()
        Q, history = sarsa(env, num_episodes=500, seed=42)
        early_reward = sum(h["total_reward"] for h in history[:50]) / 50
        late_reward = sum(h["total_reward"] for h in history[-50:]) / 50
        assert late_reward > early_reward

    def test_greedy_eval_reaches_goal(self):
        """Greedy evaluation of both Q-learning and SARSA should reach the goal."""
        from policywerk.actors.q_learner import eval_greedy
        env = CliffWorld()
        Q_ql, _ = q_learning(env, num_episodes=500, seed=42)
        Q_sa, _ = sarsa(env, num_episodes=500, seed=42)
        policy_ql = extract_greedy_policy(Q_ql, env)
        policy_sa = extract_greedy_policy(Q_sa, env)
        _, ql_reward, ql_done = eval_greedy(policy_ql, CliffWorld())
        _, sa_reward, sa_done = eval_greedy(policy_sa, CliffWorld())
        assert ql_done, "Q-learning greedy eval should reach the goal"
        assert sa_done, "SARSA greedy eval should reach the goal"
        # Q-learning should find optimal or near-optimal path
        assert ql_reward >= -15, f"Q-learning reward {ql_reward} too low"

    def test_q_vs_sarsa_greedy_paths(self):
        """Q-learning's greedy path should be shorter than SARSA's."""
        from policywerk.actors.q_learner import eval_greedy
        env = CliffWorld()
        Q_ql, _ = q_learning(env, num_episodes=500, seed=42)
        Q_sa, _ = sarsa(env, num_episodes=500, seed=42)
        policy_ql = extract_greedy_policy(Q_ql, env)
        policy_sa = extract_greedy_policy(Q_sa, env)
        ql_path, _, _ = eval_greedy(policy_ql, CliffWorld())
        sa_path, _, _ = eval_greedy(policy_sa, CliffWorld())
        # Q-learning finds the optimal (shorter) path
        assert len(ql_path) <= len(sa_path)

    def test_extract_greedy_policy(self):
        """Policy keys should match Q table's visited states."""
        env = CliffWorld()
        Q, _ = q_learning(env, num_episodes=100, seed=42)
        policy = extract_greedy_policy(Q, env)
        # Policy keys should be exactly the state labels in the Q table
        q_labels = {label for label, _action in Q._values.keys()}
        assert set(policy.keys()) == q_labels
        # Start state should have a policy
        assert "3,0" in policy

    def test_extract_greedy_policy_skip_labels(self):
        """skip_labels should exclude those states from the policy."""
        env = CliffWorld()
        Q, _ = q_learning(env, num_episodes=100, seed=42)
        skip = {"3,0", "3,11"}  # skip start and goal
        policy = extract_greedy_policy(Q, env, skip_labels=skip)
        assert "3,0" not in policy
        assert "3,11" not in policy
        # Other visited states should still be present
        assert len(policy) > 0
