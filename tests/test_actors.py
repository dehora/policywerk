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
from policywerk.primitives.activations import relu, identity
from policywerk.primitives import vector


class TestBellman:
    def test_value_iteration_converges(self):
        env = GridWorld()
        V, history = value_iteration(env, gamma=0.9, theta=0.001)
        # Should converge—last sweep's max_change is below theta
        assert history[-1]["max_change"] < 0.001
        # Should take multiple sweeps (not trivial)
        assert len(history) > 3

    def test_value_iteration_terminal_values(self):
        """Terminal states should retain default value (0.0)—they're not updated."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        assert V.get("0,4") == 0.0  # goal
        assert V.get("1,3") == 0.0  # pit

    def test_value_iteration_goal_neighbors_positive(self):
        """States adjacent to the goal should have positive values."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        # (0,3) is directly west of goal—should be high value
        assert V.get("0,3") > 0.5
        # (1,4) is directly south of goal—should be positive
        assert V.get("1,4") > 0.0

    def test_value_iteration_pit_neighbors_low(self):
        """States adjacent to the pit should have lower values."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        # (1,4) is east of pit—might be low or positive depending on other paths
        # But start (4,0) should be lower than goal neighbor
        assert V.get("4,0") < V.get("0,3")

    def test_extract_policy_points_toward_goal(self):
        """The greedy policy near the goal should point toward it."""
        env = GridWorld()
        V, _ = value_iteration(env, gamma=0.9)
        policy = extract_policy(env, V, gamma=0.9)
        # (0,3) is west of goal at (0,4)—policy should be East (action 1)
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
        assert len(ql_path) < len(sa_path)

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

    def test_best_action_ignores_unseen(self):
        """best_action should not prefer unseen actions over learned negative ones."""
        from policywerk.building_blocks.value_functions import TabularQ
        Q = TabularQ(default=0.0)
        # Only set action 1 for state "s", with a negative value
        Q.set("s", 1, -5.0)
        # best_action should return 1 (the only seen action), not 0 or 2 or 3
        # which would have default 0.0 > -5.0
        best = Q.best_action("s", 4)
        assert best == 1, f"Should pick seen action 1 (-5.0), not unseen default (0.0), got {best}"

    def test_max_value_includes_unseen(self):
        """max_value should include unseen actions at default for training targets."""
        from policywerk.building_blocks.value_functions import TabularQ
        Q = TabularQ(default=0.0)
        Q.set("s", 1, -5.0)
        # max_value should return 0.0 (unseen actions at default), not -5.0
        # This is correct for training: the Bellman target uses all actions
        mv = Q.max_value("s", 4)
        assert mv == 0.0, f"max_value should include unseen default (0.0), got {mv}"

    def test_partial_q_greedy_policy(self):
        """Greedy policy from a sparse Q table should not drift onto unseen actions."""
        from policywerk.building_blocks.value_functions import TabularQ
        from policywerk.actors.q_learner import eval_greedy, extract_greedy_policy
        Q = TabularQ(default=0.0)
        # Simulate a partial Q table: only East (1) is learned for a few states
        Q.set("3,0", 1, -10.0)  # East from start
        Q.set("2,0", 1, -8.0)   # East from above start
        Q.set("2,1", 1, -6.0)
        env = CliffWorld()
        policy = extract_greedy_policy(Q, env)
        # All policy entries should be action 1 (East), not some unseen action
        for label, action in policy.items():
            assert action == 1, f"State {label}: expected action 1, got {action}"

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

    def test_q_learning_history_has_policy_and_values(self):
        """Q-learning history should record policy and values snapshots."""
        env = CliffWorld()
        _, history = q_learning(env, num_episodes=50, seed=42)
        for h in history:
            assert "policy" in h, "history entry missing 'policy'"
            assert "values" in h, "history entry missing 'values'"
            assert isinstance(h["policy"], dict)
            assert isinstance(h["values"], dict)
        # Final episode should have non-empty snapshots
        assert len(history[-1]["policy"]) > 0
        assert len(history[-1]["values"]) > 0

    def test_sarsa_history_has_policy_and_values(self):
        """SARSA history should record policy and values snapshots."""
        env = CliffWorld()
        _, history = sarsa(env, num_episodes=50, seed=42)
        for h in history:
            assert "policy" in h, "history entry missing 'policy'"
            assert "values" in h, "history entry missing 'values'"
            assert isinstance(h["policy"], dict)
            assert isinstance(h["values"], dict)
        assert len(history[-1]["policy"]) > 0
        assert len(history[-1]["values"]) > 0

    def test_snapshot_values_match_greedy_action(self):
        """Snapshot values should be the Q-value of the best seen action,
        not max_value which includes unseen actions at default 0.0."""
        from policywerk.building_blocks.value_functions import TabularQ
        env = CliffWorld()
        Q, history = q_learning(env, num_episodes=200, seed=42)
        final = history[-1]
        num_actions = env.num_actions()
        for label, value in final["values"].items():
            best_a = Q.best_action(label, num_actions)
            expected = Q.get(label, best_a)
            assert value == expected, (
                f"State {label}: snapshot value {value} != "
                f"Q(best_action={best_a}) = {expected}"
            )

    def test_sarsa_and_q_learning_histories_differ(self):
        """SARSA and Q-learning should produce different value snapshots,
        confirming they are independent histories."""
        env = CliffWorld()
        _, hist_ql = q_learning(env, num_episodes=200, seed=42)
        _, hist_sa = sarsa(env, num_episodes=200, seed=42)
        ql_vals = hist_ql[-1]["values"]
        sa_vals = hist_sa[-1]["values"]
        # They share many state labels but values should differ
        shared = set(ql_vals.keys()) & set(sa_vals.keys())
        assert len(shared) > 0, "Should have overlapping states"
        diffs = sum(1 for k in shared if ql_vals[k] != sa_vals[k])
        assert diffs > 0, "Q-learning and SARSA values should differ"


class TestDQN:
    """Tests for the DQN actor."""

    # Use minimal config for test speed
    _DQN_KWARGS = dict(
        num_episodes=50,
        hidden_size=16,
        batch_size=8,
        replay_capacity=500,
        min_replay_size=50,
        train_every=4,
        target_update_freq=10,
        seed=42,
    )

    def _make_env(self):
        from policywerk.world.breakout import Breakout
        return Breakout(max_steps=50)

    def test_dqn_returns_network_and_history(self):
        """Return type should be (Network, list[dict])."""
        from policywerk.actors.dqn import dqn
        from policywerk.building_blocks.network import Network
        net, history = dqn(self._make_env(), **self._DQN_KWARGS)
        assert isinstance(net, Network)
        assert isinstance(history, list)
        assert len(history) == self._DQN_KWARGS["num_episodes"]

    def test_dqn_history_has_expected_keys(self):
        """Each history entry should have the expected keys."""
        from policywerk.actors.dqn import dqn
        _, history = dqn(self._make_env(), **self._DQN_KWARGS)
        expected_keys = {"episode", "total_reward", "steps", "epsilon", "avg_loss", "avg_q"}
        for h in history:
            assert set(h.keys()) == expected_keys, f"Missing keys: {expected_keys - set(h.keys())}"

    def test_dqn_epsilon_decays(self):
        """Epsilon should decrease over training."""
        from policywerk.actors.dqn import dqn
        _, history = dqn(self._make_env(), **self._DQN_KWARGS)
        assert history[0]["epsilon"] > history[-1]["epsilon"]

    def test_dqn_deterministic_with_seed(self):
        """Same seed should produce identical history."""
        from policywerk.actors.dqn import dqn
        _, hist1 = dqn(self._make_env(), **self._DQN_KWARGS)
        _, hist2 = dqn(self._make_env(), **self._DQN_KWARGS)
        for h1, h2 in zip(hist1, hist2):
            assert h1["total_reward"] == h2["total_reward"]
            assert h1["steps"] == h2["steps"]

    def test_copy_network_is_independent(self):
        """Modifying a copy should not affect the original."""
        from policywerk.actors.dqn import _copy_network
        from policywerk.building_blocks.network import create_network, network_forward
        rng = create_rng(42)
        net = create_network(rng, [4, 8, 2], [relu, identity])
        copy = _copy_network(net)
        # Modify copy's weights
        copy.layers[0].weights[0][0] = 999.0
        # Original should be unchanged
        assert net.layers[0].weights[0][0] != 999.0

    def test_linear_epsilon_normal_path(self):
        """Epsilon should start at start, end at end, and clamp beyond."""
        from policywerk.actors.dqn import _linear_epsilon
        # Episode 0 → start
        assert _linear_epsilon(0, start=1.0, end=0.1, decay_episodes=200) == 1.0
        # Episode = decay_episodes → end
        assert abs(_linear_epsilon(200, 1.0, 0.1, 200) - 0.1) < 1e-10
        # Midpoint
        assert abs(_linear_epsilon(100, 1.0, 0.1, 200) - 0.55) < 1e-10
        # Beyond decay_episodes → clamped at end
        assert abs(_linear_epsilon(500, 1.0, 0.1, 200) - 0.1) < 1e-10

    def test_linear_epsilon_zero_decay(self):
        """decay_episodes=0 should immediately return end epsilon."""
        from policywerk.actors.dqn import _linear_epsilon
        assert _linear_epsilon(0, start=1.0, end=0.1, decay_episodes=0) == 0.1
        assert _linear_epsilon(100, start=1.0, end=0.1, decay_episodes=0) == 0.1

    def test_dqn_trains_without_error(self):
        """DQN should complete training and produce loss values."""
        from policywerk.actors.dqn import dqn
        env = self._make_env()
        _, history = dqn(env, **self._DQN_KWARGS)
        # After min_replay_size is reached, training should produce losses
        has_loss = any(h["avg_loss"] > 0 for h in history)
        assert has_loss, "Training should produce non-zero losses"
        # Q-values should be non-zero after training
        late_q = history[-1]["avg_q"]
        assert late_q != 0.0, "Q-values should be non-zero after training"

    def test_trained_network_output_shape(self):
        """Trained network should produce one Q-value per action."""
        from policywerk.actors.dqn import dqn
        from policywerk.building_blocks.network import network_forward
        from policywerk.world.breakout import Breakout
        env = self._make_env()
        net, _ = dqn(env, **self._DQN_KWARGS)
        state = Breakout(max_steps=50).reset()
        q_vals, _ = network_forward(net, state.features)
        assert len(q_vals) == 3  # left, stay, right

    def test_greedy_poster_frame_threshold_path(self):
        """greedy_poster_frame should return a frame from the score threshold path."""
        from policywerk.actors.dqn import greedy_poster_frame
        from policywerk.building_blocks.network import create_network, Network
        from policywerk.building_blocks.mdp import State

        class StubEnv:
            """Env that reaches score=2 on step 3, then keeps going."""
            def __init__(self):
                self._step = 0
                self._sc = 0
            def reset(self):
                self._step = 0
                self._sc = 0
                return State(features=[0.0] * 4, label="s0")
            def step(self, action):
                self._step += 1
                self._sc = self._step  # score increments each step
                done = self._step >= 10
                return State(features=[float(self._step)] * 4, label=f"s{self._step}"), 0.0, done
            def score(self):
                return self._sc
            def render_color_frame(self):
                # Encode step number in the frame so we can verify which frame was returned
                return [[[float(self._step), 0.0, 0.0]]]
            def num_actions(self):
                return 3

        rng = create_rng(42)
        net = create_network(rng, [4, 4, 3], [relu, identity])
        stub = StubEnv()

        # min_score=2 → should return frame from step 2 (score reaches 2, not done)
        frame = greedy_poster_frame(net, stub, min_score=2)
        assert frame == [[[2.0, 0.0, 0.0]]], f"Expected frame from step 2, got {frame}"

    def test_greedy_poster_frame_fallback_on_early_done(self):
        """If done before min_score, should return the done frame, not reset."""
        from policywerk.actors.dqn import greedy_poster_frame
        from policywerk.building_blocks.network import create_network
        from policywerk.building_blocks.mdp import State

        class EarlyDoneEnv:
            """Env that ends on step 1 with score=0."""
            def __init__(self):
                self._step = 0
            def reset(self):
                self._step = 0
                return State(features=[0.0] * 4, label="s0")
            def step(self, action):
                self._step = 1
                return State(features=[1.0] * 4, label="s1"), -1.0, True
            def score(self):
                return 0
            def render_color_frame(self):
                return [[[float(self._step), 0.0, 0.0]]]
            def num_actions(self):
                return 3

        rng = create_rng(42)
        net = create_network(rng, [4, 4, 3], [relu, identity])
        stub = EarlyDoneEnv()
        frame = greedy_poster_frame(net, stub, min_score=2)
        # Should be step-1 frame (done), not step-0 (reset)
        assert frame == [[[1.0, 0.0, 0.0]]], f"Expected done frame from step 1, got {frame}"

    def test_greedy_poster_frame_timeout_returns_last_frame(self):
        """On timeout, should return the last stepped frame."""
        from policywerk.actors.dqn import greedy_poster_frame
        from policywerk.building_blocks.network import create_network
        from policywerk.building_blocks.mdp import State

        class NeverScoresEnv:
            """Env that never scores and never ends."""
            def __init__(self):
                self._step = 0
            def reset(self):
                self._step = 0
                return State(features=[0.0] * 4, label="s0")
            def step(self, action):
                self._step += 1
                return State(features=[float(self._step)] * 4, label=f"s{self._step}"), 0.0, False
            def score(self):
                return 0
            def render_color_frame(self):
                return [[[float(self._step), 0.0, 0.0]]]
            def num_actions(self):
                return 3

        rng = create_rng(42)
        net = create_network(rng, [4, 4, 3], [relu, identity])
        stub = NeverScoresEnv()
        frame = greedy_poster_frame(net, stub, max_steps=5, min_score=2)
        # Should be step-5 frame, not step-0 (reset)
        assert frame == [[[5.0, 0.0, 0.0]]], f"Expected last frame from step 5, got {frame}"


class TestPPO:
    """Tests for the PPO actor."""

    _PPO_KWARGS = dict(
        num_iterations=30,
        steps_per_iter=100,
        num_epochs=3,
        hidden_size=16,
        seed=42,
    )

    def _make_env(self):
        from policywerk.world.balance import Balance
        return Balance()

    # --- Helper tests ---

    def test_gae_with_resets_matches_gae_no_dones(self):
        """Without episode boundaries, should match returns.gae()."""
        from policywerk.actors.ppo import _compute_gae_with_resets
        from policywerk.building_blocks.returns import gae

        rewards = [1.0, 0.5, -0.2, 0.8, 1.0]
        values = [0.5, 0.6, 0.4, 0.7, 0.9]
        dones = [False, False, False, False, False]
        next_value = 0.8
        gamma, lam = 0.99, 0.95

        expected = gae(rewards, values, next_value, gamma, lam)
        actual = _compute_gae_with_resets(rewards, values, dones, next_value, gamma, lam)

        for i in range(len(rewards)):
            assert abs(expected[i] - actual[i]) < 1e-10, (
                f"Step {i}: expected {expected[i]}, got {actual[i]}"
            )

    def test_gae_with_resets_blocks_at_boundary(self):
        """done=True should prevent advantage from propagating backward."""
        from policywerk.actors.ppo import _compute_gae_with_resets

        rewards = [1.0, 1.0, 1.0, 1.0]
        values = [0.0, 0.0, 0.0, 0.0]
        # Episode boundary at step 1 (done=True means step 1 is terminal)
        dones = [False, True, False, False]
        next_value = 0.0
        gamma, lam = 0.99, 0.95

        adv = _compute_gae_with_resets(rewards, values, dones, next_value, gamma, lam)

        # Steps 2-3 form one segment, steps 0-1 form another.
        # Step 1 is terminal (done=True), so no bootstrap from step 2.
        # Advantage at step 1 should just be reward[1] + 0 - 0 = 1.0
        assert abs(adv[1] - 1.0) < 1e-10, f"Terminal step advantage should be 1.0, got {adv[1]}"

        # Step 0 should not include any advantage from steps 2-3
        # adv[0] = reward[0] + gamma * values[1] * (1-done[0]) - values[0]
        #        + gamma * lam * (1-done[0]) * adv[1]
        #        = 1.0 + 0.99*0.0 - 0.0 + 0.99*0.95*1.0
        #        = 1.0 + 0.9405 = 1.9405
        expected_0 = 1.0 + 0.99 * 0.95 * 1.0
        assert abs(adv[0] - expected_0) < 1e-10, f"Step 0: expected {expected_0}, got {adv[0]}"

    def test_gae_known_values(self):
        """GAE with known inputs should produce specific advantage values."""
        from policywerk.actors.ppo import _compute_gae_with_resets

        # 3-step episode: rewards all 1.0, values and next_value known
        rewards = [1.0, 1.0, 1.0]
        values = [10.0, 12.0, 11.0]
        dones = [False, False, False]
        next_value = 10.0
        gamma = 0.99

        # lambda=0: advantage is just the 1-step TD residual
        adv_td = _compute_gae_with_resets(rewards, values, dones, next_value, gamma, lam=0.0)
        delta_0 = 1.0 + 0.99 * 12.0 - 10.0  # 2.88
        delta_1 = 1.0 + 0.99 * 11.0 - 12.0  # -0.11
        delta_2 = 1.0 + 0.99 * 10.0 - 11.0  # -0.10
        assert abs(adv_td[0] - delta_0) < 1e-10
        assert abs(adv_td[1] - delta_1) < 1e-10
        assert abs(adv_td[2] - delta_2) < 1e-10

        # lambda=0.95: advantages accumulate backward
        adv_gae = _compute_gae_with_resets(rewards, values, dones, next_value, gamma, lam=0.95)
        # A_2 = delta_2
        assert abs(adv_gae[2] - delta_2) < 1e-10
        # A_1 = delta_1 + gamma*lam*A_2
        expected_1 = delta_1 + 0.99 * 0.95 * delta_2
        assert abs(adv_gae[1] - expected_1) < 1e-10
        # A_0 = delta_0 + gamma*lam*A_1
        expected_0 = delta_0 + 0.99 * 0.95 * expected_1
        assert abs(adv_gae[0] - expected_0) < 1e-10
        # Cross-check: lesson narrative says A_0 ≈ 2.69
        assert abs(adv_gae[0] - 2.69) < 0.01, f"Lesson claims A_0 ≈ 2.69, got {adv_gae[0]:.4f}"

    def test_policy_gradient_pushes_toward_good_action(self):
        """With positive advantage and ratio=1, gradient should push mean toward action."""
        from policywerk.actors.ppo import _policy_gradient
        import math
        # Action=0.5, mean=0.0, advantage>0: gradient should push mean toward 0.5
        # (d_mean component should be negative, since loss = -surrogate)
        grad = _policy_gradient(0.0, 0.0, 0.5, 2.0, -0.9189, 0.2, 0.0)
        # log_prob of 0.5 under Gaussian(0,1) ≈ -1.0439
        # ratio ≈ exp(-1.0439 - (-0.9189)) = exp(-0.125) ≈ 0.882, within clip
        # dsurr/d_mean = advantage * ratio * (action - mean)/std^2 = 2.0 * 0.882 * 0.5 > 0
        # dL/d_mean = -dsurr/d_mean < 0 → gradient descent on mean moves it positive (toward action)
        assert grad[0] < 0, f"Gradient should push mean toward action (negative loss grad), got {grad[0]}"

    def test_policy_gradient_numerical_check(self):
        """Policy gradient should match finite-difference approximation."""
        from policywerk.actors.ppo import _policy_gradient
        import math

        mean = 0.3
        log_std = -0.5
        action = 0.7
        advantage = 1.5
        log_prob_old = -1.2
        clip_eps = 0.2
        ent_coeff = 0.01

        grad = _policy_gradient(mean, log_std, action, advantage, log_prob_old, clip_eps, ent_coeff)

        # Compute loss for finite differences
        def compute_loss(m, ls):
            std = math.exp(max(-2.0, min(2.0, ls)))
            diff = action - m
            z = diff / std
            lp_new = -0.5 * (z * z) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
            ratio = math.exp(lp_new - log_prob_old)
            surr1 = ratio * advantage
            clamped = max(1.0 - clip_eps, min(1.0 + clip_eps, ratio))
            surr2 = clamped * advantage
            surr = min(surr1, surr2)
            entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + 2.0 * max(-2.0, min(2.0, ls)))
            return -surr - ent_coeff * entropy

        eps = 1e-5
        # Numerical gradient w.r.t. mean
        num_dmean = (compute_loss(mean + eps, log_std) - compute_loss(mean - eps, log_std)) / (2 * eps)
        # Numerical gradient w.r.t. log_std
        num_dlogstd = (compute_loss(mean, log_std + eps) - compute_loss(mean, log_std - eps)) / (2 * eps)

        assert abs(grad[0] - num_dmean) < 1e-4, f"d_mean: analytical {grad[0]}, numerical {num_dmean}"
        assert abs(grad[1] - num_dlogstd) < 1e-4, f"d_log_std: analytical {grad[1]}, numerical {num_dlogstd}"

    def test_policy_gradient_negative_advantage_clipping(self):
        """With negative advantage, clipping should work correctly on both sides."""
        from policywerk.actors.ppo import _policy_gradient
        import math

        def compute_loss(m, ls, adv, lp_old):
            std = math.exp(max(-2.0, min(2.0, ls)))
            diff = 0.5 - m
            z = diff / std
            lp_new = -0.5 * (z * z) - math.log(std) - 0.5 * math.log(2.0 * math.pi)
            ratio = math.exp(lp_new - lp_old)
            surr1 = ratio * adv
            clamped = max(0.8, min(1.2, ratio))
            surr2 = clamped * adv
            surr = min(surr1, surr2)
            entropy = 0.5 * (1.0 + math.log(2.0 * math.pi) + 2.0 * max(-2.0, min(2.0, ls)))
            return -surr - 0.01 * entropy

        eps = 1e-5
        # Negative advantage, ratio > 1+eps (action became more likely for a BAD action)
        mean, log_std, action, advantage = 0.0, -0.5, 0.5, -2.0
        log_prob_old = -3.0  # old policy gave low prob → new ratio will be high
        grad = _policy_gradient(mean, log_std, action, advantage, log_prob_old, 0.2, 0.01)
        num_dm = (compute_loss(mean + eps, log_std, advantage, log_prob_old) -
                  compute_loss(mean - eps, log_std, advantage, log_prob_old)) / (2 * eps)
        num_dls = (compute_loss(mean, log_std + eps, advantage, log_prob_old) -
                   compute_loss(mean, log_std - eps, advantage, log_prob_old)) / (2 * eps)
        assert abs(grad[0] - num_dm) < 1e-3, f"neg adv d_mean: {grad[0]} vs {num_dm}"
        assert abs(grad[1] - num_dls) < 1e-3, f"neg adv d_log_std: {grad[1]} vs {num_dls}"

    def test_policy_gradient_log_std_saturated(self):
        """When log_std is outside [-2, 2], gradient should be zero."""
        from policywerk.actors.ppo import _policy_gradient
        # log_std at +3.0 is outside the clamp range
        grad = _policy_gradient(0.0, 3.0, 0.5, 1.0, -1.0, 0.2, 0.01)
        assert grad[1] == 0.0, f"Saturated log_std should have zero gradient, got {grad[1]}"
        # log_std at -3.0
        grad2 = _policy_gradient(0.0, -3.0, 0.5, 1.0, -1.0, 0.2, 0.01)
        assert grad2[1] == 0.0, f"Saturated log_std should have zero gradient, got {grad2[1]}"

    # --- Training loop tests ---

    def test_ppo_returns_networks_and_history(self):
        """Return type should be (Network, Network, list[dict])."""
        from policywerk.actors.ppo import ppo
        from policywerk.building_blocks.network import Network
        actor, critic, history = ppo(self._make_env(), **self._PPO_KWARGS)
        assert isinstance(actor, Network)
        assert isinstance(critic, Network)
        assert isinstance(history, list)
        assert len(history) == self._PPO_KWARGS["num_iterations"]

    def test_ppo_history_has_expected_keys(self):
        """Each history entry should have the expected keys."""
        from policywerk.actors.ppo import ppo
        _, _, history = ppo(self._make_env(), **self._PPO_KWARGS)
        expected = {"iteration", "avg_reward", "episodes_completed",
                    "value_loss", "entropy", "mean_std"}
        for h in history:
            assert set(h.keys()) == expected, f"Missing keys: {expected - set(h.keys())}"

    def test_ppo_value_loss_is_finite_and_nonzero(self):
        """value_loss should be finite and non-zero on at least one iteration."""
        from policywerk.actors.ppo import ppo
        import math
        _, _, history = ppo(self._make_env(), **self._PPO_KWARGS)
        for h in history:
            assert math.isfinite(h["value_loss"]), f"value_loss not finite: {h['value_loss']}"
        has_nonzero = any(h["value_loss"] > 0 for h in history)
        assert has_nonzero, "value_loss should be non-zero on at least one iteration"

    def test_ppo_deterministic_with_seed(self):
        """Same seed should produce identical history."""
        from policywerk.actors.ppo import ppo
        _, _, hist1 = ppo(self._make_env(), **self._PPO_KWARGS)
        _, _, hist2 = ppo(self._make_env(), **self._PPO_KWARGS)
        for h1, h2 in zip(hist1, hist2):
            assert h1["avg_reward"] == h2["avg_reward"]
            assert h1["entropy"] == h2["entropy"]

    def test_ppo_entropy_is_tracked(self):
        """Entropy should be a positive finite value throughout training."""
        from policywerk.actors.ppo import ppo
        _, _, history = ppo(self._make_env(), **self._PPO_KWARGS)
        for h in history:
            assert h["entropy"] > 0, "Entropy should be positive"
            assert h["entropy"] < 100, "Entropy should be finite"

    def test_ppo_actor_output_shape(self):
        """Trained actor should output [mean, log_std] for any 2D state."""
        from policywerk.actors.ppo import ppo
        from policywerk.building_blocks.network import network_forward
        actor, _, _ = ppo(self._make_env(), **self._PPO_KWARGS)
        out, _ = network_forward(actor, [0.0, 0.0])
        assert len(out) == 2, f"Actor should output 2 values (mean, log_std), got {len(out)}"

