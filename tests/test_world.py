"""Tests for world environments."""

from policywerk.world.random_walk import RandomWalk
from policywerk.world.gridworld import GridWorld
from policywerk.world.cliffworld import CliffWorld
from policywerk.world.balance import Balance
from policywerk.world.catcher import Catcher
from policywerk.world.breakout import Breakout, ROWS, COLS, BRICK, BALL, PADDLE
from policywerk.world.pointmass import PointMass
from policywerk.world.pixel_pointmass import PixelPointMass


class TestRandomWalk:
    def test_reset(self):
        env = RandomWalk()
        state = env.reset()
        assert state.label == "C"
        assert state.features == [2.0]

    def test_walk_right_to_terminal(self):
        env = RandomWalk()
        env.reset()
        # Walk right: C -> D -> E -> terminal
        s, r, done = env.step(1)
        assert s.label == "D" and not done
        s, r, done = env.step(1)
        assert s.label == "E" and not done
        s, r, done = env.step(1)
        assert done and r == 1.0

    def test_walk_left_to_terminal(self):
        env = RandomWalk()
        env.reset()
        # Walk left: C -> B -> A -> terminal
        s, r, done = env.step(0)
        assert s.label == "B"
        s, r, done = env.step(0)
        assert s.label == "A"
        s, r, done = env.step(0)
        assert done and r == 0.0

    def test_true_values(self):
        assert len(RandomWalk.TRUE_VALUES) == 5
        assert abs(RandomWalk.TRUE_VALUES[2] - 0.5) < 1e-10

    def test_num_actions(self):
        assert RandomWalk().num_actions() == 2


class TestGridWorld:
    def test_reset(self):
        env = GridWorld()
        state = env.reset()
        assert state.label == "4,0"

    def test_goal_terminal(self):
        env = GridWorld()
        env.reset()
        # Navigate to goal at (0,4) — move north 4 times, east 4 times
        for _ in range(4):
            env.step(0)  # N
        for _ in range(3):
            env.step(1)  # E
        s, r, done = env.step(1)  # E to (0,4)
        assert done and r == 1.0

    def test_wall_blocks(self):
        env = GridWorld()
        env.reset()
        # From start (4,0), go north to (3,0), then to (2,0), (1,0)
        env.step(0)  # to (3,0)
        env.step(0)  # to (2,0)
        env.step(0)  # to (1,0)
        # Try to go east into wall at (1,1) — should stay at (1,0)
        s, _, _ = env.step(1)
        assert s.label == "1,0"

    def test_states_complete(self):
        env = GridWorld()
        states = env.states()
        # 25 cells - 1 wall = 24 non-wall states (including goal and pit)
        assert len(states) == 24

    def test_transition_probs_sum_to_one(self):
        env = GridWorld()
        for state in env.states():
            for action in range(4):
                probs = env.transition_probs(state, action)
                total_prob = sum(p for _, p, _ in probs)
                assert abs(total_prob - 1.0) < 1e-10

    def test_grid_values(self):
        from policywerk.building_blocks.value_functions import TabularV
        env = GridWorld()
        v = TabularV()
        v.set("0,0", 5.0)
        grid = env.grid_values(v)
        assert grid[0][0] == 5.0
        # Wall cell should be 0.0
        assert grid[1][1] == 0.0

    def test_num_actions(self):
        assert GridWorld().num_actions() == 4

    def test_is_terminal(self):
        env = GridWorld()
        from policywerk.building_blocks.mdp import State
        assert env.is_terminal(State(features=[0.0, 4.0], label="0,4"))  # goal
        assert env.is_terminal(State(features=[1.0, 3.0], label="1,3"))  # pit
        assert not env.is_terminal(State(features=[0.0, 0.0], label="0,0"))

    def test_terminal_states_are_absorbing(self):
        env = GridWorld()
        from policywerk.building_blocks.mdp import State
        goal = State(features=[0.0, 4.0], label="0,4")
        pit = State(features=[1.0, 3.0], label="1,3")
        # Terminal states should self-loop with zero reward
        for terminal in [goal, pit]:
            for action in range(4):
                probs = env.transition_probs(terminal, action)
                assert len(probs) == 1
                next_s, prob, reward = probs[0]
                assert next_s.label == terminal.label
                assert prob == 1.0
                assert reward == 0.0

    def test_terminal_states_in_states(self):
        env = GridWorld()
        state_labels = {s.label for s in env.states()}
        assert "0,4" in state_labels  # goal
        assert "1,3" in state_labels  # pit

    def test_step_from_terminal_is_absorbing(self):
        """step() after reaching a terminal state should self-loop."""
        env = GridWorld()
        env.reset()
        # Navigate to goal at (0,4)
        for _ in range(4):
            env.step(0)  # N
        for _ in range(3):
            env.step(1)  # E
        s, r, done = env.step(1)  # into goal
        assert done and r == 1.0
        # Now step again from the terminal state — should stay put
        s2, r2, done2 = env.step(1)
        assert done2
        assert r2 == 0.0  # absorbing: no further reward
        assert s2.label == s.label  # didn't move


class TestCliffWorld:
    def test_reset(self):
        env = CliffWorld()
        state = env.reset()
        assert state.label == "3,0"

    def test_cliff_penalty(self):
        env = CliffWorld()
        env.reset()
        # Step east into cliff at (3,1)
        s, r, done = env.step(1)
        assert r == -100.0
        assert not done  # teleported to start
        assert s.label == "3,0"

    def test_goal(self):
        env = CliffWorld()
        env.reset()
        # Go north, then east along row 2, then south to goal
        env.step(0)  # to (2,0)
        for _ in range(11):
            env.step(1)  # east to (2,11)
        s, r, done = env.step(2)  # south to (3,11) = goal
        assert done and r == 0.0

    def test_normal_step_cost(self):
        env = CliffWorld()
        env.reset()
        s, r, done = env.step(0)  # north to (2,0)
        assert r == -1.0 and not done

    def test_num_actions(self):
        assert CliffWorld().num_actions() == 4


class TestBalance:
    def test_reset(self):
        env = Balance()
        state = env.reset()
        assert len(state.features) == 2
        assert abs(state.features[0] - 0.01) < 1e-10  # initial angle

    def test_survives_with_control(self):
        env = Balance()
        env.reset()
        # Alternate actions — crude balancing
        done = False
        steps = 0
        for _ in range(50):
            s, r, done = env.step(0)
            if done:
                break
            steps += 1
        # Should survive at least a few steps
        assert steps > 5

    def test_terminal_on_angle(self):
        env = Balance(max_angle=0.01)  # very tight threshold
        env.reset()
        # Keep pushing one direction until it falls
        done = False
        for _ in range(100):
            _, _, done = env.step(1)
            if done:
                break
        assert done

    def test_discretized_label(self):
        env = Balance()
        state = env.reset()
        # Label should be "bin_a,bin_v" format
        parts = state.label.split(",")
        assert len(parts) == 2

    def test_num_actions(self):
        assert Balance().num_actions() == 2

    def test_step_continuous(self):
        env = Balance()
        env.reset()
        s, r, done = env.step_continuous(0.5)
        assert len(s.features) == 2
        assert r == 1.0  # survived one step


class TestCatcher:
    def test_reset(self):
        env = Catcher(seed=42)
        state = env.reset()
        assert len(state.features) == 256  # 16x16 flattened
        assert state.label == "8,8"  # center

    def test_render_frame(self):
        env = Catcher(seed=42)
        env.reset()
        frame = env.render_frame()
        assert len(frame) == 16
        assert len(frame[0]) == 16
        # Agent should be at center
        assert frame[8][8] == 1.0

    def test_collect_reward(self):
        env = Catcher(seed=42, num_rewards=1, num_hazards=0, max_steps=500)
        env.reset()
        # With seed=42, agent starts at (8,8) and reward is at (3,0).
        # Navigate: 8 steps west (action 3), then 5 steps north (action 0).
        collected = False
        for _ in range(8):   # west to column 0
            s, r, done = env.step(3)
            if r == 1.0:
                collected = True
                break
        if not collected:
            for _ in range(5):  # north to row 3
                s, r, done = env.step(0)
                if r == 1.0:
                    collected = True
                    break
        assert collected, "Agent should have collected the reward at (3,0)"
        assert done, "Collecting the only reward should end the episode"

    def test_num_actions(self):
        assert Catcher().num_actions() == 4


class TestBreakout:
    def test_reset(self):
        env = Breakout()
        state = env.reset()
        assert len(state.features) == ROWS * COLS + 2  # 80 pixels + 2 velocity
        assert state.label == "p2,b3,4"

    def test_features_include_velocity(self):
        """State features should end with ball velocity (dr, dc)."""
        env = Breakout()
        state = env.reset()
        # Ball starts moving down-right: dr=1, dc=1
        assert state.features[-2] == 1.0  # ball_dr
        assert state.features[-1] == 1.0  # ball_dc

    def test_render_frame(self):
        env = Breakout()
        env.reset()
        frame = env.render_frame()
        assert len(frame) == ROWS
        assert len(frame[0]) == COLS
        # Ball at (3, 4)
        assert frame[3][4] == BALL
        # Paddle at row 9, cols 2-4
        assert frame[9][2] == PADDLE
        assert frame[9][3] == PADDLE
        assert frame[9][4] == PADDLE
        # Bricks at rows 0-1, cols 1-6
        assert frame[0][1] == BRICK
        assert frame[1][6] == BRICK

    def test_ball_bounces_off_wall(self):
        env = Breakout()
        env.reset()
        # Ball at (3,4) moving right. It should hit the right wall
        # and reverse dc. Step with stay action until dc flips.
        initial_dc = env._ball_dc
        for _ in range(20):
            env.step(1)
            if env._ball_dc != initial_dc:
                break
        assert env._ball_dc == -initial_dc

    def test_paddle_catches_ball(self):
        """Moving paddle to intercept should bounce the ball."""
        env = Breakout()
        env.reset()
        # Ball starts at (3,4) moving down-right.
        # Move paddle right to intercept.
        bounced = False
        for _ in range(50):
            _, reward, done = env.step(2)  # move right
            if env._ball_dr == -1 and env._ball_r < ROWS - 2:
                bounced = True
                break
            if done:
                break
        assert bounced, "Ball should have bounced off paddle"

    def test_brick_destroyed_gives_reward(self):
        """Hitting a brick should give +1.0 reward."""
        env = Breakout()
        env.reset()
        initial_bricks = len(env._bricks)
        got_brick_reward = False
        for _ in range(100):
            _, reward, done = env.step(2)  # keep moving right
            if reward == 1.0:
                got_brick_reward = True
                break
            if done:
                break
        assert got_brick_reward, "Should get +1.0 for hitting a brick"
        assert len(env._bricks) < initial_bricks

    def test_miss_ends_episode(self):
        """Ball passing the paddle should end with -1.0."""
        env = Breakout()
        env.reset()
        # Move paddle left while ball goes right — guaranteed miss
        for _ in range(20):
            _, reward, done = env.step(0)  # move left
            if done:
                assert reward == -1.0, f"Miss should give -1.0, got {reward}"
                break
        assert done, "Ball should have missed the paddle"

    def test_step_after_done_is_noop(self):
        """Stepping after the episode ends should return 0 reward and done=True."""
        env = Breakout(max_steps=200)
        env.reset()
        # Force a miss to end the episode
        for _ in range(20):
            _, reward, done = env.step(0)
            if done:
                break
        assert done
        # Step again — should be a no-op
        s, r, d = env.step(1)
        assert d is True
        assert r == 0.0

    def test_all_bricks_cleared(self):
        """Clearing all bricks should end the episode with +1 reward."""
        env = Breakout(max_steps=2000)
        env.reset()
        # Remove all but one brick manually, then let the ball hit it
        target = next(iter(env._bricks))
        env._bricks = {target}
        # Position ball adjacent to the brick, moving toward it
        env._ball_r = target[0] + 1
        env._ball_c = target[1]
        env._ball_dr = -1
        env._ball_dc = 0
        _, reward, done = env.step(1)  # stay
        assert reward == 1.0
        assert done is True

    def test_render_color_frame(self):
        """render_color_frame should return an RGB grid."""
        env = Breakout()
        env.reset()
        frame = env.render_color_frame()
        assert len(frame) == ROWS
        assert len(frame[0]) == COLS
        # Each pixel should be [r, g, b]
        assert len(frame[0][0]) == 3
        # Ball at (3,4) should be white [1.0, 1.0, 1.0]
        assert frame[3][4] == [1.0, 1.0, 1.0]
        # Paddle at row 9, col 2 should be blue
        assert frame[9][2] == [0.3, 0.5, 0.9]
        # Brick at row 0, col 1 should be red
        assert frame[0][1] == [0.9, 0.2, 0.2]
        # Brick at row 1, col 1 should be orange
        assert frame[1][1] == [0.9, 0.6, 0.2]

    def test_max_steps_ends_episode(self):
        """Reaching max_steps should end the episode."""
        env = Breakout(max_steps=5)
        env.reset()
        done = False
        for _ in range(10):
            _, _, done = env.step(1)
            if done:
                break
        assert done

    def test_left_wall_bounce(self):
        """Ball should bounce off the left wall."""
        env = Breakout(max_steps=200)
        env.reset()
        env._ball_c = 0
        env._ball_dc = -1
        env._ball_dr = 1
        env._ball_r = 5
        env.step(1)
        assert env._ball_dc == 1  # reversed

    def test_top_wall_bounce(self):
        """Ball should bounce off the top wall."""
        env = Breakout(max_steps=200)
        env.reset()
        # Clear bricks so ball can reach the top
        env._bricks.clear()
        env._ball_r = 0
        env._ball_dr = -1
        env._ball_dc = 1
        env.step(1)
        assert env._ball_dr == 1  # reversed

    def test_num_actions(self):
        assert Breakout().num_actions() == 3


class TestCatcherCapacity:
    def test_grid_too_small_raises(self):
        """Grid that cannot fit all objects should raise ValueError."""
        import pytest
        with pytest.raises(ValueError, match="cannot fit"):
            Catcher(grid_size=2, num_rewards=3, num_hazards=2)

    def test_exact_capacity_ok(self):
        """Grid with exactly enough cells should not raise."""
        # 2x2 = 4 cells, 1 agent + 2 rewards + 1 hazard = 4
        env = Catcher(grid_size=2, num_rewards=2, num_hazards=1, max_steps=5)
        state = env.reset()
        assert len(state.features) == 4

    def test_negative_grid_size_raises(self):
        """Negative grid_size should raise before capacity math."""
        import pytest
        with pytest.raises(ValueError, match="positive integer"):
            Catcher(grid_size=-2, num_rewards=0, num_hazards=0)

    def test_zero_grid_size_raises(self):
        import pytest
        with pytest.raises(ValueError, match="positive integer"):
            Catcher(grid_size=0, num_rewards=0, num_hazards=0)

    def test_float_grid_size_raises(self):
        """Non-integer grid_size should raise even if positive."""
        import pytest
        with pytest.raises(ValueError, match="positive integer"):
            Catcher(grid_size=2.5, num_rewards=0, num_hazards=0)

    def test_bool_grid_size_raises(self):
        """bool is a subclass of int but should be rejected."""
        import pytest
        with pytest.raises(ValueError, match="positive integer"):
            Catcher(grid_size=True, num_rewards=0, num_hazards=0)


class TestCatcherExtended:
    def test_hazard_ends_episode(self):
        """Stepping onto a hazard should give -1.0 and end."""
        env = Catcher(seed=42, num_rewards=0, num_hazards=1, max_steps=500)
        state = env.reset()
        # Find the hazard position
        hazard_pos = list(env._hazard_positions)[0]
        # Teleport the agent next to the hazard
        hr, hc = hazard_pos
        env._agent_pos = (hr, hc - 1) if hc > 0 else (hr, hc + 1)
        # Step toward the hazard
        action = 1 if hc > env._agent_pos[1] else 3  # east or west
        _, reward, done = env.step(action)
        assert reward == -1.0
        assert done

    def test_max_steps_ends_episode(self):
        """Reaching max_steps without collecting should end with 0 reward."""
        env = Catcher(seed=42, num_rewards=1, num_hazards=0, max_steps=3)
        env.reset()
        # Just stay in place
        done = False
        final_reward = 0.0
        for _ in range(10):
            _, final_reward, done = env.step(0)
            if done:
                break
        assert done

    def test_partial_collection(self):
        """Collecting one of multiple rewards should not end the episode."""
        env = Catcher(seed=42, num_rewards=3, num_hazards=0, max_steps=500)
        env.reset()
        # Find first reward and teleport next to it
        first_reward = list(env._reward_positions)[0]
        rr, rc = first_reward
        env._agent_pos = (rr, rc - 1) if rc > 0 else (rr, rc + 1)
        action = 1 if rc > env._agent_pos[1] else 3
        _, reward, done = env.step(action)
        assert reward == 1.0
        assert not done  # still more rewards to collect


class TestPointMass:
    def test_reset(self):
        env = PointMass()
        state = env.reset()
        assert len(state.features) == 6
        assert state.features[0] == 0.0  # x
        assert state.features[1] == 0.0  # y

    def test_step_moves_agent(self):
        env = PointMass()
        env.reset()
        s1, _, _ = env.step(2)  # East
        assert s1.features[0] > 0.0  # x should increase

    def test_step_continuous(self):
        env = PointMass()
        env.reset()
        s, r, done = env.step_continuous([1.0, 0.0])
        assert s.features[0] > 0.0  # moved east

    def test_reaches_target(self):
        env = PointMass(target=(0.1, 0.0), reach_threshold=0.15, max_steps=100)
        env.reset()
        done = False
        for _ in range(50):
            _, _, done = env.step(2)  # push east toward (0.1, 0)
            if done:
                break
        assert done

    def test_num_actions(self):
        assert PointMass().num_actions() == 9


class TestPixelPointMass:
    def test_reset(self):
        env = PixelPointMass()
        state = env.reset()
        assert len(state.features) == 256  # 16x16

    def test_render_frame(self):
        env = PixelPointMass()
        env.reset()
        frame = env.render_frame()
        assert len(frame) == 16
        assert len(frame[0]) == 16
        # Should have agent (1.0) and target (0.7) somewhere
        flat = [v for row in frame for v in row]
        assert 1.0 in flat
        assert 0.7 in flat

    def test_step_continuous(self):
        env = PixelPointMass()
        env.reset()
        s, r, done = env.step_continuous([1.0, 0.0])
        assert len(s.features) == 256

    def test_discrete_step(self):
        """Discrete step should move agent and return pixel state."""
        env = PixelPointMass()
        env.reset()
        s, r, done = env.step(2)  # East
        assert len(s.features) == 256
        assert isinstance(r, float)

    def test_num_actions(self):
        assert PixelPointMass().num_actions() == 9
