"""Lesson 05: Deep Q-Network (DQN).

Mnih et al. (2013), 'Playing Atari with Deep Reinforcement Learning.'

Extends Q-learning from tables to neural networks. The agent sees
raw pixels and learns action values through experience replay,
a target network, and epsilon decay.

    uv run python lessons/05_dqn.py

Artifacts:
  output/05_dqn_artifact.gif   Training animation
  output/05_dqn_poster.png     Trained agent snapshot
  output/05_dqn_trace.png      Reward and loss curves
"""

import os
from dataclasses import dataclass

from policywerk.actors.dqn import dqn, greedy_poster_frame
from policywerk.building_blocks.network import network_forward
from policywerk.world.breakout import Breakout, ROWS, COLS
from policywerk.viz.animate import (
    FrameSnapshot, create_lesson_figure, save_animation, save_poster,
    TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.traces import update_trace_axes
from policywerk.primitives.progress import Spinner
from policywerk.viz.trajectories import draw_breakout_frame
from policywerk.viz.values import draw_q_bars

import matplotlib.pyplot as plt

Matrix = list[list[float]]


@dataclass
class DQNSnapshot(FrameSnapshot):
    frame: list          # RGB frame (rows × cols × 3) for color display
    epsilon: float
    avg_loss: float
    avg_q: float
    ep_reward: float
    step_count: int
    step_label: str
    score: int = 0


def main():
    print("=" * 64)
    print("  Lesson 05: Deep Q-Network (2013)")
    print("  Mnih et al., 'Playing Atari with Deep Reinforcement Learning'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. When tables fail
    # -----------------------------------------------------------------------

    print("WHEN TABLES FAIL")
    print("-" * 64)
    print("""
    Q-learning (Lesson 04) stored one value per state-action pair in
    a lookup table. On the cliff world, that table had 48 states and
    4 actions: 192 entries. It worked because every state had a short
    label like "2,3" and the agent visited each one many times.

    But what happens when the agent sees pixels instead of coordinates?
    Consider a game of Breakout. The state is an 8x10 grid of pixel
    values. Every step the ball moves, and the image changes. The
    table approach needs one entry per unique state. But each pixel
    pattern is effectively unique. The agent will almost never see
    the exact same frame twice. A table that has never seen this
    particular arrangement of ball, paddle, and bricks has nothing
    to say about it.

    Tabular Q-learning cannot scale to problems where the state is
    an image, a sensor reading, or anything with continuous
    dimensions. The cliff world had 48 states. Breakout's 8x10
    grid has 80 pixels plus 2 velocity values (82 inputs total).
    The state space is astronomically larger than any table could cover.

    The solution is function approximation: instead of storing every
    Q-value in a table, use a neural network to approximate the
    Q-function. The network takes pixel values as input and outputs
    Q-values for every action. It generalizes across states because
    it learns patterns, not individual entries. Two frames that look
    similar (ball in roughly the same place, paddle shifted one pixel)
    produce similar Q-values, even if the agent has never seen either
    exact frame before. This is what tables cannot do, and it is what
    makes DQN work.
    """)

    # -----------------------------------------------------------------------
    # 2. The neural network as a Q-table
    # -----------------------------------------------------------------------

    print("THE NEURAL NETWORK AS A Q-TABLE")
    print("-" * 64)
    print("""
    In Lesson 04, looking up a Q-value meant indexing into a dict:

      Table lookup:    Q[(state_label, action)] -> float

    The label "2,3" mapped to a row in the table. Each state-action
    pair had its own independent entry, learned from visits to that
    exact state.

    DQN replaces the dict with a forward pass through a neural network:

      Network forward: network(state) -> [Q(s, left), Q(s, stay), Q(s, right)]

    The network takes 82 values as input: 80 pixel values from
    the grid plus 2 ball velocity components (vertical and
    horizontal direction). It passes them through a hidden layer
    of 32 neurons with ReLU activation, and outputs 3 numbers:
    one Q-value per action. The velocity is necessary because a
    single frame is ambiguous: the same pixel layout can occur
    with the ball moving up or down, and the correct action
    differs. Without velocity, the observation is not Markov.
    The agent picks the action with the highest Q-value, same
    as before. What changes is how those values are produced.

    The table memorizes. The network interpolates. Two frames with
    the ball one pixel apart share most of their weights and produce
    similar Q-values. The hidden neurons learn to detect patterns
    in the pixel input: one neuron might activate when the ball is
    near the paddle, another when bricks are clustered on the left.
    These patterns are not programmed; they emerge from training.

    The update rule is the same TD error from Lesson 04, but
    applied to network weights instead of table entries. A concrete
    example: the ball is at row 5, moving down. The online network
    says Q(stay) = 0.8. The agent stays. The ball moves to row 6,
    and the target network says the best action there is worth 0.7.
    With gamma=0.95 and the step cost of -0.01:

      target = -0.01 + 0.95 * 0.7 = 0.655
      loss   = huber(0.8, 0.655)

    The network predicted 0.8 but the target was 0.655. The
    prediction was too optimistic. Backpropagation traces this
    error backward through each layer of the network, computing
    how much each weight contributed to the mistake. Every weight
    is nudged slightly in the direction that would have reduced the
    error. This is the key difference from Lesson 04: instead of
    updating one table entry, a single training step adjusts all
    the weights, changing the Q-values for every state at once.

    Architecture:  82 inputs -> 32 hidden (ReLU) -> 3 outputs
    Loss:          Huber (quadratic for small errors, linear for large)
    Optimizer:     Adam (adaptive learning rates per weight)
    """)

    # -----------------------------------------------------------------------
    # 3. Three ideas that make it work
    # -----------------------------------------------------------------------

    print("THREE IDEAS THAT MAKE IT WORK")
    print("-" * 64)
    print("""
    Replacing a table with a network is not enough on its own.
    Naive neural Q-learning is unstable: the network can diverge,
    oscillate, or forget what it learned. The DQN paper introduced
    three ideas that make training reliable.

    1. EXPERIENCE REPLAY

       The agent stores every transition (state, action, reward,
       next_state) in a circular buffer. When it is time to train,
       it draws a random mini-batch from the buffer rather than
       training on the most recent experience.

       Why this matters: consecutive game frames are nearly identical.
       If the network trains on them in order, it sees the same
       pattern over and over and overfits to recent experience. It
       might learn "always go right" because the last 50 frames all
       had the ball moving right. Random sampling breaks this
       correlation by mixing transitions from different episodes
       and different game states.

    2. TARGET NETWORK

       The TD target (the value the network is trying to match)
       uses a frozen copy of the network, updated only every N
       episodes. The online network learns from this stable target.

       Without this, every gradient step shifts the target. The
       network is trying to hit a bullseye that moves every time
       it fires. The target network holds still long enough for
       the online network to converge, then updates to a new
       position.

    3. EPSILON DECAY

       The agent starts with epsilon=1.0 (pure random actions) and
       linearly decreases to epsilon=0.1 over training. Early
       randomness fills the replay buffer with diverse experience
       from all parts of the game. As the network improves, the
       agent shifts toward exploiting what it has learned, taking
       the action with the highest Q-value most of the time but
       still exploring occasionally.
    """)

    # -----------------------------------------------------------------------
    # 4. Mini Breakout
    # -----------------------------------------------------------------------

    print("MINI BREAKOUT")
    print("-" * 64)
    print(f"""
    The original DQN paper trained on Atari games at 210x160 pixels
    for 50 million frames on a GPU. We cannot do that in pure Python.
    Instead we use a cut-down Breakout: an {COLS}x{ROWS} grid with
    12 bricks, a bouncing ball, and a paddle the agent controls.

      row 0: . B B B B B B .
      row 1: . B B B B B B .
      row 2: . . . . . . . .
      row 3: . . . . o . . .   ball starts here, moving down-right
      row 4: . . . . . . . .
      row 5: . . . . . . . .
      row 6: . . . . . . . .
      row 7: . . . . . . . .
      row 8: . . . . . . . .
      row 9: . . . = = = . .   paddle, width 3

    The agent sees the grid as a flat list of {ROWS * COLS} pixel
    values plus 2 ball velocity components ({ROWS * COLS + 2} inputs
    total). Each pixel is a float:

      0.0 = empty (black)
      0.5 = brick (red for row 0, orange for row 1)
      0.7 = ball  (white)
      1.0 = paddle (blue)

    The velocity values (+1 or -1 for vertical and horizontal
    direction) make the observation Markov: a single frame alone
    is ambiguous because the same pixel layout can occur with
    the ball moving up or down.

    The ball moves one cell per step, bouncing off walls, bricks,
    and the paddle. Hitting a brick destroys it (+1.0 reward) and
    reverses the ball's vertical direction. If the ball passes the
    paddle, the episode ends with -1.0. Each step costs -0.01 to
    discourage wandering.

    Actions: left(0), stay(1), right(2). The paddle moves one cell
    per step, clamped to the grid edges. The agent must learn to
    track the ball, position the paddle to intercept it, and angle
    bounces toward remaining bricks, all from 82 input values.
    """)

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------

    print("TRAINING")
    print("-" * 64)
    print("""
    The network starts knowing nothing. With epsilon=1.0, every
    action is random. The paddle flails while the ball sails past.
    As the replay buffer fills with these early failures, training
    begins. The network slowly learns which pixel patterns precede
    reward (+1.0 brick hit) and which precede punishment (-1.0 miss).

    Epsilon decays linearly. By episode 100, epsilon is 0.55.
    With 3 actions, the greedy action is chosen about 63% of the
    time (0.45 exploit + 0.55/3 random chance of picking it).
    By episode 200, epsilon reaches its floor of 0.1 and the
    greedy action is chosen about 93% of the time. The transition
    from exploration to exploitation
    is visible in the reward curve: early episodes cluster near -1
    (immediate miss), then rewards climb as the agent learns to
    rally and hit bricks.
    """)

    num_episodes = 300
    hidden_size = 32
    batch_size = 16
    learning_rate = 0.01
    gamma = 0.95
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_episodes = 200
    target_update_freq = 20
    train_every = 2
    replay_capacity = 3000
    min_replay_size = 100

    print(f"    Training {num_episodes} episodes")
    print(f"    Network: {ROWS * COLS + 2} -> {hidden_size} (relu) -> 3 (identity)")
    print(f"    Batch size: {batch_size}, train every {train_every} steps")
    print(f"    Replay buffer: {replay_capacity} capacity, min {min_replay_size}")
    print(f"    Target network updated every {target_update_freq} episodes")
    print(f"    Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_episodes} episodes")
    print(f"    Learning rate: {learning_rate}, gamma: {gamma}")
    print()
    print("    These values were tuned for this environment. A larger")
    print("    network and more episodes would learn better, but pure")
    print("    Python backpropagation is slow. The real DQN paper used")
    print("    millions of frames on a GPU.")
    print()

    env = Breakout(max_steps=200)

    net, history = dqn(
        env,
        num_episodes=num_episodes,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_episodes=epsilon_decay_episodes,
        learning_rate=learning_rate,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        min_replay_size=min_replay_size,
        target_update_freq=target_update_freq,
        train_every=train_every,
        hidden_size=hidden_size,
        seed=42,
    )

    # Print training summary
    window = 50
    print()
    print(f"    Average reward per {window} episodes:")
    for start in range(0, num_episodes, window):
        end = min(start + window, num_episodes)
        avg_r = sum(h["total_reward"] for h in history[start:end]) / (end - start)
        avg_eps = sum(h["epsilon"] for h in history[start:end]) / (end - start)
        avg_steps = sum(h["steps"] for h in history[start:end]) / (end - start)
        print(f"      Episodes {start:3d}-{end-1:3d}:  reward {avg_r:6.2f}  "
              f"epsilon {avg_eps:.2f}  steps {avg_steps:.0f}")
    print()

    # Early episodes
    print("    Early episodes:")
    for h in history[:10]:
        print(f"      Episode {h['episode']:3d}:  reward {h['total_reward']:7.2f}  "
              f"steps {h['steps']:3d}  epsilon {h['epsilon']:.2f}")
    print()

    # -----------------------------------------------------------------------
    # 6. What the network learned
    # -----------------------------------------------------------------------

    print("WHAT THE NETWORK LEARNED")
    print("-" * 64)
    print("""
    With exploration turned off (epsilon=0), the trained network
    plays greedily: at every step it picks the action with the
    highest Q-value. No randomness, no safety net. This is what
    the network actually learned, stripped of the exploration noise
    that helped it learn.
    """)

    # Greedy evaluation
    eval_env = Breakout(max_steps=200)
    state = eval_env.reset()
    eval_total = 0.0
    eval_steps = 0
    eval_bricks_hit = 0
    for _ in range(200):
        q_vals, _ = network_forward(net, state.features)
        action = q_vals.index(max(q_vals))
        state, reward, done = eval_env.step(action)
        eval_total += reward
        eval_steps += 1
        if reward == 1.0:
            eval_bricks_hit += 1
        if done:
            break

    action_names = ["Left", "Stay", "Right"]
    bricks_remaining = eval_env.bricks_remaining()
    print(f"    Greedy evaluation (no exploration):")
    print(f"      Reward:     {eval_total:.2f}")
    print(f"      Steps:      {eval_steps}")
    print(f"      Bricks hit: {eval_bricks_hit}/12")
    print(f"      Remaining:  {bricks_remaining}")
    print()

    # Q-values for the starting state
    start_state = Breakout(max_steps=200).reset()  # fresh reset
    start_q, _ = network_forward(net, start_state.features)
    print("    Q-values at start position:")
    for i, (name, q) in enumerate(zip(action_names, start_q)):
        marker = " <-- best" if q == max(start_q) else ""
        print(f"      {name:5s}: {q:.4f}{marker}")
    print()

    print(f"""    The network learned to play Breakout from 82 inputs. It
    destroyed {eval_bricks_hit} of 12 bricks in {eval_steps} steps, earning a
    reward of {eval_total:.2f}. Compare this to the random policy, which
    missed the ball within 6 steps and scored nothing.

    The Q-values at the start position tell us what the network
    expects. The ball starts at row 3, column 4, moving down-right.
    The network has learned that moving right is slightly better
    than staying or moving left, because it needs to track the ball's
    trajectory toward the right wall.

    How did 32 hidden neurons learn this? Through the same
    mechanism as Lesson 04's TD error, but applied to weights
    instead of table entries. Each brick hit produced a +1.0
    reward that propagated backward through the replay buffer.
    Each miss produced a -1.0 penalty. Over 300 episodes, the
    network's weights shifted until the Q-values for "move toward
    the ball" exceeded the Q-values for "move away from it."

    The network does not understand Breakout. It has no concept
    of ball trajectory, paddle interception, or brick layout. It
    has a function from 82 numbers to 3 numbers, shaped by
    thousands of gradient updates, that happens to produce
    useful behavior. That is function approximation: not
    understanding, but interpolation that works.
    """)
    print()

    # -----------------------------------------------------------------------
    # 7. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots: random policy → training curve → trained policy
    snapshots: list[DQNSnapshot] = []
    from policywerk.primitives.random import create_rng as _create_rng

    # --- Phase 1: Random policy replay (before training) ---
    # Shows the ball falling past a randomly-moving paddle.
    rand_rng = _create_rng(99)
    rand_env = Breakout(max_steps=200)
    state = rand_env.reset()
    rand_total = 0.0
    rand_step = 0
    rand_done = False
    while not rand_done and rand_step < 200:
        action = rand_rng.randint(0, 2)
        state, reward, rand_done = rand_env.step(action)
        rand_total += reward
        rand_step += 1
        snapshots.append(DQNSnapshot(
            episode=0,
            total_reward=rand_total,
            frame=rand_env.render_color_frame(),
            epsilon=1.0,
            avg_loss=0.0,
            avg_q=0.0,
            ep_reward=rand_total,
            step_count=rand_step,
            step_label=f"Random policy (step {rand_step})",
            score=rand_env.score(),
        ))
    # Hold on "Game Over" for a few frames
    game_over_label = "Game Over—random policy" if rand_total < 0 else "Cleared—random policy"
    for _ in range(8):
        snapshots.append(DQNSnapshot(
            episode=0,
            total_reward=rand_total,
            frame=rand_env.render_color_frame(),
            epsilon=1.0,
            avg_loss=0.0,
            avg_q=0.0,
            ep_reward=rand_total,
            step_count=rand_step,
            step_label=game_over_label,
            score=rand_env.score(),
        ))

    # --- Phase 2: Training curve summary ---
    # A few frames showing the reward curve growing, with the initial
    # board state as a static backdrop.
    reset_frame_env = Breakout(max_steps=200)
    reset_frame_env.reset()
    static_frame = reset_frame_env.render_color_frame()

    sample_episodes = sorted(set(
        list(range(0, num_episodes, max(1, num_episodes // 15))) +
        [num_episodes - 1]
    ))
    for ep_idx in sample_episodes:
        h = history[ep_idx]
        snapshots.append(DQNSnapshot(
            episode=ep_idx,
            total_reward=h["total_reward"],
            frame=static_frame,
            epsilon=h["epsilon"],
            avg_loss=h["avg_loss"],
            avg_q=h["avg_q"],
            ep_reward=h["total_reward"],
            step_count=h["steps"],
            step_label=f"Training: episode {ep_idx}/{num_episodes}",
            score=0,
        ))

    # --- Capture a mid-game frame from the greedy rollout for the poster ---
    poster_frame = greedy_poster_frame(net, Breakout(max_steps=200))

    # --- Phase 3: Trained policy replay (looped twice) ---
    for _loop in range(2):
        g_env = Breakout(max_steps=200)
        state = g_env.reset()
        g_total = 0.0
        g_step = 0
        g_done = False
        while not g_done and g_step < 200:
            q_vals, _ = network_forward(net, state.features)
            action = q_vals.index(max(q_vals))
            state, reward, g_done = g_env.step(action)
            g_total += reward
            g_step += 1
            snapshots.append(DQNSnapshot(
                episode=num_episodes,
                total_reward=g_total,
                frame=g_env.render_color_frame(),
                epsilon=0.0,
                avg_loss=0.0,
                avg_q=0.0,
                ep_reward=g_total,
                step_count=g_step,
                step_label=f"Trained policy (step {g_step})",
                score=g_env.score(),
            ))
        # Hold on final frame with Game Over / Cleared
        if g_env.bricks_remaining() == 0:
            end_label = f"Cleared! score={g_env.score()}"
        else:
            end_label = f"Game Over—score {g_env.score()}/12"
        for _ in range(12):
            snapshots.append(DQNSnapshot(
                episode=num_episodes,
                total_reward=g_total,
                frame=g_env.render_color_frame(),
                epsilon=0.0,
                avg_loss=0.0,
                avg_q=0.0,
                ep_reward=g_total,
                step_count=g_step,
                step_label=end_label,
                score=g_env.score(),
            ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 05: DQN",
        subtitle="Mnih et al. (2013) | neural network replaces Q-table",
        figsize=(12, 7),
    )

    rewards_list = [h["total_reward"] for h in history]

    def update(frame_idx):
        snap = snapshots[frame_idx]
        is_trained = snap.episode >= num_episodes
        is_random = snap.epsilon >= 1.0 and not is_trained

        # Top-left: Breakout frame with score
        draw_breakout_frame(axes["env"], snap.frame, score=snap.score)
        axes["env"].set_title(snap.step_label, fontsize=10)

        # Top-right: info
        axes["algo"].clear()
        axes["algo"].axis("off")
        if is_random:
            info_lines = [
                "Before training",
                "(random actions)",
                "",
                f"Reward:  {snap.ep_reward:.2f}",
                f"Steps:   {snap.step_count}",
            ]
            phase_title = "Random Policy"
        elif is_trained:
            info_lines = [
                "After training",
                "(greedy policy)",
                "",
                f"Reward:  {snap.ep_reward:.2f}",
                f"Steps:   {snap.step_count}",
            ]
            phase_title = "Trained Policy"
        else:
            info_lines = [
                f"Episode: {snap.episode}",
                f"Reward:  {snap.ep_reward:.2f}",
                "",
                f"Epsilon: {snap.epsilon:.2f}",
            ]
            phase_title = "Training"
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title(phase_title, fontsize=10)

        # Bottom: reward trace (only during/after training)
        axes["trace"].clear()
        if is_random:
            axes["trace"].axis("off")
            axes["trace"].text(0.5, 0.5, "Training not started",
                               transform=axes["trace"].transAxes,
                               ha="center", va="center", fontsize=10,
                               color=DARK_GRAY, style="italic")
        else:
            n = min(snap.episode + 1, len(rewards_list))
            if n > 0:
                update_trace_axes(axes["trace"], rewards_list[:n],
                                  label="Episode reward", color=TEAL)
            axes["trace"].set_ylabel("Reward", fontsize=9)
            axes["trace"].set_xlabel("Episode", fontsize=9)
            axes["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/05_dqn_artifact.gif", fps=4)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 05: DQN",
        subtitle="Mnih et al. (2013)",
        figsize=(12, 7),
    )

    def update_poster(frame_idx):
        draw_breakout_frame(axes2["env"], poster_frame)
        axes2["env"].set_title("Trained Agent (greedy)", fontsize=10)

        draw_q_bars(axes2["algo"], list(start_q), action_names)
        axes2["algo"].set_title("Q-values at start", fontsize=10)

        axes2["trace"].plot(range(num_episodes), rewards_list,
                            color=TEAL, linewidth=0.5, alpha=0.3)
        w = 20
        if num_episodes > w:
            smooth = [sum(rewards_list[max(0, i - w + 1):i + 1]) /
                      len(rewards_list[max(0, i - w + 1):i + 1])
                      for i in range(num_episodes)]
            axes2["trace"].plot(range(num_episodes), smooth,
                                color=TEAL, linewidth=1.5, label="Reward (smoothed)")
        axes2["trace"].set_ylabel("Reward", fontsize=9)
        axes2["trace"].set_xlabel("Episode", fontsize=9)
        axes2["trace"].set_title("Training Progress", fontsize=10)
        axes2["trace"].legend(fontsize=8)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/05_dqn_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Trace ---

    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    ax3.plot(range(num_episodes), rewards_list,
             color=TEAL, linewidth=0.5, alpha=0.3, label="Raw reward")

    w = 20
    if num_episodes > w:
        smooth_r = [sum(rewards_list[max(0, i - w + 1):i + 1]) /
                    len(rewards_list[max(0, i - w + 1):i + 1])
                    for i in range(num_episodes)]
        ax3.plot(range(num_episodes), smooth_r,
                 color=TEAL, linewidth=1.5, label="Reward (smoothed)")

    ax3.set_ylabel("Reward", fontsize=9)
    ax3.set_xlabel("Episode", fontsize=9)
    ax3.set_title("DQN Training on Mini Breakout", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    with Spinner("Generating trace"):
        fig3.savefig("output/05_dqn_trace.png")
    plt.close(fig3)

    print("    Generating animation... done.")
    print("    Generating poster... done.")
    print("    Generating trace... done.")
    print()
    print("    Artifacts saved to output/:")
    print("      05_dqn_artifact.gif  DQN training on mini Breakout")
    print("      05_dqn_poster.png    Trained agent snapshot")
    print("      05_dqn_trace.png     Reward curve")
    print()

    print("=" * 64)
    print("  Lesson 05 complete.")
    print()
    print("  DQN closed the gap between tabular RL and the real world.")
    print("  Lesson 04's Q-learning needed a small, discrete state space.")
    print("  DQN replaced the table with a neural network. Three")
    print("  stabilization ideas (experience replay, target networks,")
    print("  epsilon decay) made it trainable. The agent learned to play")
    print("  Breakout from raw pixels and velocity: 82 numbers in, 3 Q-values out,")
    print("  thousands of gradient steps in between.")
    print()
    print("  But DQN still has a limitation. It learns values and derives")
    print("  a policy indirectly: pick the action with the highest Q-value.")
    print("  That works when actions are discrete (left, stay, right).")
    print("  But what about continuous actions, like applying 0.73 units")
    print("  of force or turning 12 degrees? There is no finite set of")
    print("  actions to take the max over.")
    print()
    print("  In Lesson 06, an algorithm called Proximal Policy")
    print("  Optimization (PPO) takes a different approach. Instead of")
    print("  learning Q-values and deriving a policy, PPO learns the")
    print("  policy directly: the network outputs a probability")
    print("  distribution over actions, and the agent samples from it.")
    print("=" * 64)


if __name__ == "__main__":
    main()
