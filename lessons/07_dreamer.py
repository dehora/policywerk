"""Lesson 07: DreamerV3 World Model (simplified).

Hafner et al. (2023), 'Mastering Diverse Domains through World Models.'

Learns a model of the environment from pixel observations, then
imagines trajectories in latent space and trains the policy on
imagined data. Real data trains the world model; imagined data
trains the policy.

    uv run python lessons/07_dreamer.py

Artifacts:
  output/07_dreamer_artifact.gif   Training animation
  output/07_dreamer_poster.png     Trained agent snapshot
  output/07_dreamer_trace.png      Loss curves
"""

import os
from dataclasses import dataclass

from policywerk.actors.dreamer import dreamer
from policywerk.building_blocks.network import network_forward
from policywerk.building_blocks.recurrent import gru_forward
from policywerk.building_blocks.distributions import Gaussian
from policywerk.world.pixel_pointmass import PixelPointMass, SIZE
from policywerk.viz.animate import (
    FrameSnapshot, create_lesson_figure, save_animation, save_poster,
    TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.traces import update_trace_axes
from policywerk.viz.trajectories import draw_real_vs_imagined, draw_pixel_env
from policywerk.primitives.progress import Spinner
from policywerk.primitives import scalar, matrix

import matplotlib.pyplot as plt

Matrix = list[list[float]]


@dataclass
class DreamerSnapshot(FrameSnapshot):
    real_frame: Matrix
    imagined_frame: Matrix
    step_label: str
    phase: str = ""


def main():
    print("=" * 64)
    print("  Lesson 07: DreamerV3 World Model (2023)")
    print("  Hafner et al., 'Mastering Diverse Domains'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. The imagination insight
    # -----------------------------------------------------------------------

    print("THE IMAGINATION INSIGHT")
    print("-" * 64)
    print("""
    Every algorithm so far has learned from real experience. The
    agent takes an action, the environment responds, and the agent
    learns from what happened. DQN stored these transitions in a
    replay buffer. PPO collected fresh batches each iteration.
    But every training step required a real environment step.

    DreamerV3 asks: what if the agent could practice in its head?

    Instead of learning a value function (L01-L05) or a policy
    (L06), Dreamer learns a model of the world itself. Given the
    current state and an action, the model predicts what happens
    next—what the agent will see and what reward it will get.
    Once the model is good enough, the agent can imagine thousands
    of trajectories without touching the real environment, and
    train its policy on imagined experience.

    This is the final paradigm in this series:

      L01-L05:  learn values, derive actions
      L06:      learn the policy directly
      L07:      learn the world, train in imagination
    """)

    # -----------------------------------------------------------------------
    # 2. The PixelPointMass
    # -----------------------------------------------------------------------

    print("THE PIXEL WORLD")
    print("-" * 64)
    print(f"""
    The environment is a {SIZE}x{SIZE} pixel grid. A white dot
    is the agent. A gray dot is the target. The background is
    black. The agent applies a 2D force to move toward the
    target—think of pushing a marble across a table.

      Observation: {SIZE}x{SIZE} = {SIZE * SIZE} pixel values (flattened)
      Action:      [force_x, force_y] in [-1, +1] (continuous 2D)
      Reward:      negative distance to target (closer = less negative)
      Goal:        reach the target

    The agent does not see coordinates—it sees {SIZE * SIZE} numbers
    and must learn to navigate from pixels alone. The physics is
    a "point-mass" (an object with position and velocity but no
    shape), the same as the PointMass environment from L06's
    building blocks but now observed through a camera.
    """)

    # -----------------------------------------------------------------------
    # 3. The world model
    # -----------------------------------------------------------------------

    print("THE WORLD MODEL")
    print("-" * 64)
    print("""
    The world model has four parts:

      Encoder:     pixels (256) -> latent state (32)
      GRU dynamics: (latent state, action) -> next latent state
      Decoder:     latent state (32) -> predicted pixels (256)
      Reward head: latent state (32) -> predicted reward (1)

    The encoder compresses 256 pixel values into 32 numbers—a
    compact representation of what matters in the scene. The GRU
    (a recurrent network from L03's building blocks) predicts
    how the latent state changes when the agent takes an action.
    This is the learned "physics engine."

    The decoder and reward head are training signals, not used
    during imagination. The decoder forces the encoder to
    preserve enough information to reconstruct the scene. The
    reward head forces the dynamics to track reward-relevant
    features—where the agent is relative to the target.

    All four components are dense networks. The real DreamerV3
    uses convolutional encoders for high-resolution images.
    For a 16x16 grid with two single-pixel markers, dense
    layers are sufficient.
    """)

    # -----------------------------------------------------------------------
    # 4. Teacher forcing
    # -----------------------------------------------------------------------

    print("TEACHER FORCING")
    print("-" * 64)
    print("""
    During training, the world model sees real observations at
    every step. The encoder converts each pixel frame into a
    latent state, the GRU predicts the next state from (latent,
    action), and the decoder tries to reconstruct the next frame.

    At each step, we reset the latent state to the encoder's
    output from the actual observation rather than using the
    GRU's prediction. This is called teacher forcing: the model
    always sees the ground truth, never its own mistakes.

    Teacher forcing makes training stable—each step's gradients
    are independent, no errors accumulate. But it creates a gap:
    during training, the model never practices running without
    observations. During imagination, it must run open-loop—no
    observations to correct it.

    The result: early imagination steps closely match reality
    (the model just saw a real observation), but later steps
    diverge as prediction errors compound. The animation shows
    this—the imagined frame tracks the real one at first, then
    drifts. This is the fundamental tradeoff of world models:
    imagination is free but imperfect.
    """)

    # -----------------------------------------------------------------------
    # 5. Imagining trajectories
    # -----------------------------------------------------------------------

    print("IMAGINING TRAJECTORIES")
    print("-" * 64)
    print("""
    Once the world model is trained, the agent can imagine:

      1. Encode a real observation into a latent state
      2. Choose an action using the actor network
      3. Predict the next latent state using the GRU
      4. Predict the reward using the reward head
      5. Repeat from step 2 (no real environment needed)

    The GRU rolls forward in latent space, the actor chooses
    actions, and the reward head estimates rewards. The entire
    trajectory happens in the agent's "head"—256 pixels are
    never rendered, physics is never simulated. Only 32 latent
    numbers flow through the GRU at each step.

    This is why world models are valuable: imagining a step
    costs one GRU forward pass (32 numbers), while a real step
    costs a full environment simulation plus 256-pixel rendering.
    """)

    # -----------------------------------------------------------------------
    # 6. Training on imagination
    # -----------------------------------------------------------------------

    print("TRAINING ON IMAGINATION")
    print("-" * 64)
    print("""
    The actor and critic operate entirely in latent space.
    They never see pixels—only the 32-number latent state.

    From each imagined trajectory, the critic estimates the
    value of each latent state, and lambda returns (the same
    mechanism from Lessons 03 and 06) compute the target return.
    The advantage is the difference: how much better was the
    imagined outcome than the critic predicted?

    The actor gradient is the same as L06's simplest policy
    gradient: increase the probability of actions with positive
    advantage, decrease the probability of actions with negative
    advantage. No PPO clip is needed because the imagined data
    is freshly generated (on-policy by construction).

    The full training loop alternates:
      1. Collect real data (actor chooses actions from pixels)
      2. Train world model on real data (teacher-forced)
      3. Imagine trajectories in latent space
      4. Train actor-critic on imagined data
    """)

    # -----------------------------------------------------------------------
    # 7. Training
    # -----------------------------------------------------------------------

    print("TRAINING")
    print("-" * 64)

    num_iterations = 60
    steps_per_iter = 100
    world_model_epochs = 3
    imagination_horizon = 10
    num_imaginations = 8
    latent_dim = 32
    hidden_dim = 64

    print(f"""
    Encoder:     256 -> {hidden_dim} (tanh) -> {latent_dim} (tanh)
    GRU:         input={2} (action), hidden={latent_dim}
    Decoder:     {latent_dim} -> {hidden_dim} (tanh) -> 256 (sigmoid)
    Reward head: {latent_dim} -> 32 (tanh) -> 1 (identity)
    Actor:       {latent_dim} -> 32 (tanh) -> 4 (identity)
    Critic:      {latent_dim} -> 32 (tanh) -> 1 (identity)

    Hyperparameters:
      Iterations:          {num_iterations}
      Steps/iter:          {steps_per_iter}
      World model epochs:  {world_model_epochs}
      Imagination horizon: {imagination_horizon}
      Num imaginations:    {num_imaginations}
    """)

    env = PixelPointMass(max_steps=100)

    networks, history = dreamer(
        env,
        num_iterations=num_iterations,
        steps_per_iter=steps_per_iter,
        world_model_epochs=world_model_epochs,
        imagination_horizon=imagination_horizon,
        num_imaginations=num_imaginations,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        seed=42,
    )

    # Print training summary
    window = 15
    print()
    print(f"    Average per {window} iterations:")
    for start in range(0, num_iterations, window):
        end = min(start + window, num_iterations)
        avg_r = sum(h["avg_reward"] for h in history[start:end]) / (end - start)
        avg_rc = sum(h["recon_loss"] for h in history[start:end]) / (end - start)
        avg_rl = sum(h["reward_loss"] for h in history[start:end]) / (end - start)
        print(f"      Iterations {start:3d}-{end-1:3d}:  "
              f"reward {avg_r:+7.1f}  recon {avg_rc:.4f}  rew_loss {avg_rl:.4f}")
    print()

    # -----------------------------------------------------------------------
    # 8. What Dreamer learned
    # -----------------------------------------------------------------------

    print("WHAT DREAMER LEARNED")
    print("-" * 64)

    # Greedy evaluation with reconstruction comparison
    eval_env = PixelPointMass(max_steps=100)
    state = eval_env.reset()
    eval_steps = 0
    eval_reward = 0.0

    encoder = networks["encoder"]
    gru = networks["gru"]
    decoder = networks["decoder"]
    actor_net = networks["actor"]
    reward_head = networks["reward_head"]

    eval_real_frames = []
    eval_imagined_frames = []
    eval_positions = []

    for _ in range(100):
        pixels = state.features
        real_frame = eval_env.render_frame()
        eval_real_frames.append(real_frame)

        # Encode and reconstruct
        z, _ = network_forward(encoder, pixels)
        recon, _ = network_forward(decoder, z)
        recon_frame = matrix.reshape(recon, SIZE, SIZE)
        eval_imagined_frames.append(recon_frame)

        # Actor chooses action from latent state
        actor_out, _ = network_forward(actor_net, z)
        mean_x, mean_y = actor_out[0], actor_out[1]
        action = [scalar.clamp(mean_x, -1.0, 1.0),
                  scalar.clamp(mean_y, -1.0, 1.0)]

        state, reward, done = eval_env.step_continuous(action)
        eval_reward += reward
        eval_steps += 1
        eval_positions.append(eval_env._inner.position)

        if done:
            break

    print(f"    Greedy evaluation (deterministic policy):")
    print(f"      Steps:      {eval_steps}")
    print(f"      Reward:     {eval_reward:.1f}")

    # Compute reconstruction quality
    total_recon_err = 0.0
    for real, imag in zip(eval_real_frames, eval_imagined_frames):
        for r in range(SIZE):
            for c in range(SIZE):
                total_recon_err += (real[r][c] - imag[r][c]) ** 2
    avg_recon = total_recon_err / (len(eval_real_frames) * SIZE * SIZE)
    print(f"      Avg recon MSE: {avg_recon:.4f}")
    print()

    reached_target = eval_steps < 100  # terminated early = reached target
    if reached_target:
        print(f"    The agent reached the target in {eval_steps} steps.")
    else:
        print(f"    The agent did not reach the target in {eval_steps} steps.")
        final_pos = eval_positions[-1] if eval_positions else (0, 0)
        print(f"    Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        print(f"    Target: (0.80, 0.80)")
    print()

    print(f"""    The world model learned to reconstruct pixel frames with
    an average MSE of {avg_recon:.4f}. The encoder compresses
    256 pixels into 32 latent numbers, and the decoder
    reconstructs them well enough for the dynamics model to
    predict meaningful future states.

    This is a simplified version of DreamerV3. The full paper
    uses a stochastic RSSM (recurrent state-space model) with
    separate prior and posterior distributions, KL regularization,
    symlog predictions, twohot value distributions, and several
    other stabilization tricks. We kept the core idea—learn what
    happens next, then practice in imagination—and used dense
    networks with teacher forcing and MSE losses.

    The real DreamerV3 trains on Atari, robotics, and Minecraft
    with a single set of hyperparameters. Our version trains on a
    16x16 pixel grid in {num_iterations} iterations. The scale
    differs by orders of magnitude, but the architecture is
    recognizably the same: encode, predict, imagine, act.
    """)
    print()

    # -----------------------------------------------------------------------
    # 9. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    snapshots: list[DreamerSnapshot] = []

    blank_frame = [[0.0] * SIZE for _ in range(SIZE)]

    # --- Phase 1: Random policy (no world model yet) ---
    # Use strong random forces so the dot visibly moves at 16x16 resolution.
    # Sample every 3rd step to keep this phase short.
    from policywerk.primitives.random import create_rng as _create_rng
    rand_rng = _create_rng(99)
    rand_env = PixelPointMass(max_steps=50)
    state = rand_env.reset()
    rand_reward = 0.0
    for step in range(30):
        action = [rand_rng.choice([-1.0, 1.0]), rand_rng.choice([-1.0, 1.0])]
        state, reward, done = rand_env.step_continuous(action)
        rand_reward += reward
        if step % 3 == 0:
            snapshots.append(DreamerSnapshot(
                episode=0, total_reward=rand_reward,
                real_frame=rand_env.render_frame(), imagined_frame=blank_frame,
                step_label=f"1/3  Random policy (step {step})",
                phase="random",
            ))
        if done:
            break
    for _ in range(4):
        snapshots.append(DreamerSnapshot(
            episode=0, total_reward=rand_reward,
            real_frame=rand_env.render_frame(), imagined_frame=blank_frame,
            step_label=f"1/3  Random: reward {rand_reward:.1f}",
            phase="random",
        ))

    # --- Phase 2: Training progression ---
    # Show the starting frame as static context while the loss curve builds.
    start_frame = PixelPointMass(max_steps=10).reset()
    ref_frame = PixelPointMass(max_steps=10)
    ref_frame.reset()
    reference_frame = ref_frame.render_frame()

    sample_iters = sorted(set(
        list(range(0, num_iterations, max(1, num_iterations // 10))) +
        [num_iterations - 1]
    ))
    for idx in sample_iters:
        h = history[idx]
        snapshots.append(DreamerSnapshot(
            episode=idx,
            total_reward=h["avg_reward"],
            real_frame=reference_frame,
            imagined_frame=blank_frame,
            step_label=f"2/3  Training: iter {idx}  recon={h['recon_loss']:.4f}",
            phase="training",
        ))

    # --- Phase 3: Trained agent with real vs reconstructed ---
    for step_idx in range(min(len(eval_real_frames), 50)):
        snapshots.append(DreamerSnapshot(
            episode=num_iterations,
            total_reward=eval_reward,
            real_frame=eval_real_frames[step_idx],
            imagined_frame=eval_imagined_frames[step_idx],
            step_label=f"3/3  Trained agent (step {step_idx})",
            phase="eval",
        ))
    for _ in range(12):
        snapshots.append(DreamerSnapshot(
            episode=num_iterations,
            total_reward=eval_reward,
            real_frame=eval_real_frames[-1] if eval_real_frames else blank_frame,
            imagined_frame=eval_imagined_frames[-1] if eval_imagined_frames else blank_frame,
            step_label=f"3/3  Reward: {eval_reward:.1f}",
            phase="eval",
        ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 07: DreamerV3",
        subtitle="Hafner et al. (2023) | learn the world, train in imagination",
        figsize=(12, 7),
    )

    recon_list = [h["recon_loss"] for h in history]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: pixel frames
        axes["env"].clear()
        if snap.phase == "random":
            # Center the single frame in the same canvas size as the
            # split-screen (16+1+16=33 cols) so the animation doesn't jump.
            from policywerk.viz.trajectories import _frame_to_rgb, _add_pixel_grid
            rows_f = len(snap.real_frame)
            cols_f = len(snap.real_frame[0]) if snap.real_frame else 0
            canvas_cols = cols_f * 2 + 1  # match split-screen width
            bg = [0.10, 0.10, 0.18]
            canvas = [[list(bg) for _ in range(canvas_cols)] for _ in range(rows_f)]
            rgb_frame = _frame_to_rgb(snap.real_frame)
            offset = (canvas_cols - cols_f) // 2  # center horizontally
            for r in range(rows_f):
                for c in range(cols_f):
                    canvas[r][offset + c] = rgb_frame[r][c]
            axes["env"].imshow(canvas, interpolation="nearest", aspect="equal")
            # Grid only over the 16x16 frame area, not the padding
            for r in range(rows_f + 1):
                axes["env"].axhline(r - 0.5, xmin=offset / canvas_cols,
                                    xmax=(offset + cols_f) / canvas_cols,
                                    color="#333333", linewidth=0.3, zorder=2)
            for c in range(cols_f + 1):
                axes["env"].axvline(offset + c - 0.5,
                                    color="#333333", linewidth=0.3, zorder=2)
            axes["env"].set_xticks([])
            axes["env"].set_yticks([])
        else:
            # Training and eval: show real vs reconstructed
            draw_real_vs_imagined(axes["env"], snap.real_frame, snap.imagined_frame)
        axes["env"].set_title(snap.step_label, fontsize=10)

        # Top-right: phase info
        axes["algo"].clear()
        axes["algo"].axis("off")
        if snap.phase == "random":
            axes["algo"].text(0.5, 0.5, "Before training\n(random actions)\n\n"
                              f"Reward: {snap.total_reward:+.1f}",
                              transform=axes["algo"].transAxes,
                              ha="center", va="center", fontsize=10, color=DARK_GRAY)
        elif snap.phase == "training":
            axes["algo"].text(0.1, 0.7,
                              f"Iteration: {snap.episode}\n"
                              f"Reward:    {snap.total_reward:+.1f}\n"
                              f"Recon MSE: {history[snap.episode]['recon_loss']:.4f}",
                              transform=axes["algo"].transAxes,
                              fontsize=10, color=DARK_GRAY, fontfamily="monospace",
                              verticalalignment="top")
            axes["algo"].set_title("World Model Training", fontsize=10)
        else:
            axes["algo"].text(0.5, 0.5, "Real (left) vs\nReconstructed (right)\n\n"
                              f"Reward: {snap.total_reward:+.1f}",
                              transform=axes["algo"].transAxes,
                              ha="center", va="center", fontsize=10, color=DARK_GRAY)
            axes["algo"].set_title("Trained Agent", fontsize=10)

        # Bottom: loss trace
        axes["trace"].clear()
        if snap.phase == "random":
            axes["trace"].axis("off")
            axes["trace"].text(0.5, 0.5, "Training not started",
                               transform=axes["trace"].transAxes,
                               ha="center", va="center", fontsize=10,
                               color=DARK_GRAY, style="italic")
        else:
            n = min(snap.episode + 1, len(recon_list))
            if n > 0:
                update_trace_axes(axes["trace"], recon_list[:n],
                                  label="Reconstruction loss", color=TEAL)
            axes["trace"].set_ylabel("Loss", fontsize=9)
            axes["trace"].set_xlabel("Iteration", fontsize=9)
            axes["trace"].set_title("World Model Learning", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/07_dreamer_artifact.gif", fps=6)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 07: DreamerV3",
        subtitle="Hafner et al. (2023)",
        figsize=(12, 7),
    )

    # Use a mid-evaluation frame
    mid = len(eval_real_frames) // 3
    poster_real = eval_real_frames[mid] if eval_real_frames else blank_frame
    poster_imag = eval_imagined_frames[mid] if eval_imagined_frames else blank_frame

    def update_poster(frame_idx):
        draw_real_vs_imagined(axes2["env"], poster_real, poster_imag)
        axes2["env"].set_title("Real (left) vs Reconstructed (right)", fontsize=10)

        axes2["algo"].clear()
        axes2["algo"].axis("off")
        axes2["algo"].text(0.1, 0.9,
                           f"Eval reward: {eval_reward:+.1f}\n"
                           f"Eval steps:  {eval_steps}\n"
                           f"Avg recon MSE: {avg_recon:.4f}",
                           transform=axes2["algo"].transAxes,
                           fontsize=10, verticalalignment="top",
                           fontfamily="monospace", color=DARK_GRAY)
        axes2["algo"].set_title("Trained Agent", fontsize=10)

        axes2["trace"].plot(range(num_iterations), recon_list,
                            color=TEAL, linewidth=1.0, label="Reconstruction loss")
        reward_list = [h["avg_reward"] for h in history]
        ax2 = axes2["trace"].twinx()
        ax2.plot(range(num_iterations), reward_list,
                 color=ORANGE, linewidth=1.0, alpha=0.7, label="Reward")
        ax2.set_ylabel("Reward", fontsize=9, color=ORANGE)
        axes2["trace"].set_ylabel("Recon Loss", fontsize=9, color=TEAL)
        axes2["trace"].set_xlabel("Iteration", fontsize=9)
        axes2["trace"].set_title("Training Progress", fontsize=10)
        axes2["trace"].legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/07_dreamer_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Trace ---

    fig3, (ax_r, ax_l) = plt.subplots(2, 1, figsize=(10, 5), dpi=150, sharex=True)

    reward_list = [h["avg_reward"] for h in history]
    ax_r.plot(range(num_iterations), reward_list,
              color=TEAL, linewidth=1.0, label="Reward")
    ax_r.set_ylabel("Avg Reward", fontsize=9)
    ax_r.set_title("DreamerV3 Training on Pixel World", fontsize=10)
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    ax_l.plot(range(num_iterations), recon_list,
              color=ORANGE, linewidth=1.0, label="Reconstruction loss")
    rew_loss_list = [h["reward_loss"] for h in history]
    ax_l.plot(range(num_iterations), rew_loss_list,
              color=TEAL, linewidth=1.0, alpha=0.7, label="Reward prediction loss")
    ax_l.set_ylabel("Loss", fontsize=9)
    ax_l.set_xlabel("Iteration", fontsize=9)
    ax_l.legend(fontsize=8)
    ax_l.grid(True, alpha=0.3)

    fig3.tight_layout()

    with Spinner("Generating trace"):
        fig3.savefig("output/07_dreamer_trace.png")
    plt.close(fig3)

    print("    Generating animation... done.")
    print("    Generating poster... done.")
    print("    Generating trace... done.")
    print()
    print("    Artifacts saved to output/:")
    print("      07_dreamer_artifact.gif  World model training")
    print("      07_dreamer_poster.png    Real vs reconstructed")
    print("      07_dreamer_trace.png     Loss curves")
    print()

    print("=" * 64)
    print("  Lesson 07 complete.")
    print()
    print("  DreamerV3 closed the final gap in this series: learning")
    print("  a model of the world and training in imagination.")
    print()
    print("  Lesson 01 started with a known model and exact planning.")
    print("  Lesson 02 removed the model and learned from experience.")
    print("  Lessons 03-04 refined how to learn from experience.")
    print("  Lesson 05 scaled to pixels with neural networks.")
    print("  Lesson 06 learned the policy directly.")
    print("  Lesson 07 learned the world itself.")
    print()
    print("  The trajectory from Bellman (1957) to Hafner (2023) is")
    print("  66 years of asking the same question in different ways:")
    print("  given what I know, what should I do next?")
    print("=" * 64)


if __name__ == "__main__":
    main()
