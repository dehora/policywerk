"""Lesson 07: Dreamer-style World Model.

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
    real_path: list          # list of (x, y) tuples—real trajectory so far
    imagined_path: list      # list of (x, y) tuples—imagined trajectory so far
    real_frame: Matrix       # current pixel frame (real)
    imagined_frame: Matrix   # current pixel frame (reconstructed)
    step_label: str
    phase: str = ""


def main():
    print("=" * 64)
    print("  Lesson 07: Dreamer-style World Model (2023)")
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

    Dreamer asks: what if the agent could practice in its head?

    Instead of learning a value function (L01-L05) or a policy
    (L06), Dreamer learns a model of the world itself. Given the
    current state and an action, the model predicts what happens
    next—what the agent will see and what reward it will get.
    Once the model is good enough, the agent can imagine thousands
    of trajectories without touching the real environment, and
    train its policy on imagined experience.

    This matters because real environment steps are expensive.
    Physics must be simulated, pixels must be rendered, and the
    agent must wait for the result. In robotics, each step means
    a physical robot moving and risking damage. A world model
    turns one real episode into many training episodes. This is
    sample efficiency—getting more learning out of fewer real
    interactions.

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

    The encoder compresses 256 pixel values into 32 numbers
    called the latent state. "Latent" means hidden—the raw
    pixels are what the agent observes, but the latent state is
    a hidden summary that the encoder learns to extract. Of the
    256 pixels, most are black background. The encoder learns to
    keep what matters—where the agent and target are—and
    discard the rest.

    The GRU (Gated Recurrent Unit) predicts how the latent state
    changes when the agent takes an action. This is the learned
    "physics engine"—the transition function that turns (current
    state, action) into a predicted next state.

    Unlike the feed-forward networks from L05, the GRU is a
    recurrent network: it carries a hidden state from one step
    to the next, so its output depends on the history of inputs.
    A feed-forward network processes one input and forgets it.
    The GRU remembers. Its internal gates control what to keep
    from the previous state and what to update. A gate is a
    learned value between 0 and 1 that multiplies another
    signal—near 0 it blocks information, near 1 it passes
    through. The GRU has two gates: one decides how much of
    the old state to keep, the other decides how much new
    information to mix in.

    The decoder and reward head are training signals, not used
    during imagination. The decoder forces the encoder to
    preserve enough information to reconstruct the full scene
    from the 32-number summary. The reward head forces the
    dynamics to track features that matter for reward—
    specifically, where the agent is relative to the target.

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
    GRU's prediction. This is called teacher forcing—named by
    analogy to a teacher who corrects a student's workbook at
    every step rather than letting errors accumulate. The model
    always sees the ground truth, never its own mistakes.

    Teacher forcing makes training stable—each step's gradients
    are independent, no errors accumulate. But it creates a gap:
    during training, the model never practices running without
    observations. During imagination, it must run open-loop—
    advancing step by step with no real observations to correct
    accumulated errors, like navigating with your eyes closed.

    The result: early imagination steps closely match reality
    (the model just saw a real observation), but later steps
    diverge as prediction errors compound. Each GRU prediction
    becomes the input to the next prediction, and small errors
    in one step shift the input distribution for the next,
    causing drift to accelerate. The animation shows this—the
    imagined path tracks the real one at first, then veers away.
    This is the fundamental tradeoff of world models:
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

    Concrete example. The encoder sees a 16x16 frame where the
    agent is at pixel (4, 5) and the target is at (12, 12). It
    produces a 32-number latent state. The actor reads this
    latent state and outputs mean_x=0.6, mean_y=0.4—meaning
    "push mostly right and somewhat up." The GRU takes the
    latent state and the action [0.6, 0.4] and produces a new
    32-number latent state: its prediction of what the world
    looks like after that push. The reward head reads this new
    state and predicts reward=-1.2, meaning "still far from the
    target." No pixels were rendered. No physics was simulated.
    The entire step happened in 32 numbers.
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
    value of each latent state. Lambda returns (the same
    mechanism from Lessons 03 and 06) compute the target
    return by blending short-term predicted rewards with the
    critic's estimate of future value. The advantage is the
    difference between the lambda return and the critic's
    prediction: a positive advantage means the imagined
    trajectory turned out better than the critic expected, a
    negative advantage means worse. The only difference from
    L06 is that the rewards and values come from the world
    model's predictions rather than from the real environment.

    The actor gradient is the same as L06's policy gradient:
    increase the probability of actions with positive
    advantage, decrease the probability of actions with negative
    advantage. No PPO clip is needed because the imagined data
    is freshly generated—on-policy, meaning the data comes
    from the same policy being trained, not from old stored
    experience like DQN's replay buffer.

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

    The imagination horizon is how many steps the GRU rolls
    forward without real observations. Ten steps means each
    imagined trajectory is ten actions long. Longer horizons
    give the actor more future context but accumulate more
    prediction error—the same compounding drift described in
    the teacher forcing section.

    Reconstruction loss measures how well the decoder recreates
    the original pixels from the latent state. Lower means the
    encoder is preserving more useful information in its
    32-number summary. Reward prediction loss measures how well
    the dynamics model predicts the reward the agent will
    receive—lower means the GRU is tracking the agent's
    position relative to the target.

    Expect noisy training. Sixty iterations at 100 steps each
    is 6,000 real environment steps—the world model is learning
    on very little data. The full DreamerV3 trains for millions
    of steps. On top of that, the world model and the policy
    learn simultaneously, and when one improves, the other's
    data changes. When the world model gets better at predicting
    pixels, the imagined trajectories the actor trains on shift.
    When the actor improves, it visits new states the world
    model has not seen. This co-adaptation is a fundamental
    challenge of model-based reinforcement learning.
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

    print("""    The reconstruction loss drops steadily—the world model gets
    better at predicting pixels. The reward is non-monotonic,
    reflecting the co-adaptation described above: when the world
    model improves, the actor's imagined training data shifts;
    when the actor improves, it explores states the world model
    has not seen.
    """)

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
    eval_positions = [eval_env.position]  # include initial position

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
        eval_positions.append(eval_env.position)

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
        print("""    The greedy reward is close to a random policy, yet
    training reached -30 by iterations 30-44. The main reason
    is a distribution shift between training and evaluation.

    During imagination, the actor trains on GRU hidden states—
    the dynamics model's predictions of what comes next. During
    evaluation, the actor sees encoder outputs—the encoder's
    compression of actual pixels. These are different
    distributions. The actor learned a policy that works on
    imagined GRU states, then encounters encoder states at
    test time. The mismatch is enough to erase the training
    gains.

    The full DreamerV3 solves this with the RSSM (recurrent
    state-space model). At each step, the RSSM combines the
    GRU's prediction with the current observation to produce
    a consistent latent state used in both training and
    inference. Our simplified version skips this—the encoder
    and GRU produce separate representations that the actor
    must bridge, and it cannot.

    Three further simplifications compound the problem: the
    teacher forcing gap described earlier (the GRU never
    practices open-loop prediction); no uncertainty (the full
    DreamerV3 uses the gap between prediction and observation
    to measure trust, while our deterministic GRU commits
    fully to one estimate); and a minimal training budget
    that does not give the actor-critic enough imagined
    experience to converge.""")
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
    empty_path: list[tuple[float, float]] = []

    # --- Generate imagined rollout for trajectory comparison ---
    # Start from the same state as the real evaluation, then roll
    # the GRU forward without real observations. The actor chooses
    # actions in latent space, and we decode each step to find where
    # the model thinks the agent is.
    def _grid_to_pos(frame: Matrix, decoded: bool = False) -> tuple[float, float]:
        """Extract agent position from a pixel frame.

        Real frames: agent is the brightest pixel (1.0 > 0.7 target).
        Decoded frames: the decoder reconstructs the target more
        strongly (~0.7) than the agent (~0.35), so the agent is the
        second-brightest pixel—provided there is a clearly separate
        second peak. When agent and target overlap or the decoder
        produces a single blob, the brightest pixel is used instead.
        """
        pixels = []
        for r in range(SIZE):
            for c in range(SIZE):
                if frame[r][c] > 0.02:
                    pixels.append((frame[r][c], r, c))
        pixels.sort(key=lambda x: -x[0])

        if decoded and len(pixels) >= 2:
            p0, p1 = pixels[0], pixels[1]
            # Only use the second peak if it is clearly distinct:
            # different cell AND noticeably above background noise.
            same_cell = (p0[1] == p1[1] and p0[2] == p1[2])
            well_separated = p1[0] > 0.15  # above decoder noise floor
            if not same_cell and well_separated:
                pick = p1  # second-brightest is the agent
            else:
                pick = p0  # single blob or merged—use brightest
        elif pixels:
            pick = pixels[0]  # brightest is the agent (real frame)
        else:
            pick = (0.0, SIZE // 2, SIZE // 2)

        bounds = 2.0
        x = (pick[2] / (SIZE - 1)) * 2 * bounds - bounds
        y = (pick[1] / (SIZE - 1)) * 2 * bounds - bounds
        return (x, y)

    # Real trajectory positions (already collected during eval)
    real_positions = list(eval_positions)

    # Imagined trajectory: start from the first real observation,
    # then run the GRU forward without real pixels
    imag_env = PixelPointMass(max_steps=100)
    imag_state = imag_env.reset()
    h_imag, _ = network_forward(encoder, imag_state.features)
    imagined_positions: list[tuple[float, float]] = [_grid_to_pos(imag_env.render_frame())]

    for _step in range(min(len(real_positions), 50)):
        actor_out, _ = network_forward(actor_net, h_imag)
        mean_x, mean_y = actor_out[0], actor_out[1]
        action = [scalar.clamp(mean_x, -1.0, 1.0),
                  scalar.clamp(mean_y, -1.0, 1.0)]
        # GRU predicts next state from (current_latent, action)
        h_imag, _ = gru_forward(gru, h_imag, action)
        # Decode to pixels to find the predicted position
        recon, _ = network_forward(decoder, h_imag)
        recon_frame = matrix.reshape(recon, SIZE, SIZE)
        imagined_positions.append(_grid_to_pos(recon_frame, decoded=True))

    # Target position for drawing
    target_pos = (0.8, 0.8)  # PointMass default

    # --- Phase 1: Random agent trajectory ---
    from policywerk.primitives.random import create_rng as _create_rng
    rand_rng = _create_rng(99)
    rand_env = PixelPointMass(max_steps=50)
    state = rand_env.reset()
    rand_reward = 0.0
    rand_path: list[tuple[float, float]] = [rand_env.position]
    for step in range(30):
        action = [rand_rng.choice([-1.0, 1.0]), rand_rng.choice([-1.0, 1.0])]
        state, reward, done = rand_env.step_continuous(action)
        rand_reward += reward
        rand_path.append(rand_env.position)
        if step % 3 == 0:
            snapshots.append(DreamerSnapshot(
                episode=0, total_reward=rand_reward,
                real_path=list(rand_path), imagined_path=empty_path,
                real_frame=blank_frame, imagined_frame=blank_frame,
                step_label=f"1/3  Random policy (step {step})",
                phase="random",
            ))
        if done:
            break
    for _ in range(4):
        snapshots.append(DreamerSnapshot(
            episode=0, total_reward=rand_reward,
            real_path=list(rand_path), imagined_path=empty_path,
            real_frame=blank_frame, imagined_frame=blank_frame,
            step_label=f"1/3  Random: reward {rand_reward:.1f}",
            phase="random",
        ))

    # Save the final random walk for fading in Phase 2
    final_rand_path = list(rand_path)

    # --- Phase 2: Training progression (random walk stays faded) ---
    sample_iters = sorted(set(
        list(range(0, num_iterations, max(1, num_iterations // 10))) +
        [num_iterations - 1]
    ))
    for idx in sample_iters:
        h = history[idx]
        snapshots.append(DreamerSnapshot(
            episode=idx,
            total_reward=h["avg_reward"],
            real_path=list(final_rand_path),  # faded in renderer
            imagined_path=empty_path,
            real_frame=blank_frame, imagined_frame=blank_frame,
            step_label=f"Iteration {idx} of {num_iterations}",
            phase="training",
        ))

    # --- Phase 3: Real vs imagined trajectory, building step by step ---
    n_traj = min(len(real_positions), len(imagined_positions), 50)
    for step_idx in range(n_traj):
        snapshots.append(DreamerSnapshot(
            episode=num_iterations,
            total_reward=eval_reward,
            real_path=[(p[0], p[1]) for p in real_positions[:step_idx + 1]],
            imagined_path=list(imagined_positions[:step_idx + 1]),
            real_frame=blank_frame, imagined_frame=blank_frame,
            step_label=f"Step {step_idx}",
            phase="eval",
        ))
    for _ in range(12):
        snapshots.append(DreamerSnapshot(
            episode=num_iterations,
            total_reward=eval_reward,
            real_path=[(p[0], p[1]) for p in real_positions[:n_traj]],
            imagined_path=list(imagined_positions[:n_traj]),
            real_frame=blank_frame, imagined_frame=blank_frame,
            step_label=f"Final reward: {eval_reward:.1f}",
            phase="eval",
        ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 07: Dreamer",
        subtitle="Hafner et al. (2023) | learn the world, train in imagination",
        figsize=(12, 7),
    )

    recon_list = [h["recon_loss"] for h in history]

    from policywerk.viz.trajectories import draw_trajectory, draw_agent, draw_target

    # Compute trajectory bounds from all data so nothing clips
    all_pts = (list(rand_path) + list(real_positions)
               + list(imagined_positions) + [target_pos])
    max_x = max(abs(p[0]) for p in all_pts) * 1.15
    max_y = max(abs(p[1]) for p in all_pts) * 1.15
    traj_bound = max(max_x, max_y, 1.1)  # at least 1.1 to show target

    def _setup_trajectory_axes(ax):
        """Configure a clean 2D trajectory plot with data-driven bounds."""
        ax.set_xlim(-traj_bound * 0.3, traj_bound)
        ax.set_ylim(-traj_bound * 0.3, traj_bound)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.tick_params(labelsize=6, length=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#999999")
        draw_target(ax, target_pos)

    # --- Unified text template for the RHS pane ---
    def _draw_info(ax, phase_label, description, metrics):
        """Draw consistent text layout: label at top, description, metrics."""
        ax.clear()
        ax.axis("off")
        # Phase label (bold, top)
        ax.text(0.5, 0.92, phase_label,
                transform=ax.transAxes, ha="center", va="top",
                fontsize=11, fontweight="bold", color=DARK_GRAY)
        # Description (middle)
        ax.text(0.5, 0.58, description,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color=DARK_GRAY)
        # Metrics (bottom, monospace)
        if metrics:
            ax.text(0.5, 0.12, metrics,
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8, color=DARK_GRAY, fontfamily="monospace")

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # ---- Left pane: 2D trajectory (always present) ----
        axes["env"].clear()
        _setup_trajectory_axes(axes["env"])

        if snap.phase == "random":
            # Random walk building up
            if snap.real_path:
                draw_trajectory(axes["env"], snap.real_path, color=TEAL, linewidth=1.5)
                draw_agent(axes["env"], snap.real_path[-1], color=TEAL)
        elif snap.phase == "training":
            # Random walk stays visible but faded
            if snap.real_path:
                draw_trajectory(axes["env"], snap.real_path,
                                color=TEAL, linewidth=1.0, alpha=0.2)
        else:
            # Real path (solid teal) and imagined path (dashed orange)
            if snap.real_path:
                draw_trajectory(axes["env"], snap.real_path, color=TEAL, linewidth=2.0)
                draw_agent(axes["env"], snap.real_path[-1], color=TEAL)
            if snap.imagined_path:
                xs = [p[0] for p in snap.imagined_path]
                ys = [p[1] for p in snap.imagined_path]
                axes["env"].plot(xs, ys, color=ORANGE, linewidth=2.0,
                                 linestyle="--", alpha=0.8, clip_on=True)
                axes["env"].scatter([xs[-1]], [ys[-1]], c=ORANGE, s=60,
                                    zorder=10, edgecolors=DARK_GRAY, linewidths=0.5)

        axes["env"].set_title(snap.step_label, fontsize=9)

        # ---- Right pane: consistent text template ----
        if snap.phase == "random":
            _draw_info(axes["algo"],
                       "Before Training",
                       "The agent acts randomly.\n"
                       "No world model exists yet.",
                       f"Reward: {snap.total_reward:+.1f}")
        elif snap.phase == "training":
            _draw_info(axes["algo"],
                       "Learning the World",
                       "The world model learns to\n"
                       "predict next states and rewards\n"
                       "from pixel observations.",
                       f"Iteration:  {snap.episode} / {num_iterations}\n"
                       f"Recon MSE:  {history[snap.episode]['recon_loss']:.4f}\n"
                       f"Reward:     {snap.total_reward:+.1f}")
        else:
            _draw_info(axes["algo"],
                       "Real vs Imagined",
                       "\u2500\u2500  Real path (teal)\n"
                       "- -  Imagined path (orange)\n"
                       "\u2605   Target\n\n"
                       "The agent trained on imagined\n"
                       "paths, then acted in the real\n"
                       "world. The gap shows where\n"
                       "predictions diverge from reality.",
                       f"Reward: {snap.total_reward:+.1f}")

        # ---- Bottom pane: loss trace ----
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

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/07_dreamer_artifact.gif", fps=6)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 07: Dreamer",
        subtitle="Hafner et al. (2023)",
        figsize=(12, 7),
    )

    # Poster matches Phase 3: trajectory comparison + text
    def update_poster(frame_idx):
        # Left: full trajectory comparison
        _setup_trajectory_axes(axes2["env"])
        real_pts = [(p[0], p[1]) for p in real_positions]
        draw_trajectory(axes2["env"], real_pts, color=TEAL, linewidth=2.0)
        if real_pts:
            draw_agent(axes2["env"], real_pts[-1], color=TEAL)
        xs = [p[0] for p in imagined_positions]
        ys = [p[1] for p in imagined_positions]
        axes2["env"].plot(xs, ys, color=ORANGE, linewidth=2.0,
                          linestyle="--", alpha=0.8, clip_on=True)
        if xs:
            axes2["env"].scatter([xs[-1]], [ys[-1]], c=ORANGE, s=60,
                                 zorder=10, edgecolors=DARK_GRAY, linewidths=0.5)
        axes2["env"].set_title("Real vs Imagined Trajectory", fontsize=10)

        # Right: pixel reconstruction comparison + legend
        axes2["algo"].clear()
        axes2["algo"].axis("off")

        # Title
        axes2["algo"].text(0.5, 0.97, "Trained Agent",
                           transform=axes2["algo"].transAxes, ha="center", va="top",
                           fontsize=11, fontweight="bold", color=DARK_GRAY)

        # Show real vs reconstructed pixel frames side by side
        mid = min(len(eval_real_frames), len(eval_imagined_frames)) // 2
        if mid > 0:
            # Real frame
            ax_real = axes2["algo"].inset_axes([0.05, 0.48, 0.4, 0.4])
            ax_real.imshow(eval_real_frames[mid], cmap="gray", vmin=0, vmax=1,
                           origin="upper", interpolation="nearest")
            ax_real.set_title("Real", fontsize=8, pad=3, color=DARK_GRAY)
            ax_real.set_xticks([])
            ax_real.set_yticks([])

            # Reconstructed frame
            ax_recon = axes2["algo"].inset_axes([0.55, 0.48, 0.4, 0.4])
            ax_recon.imshow(eval_imagined_frames[mid], cmap="gray", vmin=0, vmax=1,
                            origin="upper", interpolation="nearest")
            ax_recon.set_title("Reconstruction", fontsize=8, pad=3, color=DARK_GRAY)
            ax_recon.set_xticks([])
            ax_recon.set_yticks([])

        # Legend + metrics below
        axes2["algo"].text(0.5, 0.35,
                           "\u2500\u2500  Real path (teal)\n"
                           "- -  Imagined path (orange)\n"
                           "\u2605   Target",
                           transform=axes2["algo"].transAxes, ha="center", va="top",
                           fontsize=9, color=DARK_GRAY)
        axes2["algo"].text(0.5, 0.08,
                           f"Reward:     {eval_reward:+.1f}\n"
                           f"Recon MSE:  {avg_recon:.4f}",
                           transform=axes2["algo"].transAxes, ha="center", va="bottom",
                           fontsize=8, color=DARK_GRAY, fontfamily="monospace")

        # Bottom: loss + reward curves
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
    ax_r.set_title("Dreamer Training on Pixel World", fontsize=10)
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
    print("  Dreamer closed the final gap in this series: learning")
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
