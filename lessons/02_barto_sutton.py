"""Lesson 2: The Actor-Critic (Barto, Sutton & Anderson, 1983).

In 1983, Barto, Sutton, and Anderson showed that two simple neurons
could learn to balance a pole through trial and error alone. Unlike
Lesson 01, the agent has no access to the environment's rules. It
must discover good behavior by acting, observing consequences, and
adjusting.

This is the first lesson where the agent learns from experience.

Run: uv run python lessons/02_barto_sutton.py
"""

import os
from dataclasses import dataclass

from policywerk.world.balance import Balance
from policywerk.actors.barto_sutton import train, create_ace_ase
from policywerk.primitives.progress import Spinner
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.traces import update_trace_axes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot
# ---------------------------------------------------------------------------

@dataclass
class BartoSuttonSnapshot(FrameSnapshot):
    angles: list[float]      # angle at each step of the episode
    episode_length: int
    episode_num: int


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_weight_grid(weights: list[float], label: str, rows: int = 6, cols: int = 6) -> None:
    """Print ACE or ASE weights as a grid (angle bins x velocity bins)."""
    print(f"    {label}:")
    print(f"    {'':8s}", end="")
    for c in range(cols):
        print(f"  v{c:d}   ", end="")
    print()
    for r in range(rows):
        print(f"    a{r:d}  ", end="")
        for c in range(cols):
            idx = r * cols + c
            val = weights[idx]
            print(f"{val:+6.2f} ", end="")
        print()
    print()


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Lesson 02: The Actor-Critic (1983)")
    print("  Barto, Sutton & Anderson,")
    print("  'Neuronlike Adaptive Elements'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. The shift from planning to learning
    # -----------------------------------------------------------------------

    print("FROM PLANNING TO LEARNING")
    print("-" * 64)
    print("""
    In Lesson 01, the agent had a map. It knew every cell, every
    wall, every reward. It could plan the optimal path without
    taking a single step.

    Now the map is gone. The agent faces a balance task: keep a
    pole upright by pushing left or right. It does not know the
    physics. It does not know what "left" or "right" will do. It
    only knows whether it succeeded (the pole stayed up) or
    failed (the pole fell).

    This is the fundamental shift: from reasoning about a known
    model to learning from consequences. Every RL algorithm from
    here onward works this way.
    """)

    # -----------------------------------------------------------------------
    # 2. The environment
    # -----------------------------------------------------------------------

    print("THE BALANCE TASK")
    print("-" * 64)
    print("""
    A pole is hinged at a point. The agent applies left or right
    torque to keep it upright. Think of balancing a broomstick on
    your fingertip.

    State:   (angle, angular velocity) -- two continuous numbers
    Actions: 0 = push left, 1 = push right
    Reward:  +1 per step of survival, 0 when the pole falls
    Done:    when |angle| > 0.3 radians (about 17 degrees)
    Goal:    survive 500 steps

    The continuous state is discretized into bins:
      6 angle bins x 6 velocity bins = 36 "boxes"

    Each box represents a region of the state space. The agent
    learns a separate weight for each box, like having 36 slots
    in a lookup table. The question is: how does it fill them in
    without knowing the physics?
    """)

    env = Balance()

    # -----------------------------------------------------------------------
    # 3. The architecture
    # -----------------------------------------------------------------------

    print("THE ACE/ASE ARCHITECTURE")
    print("-" * 64)
    print("""
    The 1983 paper introduced two "neuronlike elements" that work
    together:

      ASE (Adaptive Search Element) -- the ACTOR
        Decides what to do. For each box, it has a weight: positive
        favors pushing right, negative favors pushing left. The
        action is: compute the weighted sum, add some random noise
        (for exploration), and threshold: >= 0 means push right,
        < 0 means push left.

      ACE (Adaptive Critic Element) -- the CRITIC
        Predicts how good the current state is. For each box, it
        stores an estimated value (like V(s) from Lesson 01, but
        learned from experience, not computed from a model).

    The critic drives the actor's learning through the TD error:

      td_error = reward + gamma * prediction(now) - prediction(before)

    This is the critic's "surprise signal":
      Positive: "things turned out better than I predicted"
      Negative: "things turned out worse than I predicted"

    When the pole falls, the reward is -1 and the prediction drops
    to zero, producing a large negative TD error. That negative
    signal flows backward through the eligibility traces, reducing
    the weights of actions that contributed to the failure.

    Eligibility traces remember which boxes were recently active.
    When the TD error arrives, boxes with high traces get updated
    more than boxes visited long ago. This solves the credit
    assignment problem: which of the last 50 actions caused the
    pole to fall?

    Here is a concrete example. Suppose the agent is in box (3,3)
    (centered angle, low velocity). It pushes right (action 1).
    The ASE trace for box 15 (= 3*6 + 3) increases. One step later,
    the pole tilts right and the agent enters box (3,4). If the
    critic's prediction was too optimistic, the TD error is negative,
    and the ASE weight for box 15 decreases, making "push right from
    centered" less likely next time.
    """)

    # -----------------------------------------------------------------------
    # 4. Training
    # -----------------------------------------------------------------------

    print("TRAINING")
    print("-" * 64)
    print("""
    Training runs for 100 episodes. Each episode starts with the
    pole slightly tilted and ends when the pole falls or 500 steps
    pass (success).

    Parameters:
      gamma      = 0.95  (discount factor)
      alpha      = 10.0  (actor learning rate)
      beta       = 0.1   (critic learning rate)
      trace_decay = 0.5  (eligibility trace decay)
      noise_std  = 0.1   (exploration noise)
    """)

    num_episodes = 100
    ace, ase, lengths, all_angles = train(
        env, num_episodes=num_episodes, seed=42,
        gamma=0.95, alpha=10.0, beta=0.1,
        trace_decay=0.5, noise_std=0.1,
    )

    # Show training progress
    print("    Episode lengths (how long the pole stayed up):")
    for i in range(0, num_episodes, 10):
        chunk = lengths[i:i + 10]
        avg = sum(chunk) / len(chunk)
        bar_len = int(avg / 500 * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"      Episodes {i:3d}-{i+9:3d}:  avg {avg:5.0f} steps  [{bar}]")
    print()

    first_success = next((i for i, l in enumerate(lengths) if l >= 500), None)
    if first_success is not None:
        print(f"    First successful balance (500 steps): episode {first_success}")
    print(f"    Final 10 episodes average: {sum(lengths[-10:])/10:.0f} steps")
    print()

    print("""    The agent learns quickly. Early episodes end in seconds as the
    pole topples. Within a few episodes, the agent discovers that
    pushing against the direction of tilt keeps the pole up. By
    episode 10, it can balance indefinitely.
    """)

    # -----------------------------------------------------------------------
    # 5. Learned weights
    # -----------------------------------------------------------------------

    print("LEARNED WEIGHTS")
    print("-" * 64)
    print("""
    The ACE and ASE each have 36 weights (one per box). Here is
    what they learned. Rows are angle bins (0=far left, 5=far right).
    Columns are velocity bins (0=fast left, 5=fast right).
    """)

    print_weight_grid(ase.weights, "ASE (actor) weights -- positive = favor right")
    print_weight_grid(ace.weights, "ACE (critic) weights -- higher = better state")

    print("""    Reading the actor weights: when the pole tilts left (low angle
    bins), the weights are positive (push right to correct). When it
    tilts right (high angle bins), the weights are negative (push
    left). The agent learned the obvious strategy: push against the
    lean.

    Reading the critic weights: the center boxes (balanced pole)
    have the highest values. The edge boxes (pole about to fall)
    have the lowest. The critic learned that being centered is
    good and being tilted is bad.
    """)

    # -----------------------------------------------------------------------
    # 6. Comparison to Lesson 01
    # -----------------------------------------------------------------------

    print("COMPARISON TO LESSON 01")
    print("-" * 64)
    print("""
    Lesson 01 (Bellman):
      - Knew the environment's rules completely
      - Computed optimal values by reasoning, no interaction needed
      - Guaranteed to find the optimal solution
      - Only works when you have the model

    Lesson 02 (Barto/Sutton):
      - Knew nothing about the physics
      - Learned by trial and error over 100 episodes
      - Found a good (not provably optimal) policy
      - Works with any environment you can interact with

    The price of not having a model: the agent must explore, fail,
    and learn from failures. But the payoff is generality. The same
    actor-critic framework works whether the environment is a grid,
    a pole, a game, or a robot.
    """)

    # -----------------------------------------------------------------------
    # 7. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots: sample episodes at intervals
    snapshots = []
    record_interval = max(1, num_episodes // 30)  # ~30 frames
    for ep_idx in range(num_episodes):
        if ep_idx % record_interval == 0 or ep_idx == num_episodes - 1:
            snapshots.append(BartoSuttonSnapshot(
                episode=ep_idx,
                total_reward=float(lengths[ep_idx]),
                angles=all_angles[ep_idx],
                episode_length=lengths[ep_idx],
                episode_num=ep_idx,
            ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 02: Actor-Critic Balance",
        subtitle="Barto/Sutton/Anderson (1983) -- chaotic motion becomes control",
    )

    # Real episode lengths for the trace (no fake points)
    real_lengths = list(lengths)

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: angle trajectory for this episode
        axes["env"].clear()
        steps = list(range(len(snap.angles)))
        axes["env"].plot(steps, snap.angles, color=TEAL, linewidth=1.0, alpha=0.8)
        axes["env"].axhline(0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        axes["env"].axhline(-0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        axes["env"].axhline(0.0, color=DARK_GRAY, linestyle="-", linewidth=0.5, alpha=0.3)
        axes["env"].set_xlim(0, 500)
        axes["env"].set_ylim(-0.35, 0.35)
        axes["env"].set_xlabel("Step", fontsize=9)
        axes["env"].set_ylabel("Angle (rad)", fontsize=9)
        axes["env"].set_title(f"Episode {snap.episode_num} -- {snap.episode_length} steps", fontsize=10)

        # Top-right: episode info
        axes["algo"].clear()
        axes["algo"].axis("off")
        info_lines = [
            f"Episode: {snap.episode_num}",
            f"Length:  {snap.episode_length} steps",
            f"Status:  {'balanced!' if snap.episode_length >= 500 else 'fell'}",
            "",
            f"gamma:       0.95",
            f"alpha (ASE): 10.0",
            f"beta (ACE):  0.1",
        ]
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title("Training State", fontsize=10)

        # Bottom: episode length over training
        n = min(snap.episode_num + 1, len(real_lengths))
        trace_data = real_lengths[:n]
        update_trace_axes(axes["trace"], trace_data,
                          label="Episode length", color=TEAL)
        axes["trace"].set_ylabel("Steps", fontsize=9)
        axes["trace"].set_xlabel("Episode", fontsize=9)
        axes["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/02_barto_sutton_artifact.gif", fps=3)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 02: Actor-Critic Balance (Trained)",
        subtitle="Barto/Sutton/Anderson (1983)",
    )

    def update_poster(frame_idx):
        snap = snapshots[-1]
        # Show final episode angle trace
        steps = list(range(len(snap.angles)))
        axes2["env"].plot(steps, snap.angles, color=TEAL, linewidth=1.0, alpha=0.8)
        axes2["env"].axhline(0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        axes2["env"].axhline(-0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        axes2["env"].axhline(0.0, color=DARK_GRAY, linestyle="-", linewidth=0.5, alpha=0.3)
        axes2["env"].set_xlim(0, 500)
        axes2["env"].set_ylim(-0.35, 0.35)
        axes2["env"].set_xlabel("Step", fontsize=9)
        axes2["env"].set_ylabel("Angle (rad)", fontsize=9)
        axes2["env"].set_title("Final Episode (500 steps)", fontsize=10)

        axes2["algo"].clear()
        axes2["algo"].axis("off")
        info = (
            f"Learned in ~{first_success or '?'} episodes\n"
            f"36 boxes (6x6 discretization)\n"
            f"2 neurons: actor + critic\n\n"
            f"The same balance task appears\n"
            f"in Lesson 06 with PPO --\n"
            f"34 years of progress."
        )
        axes2["algo"].text(0.1, 0.9, info, transform=axes2["algo"].transAxes,
                           fontsize=10, verticalalignment="top",
                           fontfamily="monospace", color=DARK_GRAY)
        axes2["algo"].set_title("Summary", fontsize=10)

        update_trace_axes(axes2["trace"], real_lengths,
                          label="Episode length", color=TEAL)
        axes2["trace"].set_ylabel("Steps", fontsize=9)
        axes2["trace"].set_xlabel("Episode", fontsize=9)
        axes2["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/02_barto_sutton_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Learning curve ---

    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    ax3.plot(range(num_episodes), lengths,
             color=TEAL, linewidth=1.0, alpha=0.8, label="Episode length")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Steps")
    ax3.set_title("Learning Curve: Episode Length over Training")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    with Spinner("Generating trace"):
        save_figure(fig3, "output/02_barto_sutton_trace.png")

    print()
    print("    Artifacts saved to output/:")
    print("      artifact.gif  angle trajectories across training episodes")
    print("      artifact.pdf  PDF storyboard of every frame")
    print("      poster.png    final balanced episode with learning curve")
    print("      trace.png     episode length over training")

    # -----------------------------------------------------------------------
    # Closing
    # -----------------------------------------------------------------------

    print()
    print("=" * 64)
    print("  Lesson 02 complete.")
    print()
    print("  The actor-critic learned to balance through consequences.")
    print("  But both the actor and the critic used eligibility traces")
    print("  and TD-like updates. What exactly IS temporal-difference")
    print("  learning? In Lesson 03, Sutton isolates the idea: learning")
    print("  to predict by the methods of temporal differences.")
    print("=" * 64)


if __name__ == "__main__":
    main()
