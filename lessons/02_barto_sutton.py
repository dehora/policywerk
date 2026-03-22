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
import math
from dataclasses import dataclass

from policywerk.world.balance import Balance
from policywerk.actors.barto_sutton import train, create_ace_ase
from policywerk.primitives.progress import Spinner
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY, LIGHT_GRAY,
)
from policywerk.viz.traces import update_trace_axes
from policywerk.viz.trajectories import draw_pole

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot
# ---------------------------------------------------------------------------

@dataclass
class BartoSuttonSnapshot(FrameSnapshot):
    angles: list[float]      # angle at each step of the episode
    actions: list[int]       # action at each step (for pole viz)
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

    State:   (angle, angular velocity)
    Actions: 0 = push left, 1 = push right
    Reward:  +1 per step of survival, 0 when the pole falls
    Done:    when |angle| > 0.3 radians (about 17 degrees)
    Goal:    survive 500 steps

    The continuous state is discretized into bins:

      Angle bins:    a0 (< -0.2)  a1  a2  a3  a4  a5 (> +0.2)
                     far left  <-- centered -->  far right

      Velocity bins: v0 (< -1.0)  v1  v2  v3  v4  v5 (> +1.0)
                     fast left <-- still -->  fast right

      Total: 6 angle bins x 6 velocity bins = 36 "boxes"

    Each box gets one weight in the actor and one in the critic.
    The input is a "one-hot" vector: 36 numbers, all zeros except
    a 1.0 at the current box. For example, if angle is in bin 3
    and velocity in bin 2, the box index is 3*6 + 2 = 20, so only
    weight 20 is active.
    """)

    env = Balance()

    # -----------------------------------------------------------------------
    # 3. The architecture
    # -----------------------------------------------------------------------

    print("THE ACE/ASE ARCHITECTURE")
    print("-" * 64)
    print("""
    The 1983 paper introduced two "neuronlike elements" that work
    together. Each is a single neuron with 36 weights:

      ASE (Adaptive Search Element) -- the ACTOR
        Decides what to do. It computes: weighted_sum + noise.
        If the result >= 0, push right. Otherwise, push left.
        The noise is deliberate -- without it, the agent would
        always do the same thing from the same state and could
        never discover alternatives.

      ACE (Adaptive Critic Element) -- the CRITIC
        Predicts how good the current state is, like V(s) from
        Lesson 01 but learned from experience, not computed from
        a model. Its prediction for a box is simply the weight
        for that box (since the input is one-hot).

    The critic drives learning through the TD error:

      td_error = reward + gamma * prediction(now) - prediction(before)

    This is the critic's "surprise signal":
      Positive: "things turned out better than I predicted"
      Negative: "things turned out worse than I predicted"
      Zero:     "things went exactly as expected, nothing to learn"

    The environment gives reward +1 per step and 0 on failure. For
    learning, we transform this to the paper's convention: 0 during
    balancing, -1 on failure. This way the TD error is near zero
    while the pole is up, and sharply negative when it falls.
    """)

    print("""    WHAT HAPPENS WHEN THE POLE FALLS

    Suppose the agent has been balancing for 50 steps, then the
    pole falls at step 51.

      Step 50: state is box 22. Critic predicts V = 0.3.
      Step 51: pole falls. Reward = -1. No future state.
               TD error = -1 - 0.3 = -1.3

    This large negative TD error updates every box that has a
    non-zero eligibility trace:

      Box 22 (visited 1 step ago):  trace = 0.5^1 = 0.50
        Actor weight update: w[22] += 10.0 * (-1.3) * 0.50 = -6.5
        Critic weight update: v[22] += 0.1 * (-1.3) * 0.50 = -0.065

      Box 21 (visited 2 steps ago): trace = 0.5^2 = 0.25
        Actor weight update: w[21] += 10.0 * (-1.3) * 0.25 = -3.25

      Box 18 (visited 5 steps ago): trace = 0.5^5 = 0.03
        Actor weight update: w[18] += 10.0 * (-1.3) * 0.03 = -0.39

    Recent boxes get large updates (they were probably responsible),
    distant boxes get small ones. This is how eligibility traces
    solve credit assignment.
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
      gamma       = 0.95  discount factor (how much to value future)
      alpha       = 10.0  actor learning rate (large because one-hot
                          input means only one weight changes per step)
      beta        = 0.1   critic learning rate (learns conservatively)
      trace_decay = 0.5   traces halve each step -- a state visited 3
                          steps ago has trace 0.5^3 = 0.125 (12.5% of
                          the update of the current state)
      noise_std   = 0.1   exploration noise -- small perturbation so
                          the agent occasionally tries the other action
    """)

    num_episodes = 100
    ace, ase, lengths, all_angles, all_actions = train(
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

    print("""    The agent learns quickly. Early episodes end as the pole
    topples. Within a few episodes, the large negative TD errors
    from failures have pushed the actor weights in the right
    direction. By episode 10, it balances indefinitely.

    The original 1983 paper used full cart-pole (4 state variables,
    162 boxes) and needed roughly 100 episodes. Our simplified
    balance (2 variables, 36 boxes) converges faster because the
    strategy is simpler and there are fewer states to learn. But
    the mechanism is the same -- TD error flowing through eligibility
    traces -- and it scales to harder problems.
    """)

    # -----------------------------------------------------------------------
    # 5. Learned weights
    # -----------------------------------------------------------------------

    print("LEARNED WEIGHTS")
    print("-" * 64)
    print("""
    The ACE and ASE each have 36 weights (one per box).
    Rows = angle bins (a0=far left, a5=far right).
    Columns = velocity bins (v0=fast left, v5=fast right).
    """)

    print_weight_grid(ase.weights, "ASE (actor) -- positive = favor push right")
    print_weight_grid(ace.weights, "ACE (critic) -- higher = better state")

    # Describe specific weights, handling the zero case honestly
    def weight_direction(w):
        if w > 0.01:
            return "favor right"
        elif w < -0.01:
            return "favor left"
        else:
            return "no preference (unvisited)"

    a1v1 = ase.weights[7]   # a1,v1: tilted left, moving left
    a4v3 = ase.weights[27]  # a4,v3: tilted right, moving right

    print(f"""    Reading the actor: look at the inner rows where the agent
    actually spends time (a1-a4, near center):

      a1,v1: weight {a1v1:+.2f} ({weight_direction(a1v1)}
             when tilted slightly left)
      a4,v3: weight {a4v3:+.2f} ({weight_direction(a4v3)}
             when tilted slightly right)

    In the frequently visited center rows, the pattern is clear:
    positive weights when tilted left (push right to correct),
    negative weights when tilted right (push left to correct).

    The extreme rows (a0, a5) have large noisy weights because
    the agent rarely visits them. Those bins represent states
    where the pole is nearly falling -- the few experiences there
    produce outsized weight updates that do not reflect a stable
    learned strategy.

    Many boxes show 0.00 because the agent never visited them.
    Once it learns to balance, it stays near center.

    Reading the critic: center boxes (a2-a3) have values near zero
    (balanced is "normal"). Edge boxes (a0, a5) have negative
    values (tilted far = danger). The critic learned that being
    centered is safe and being tilted is risky.
    """)

    # -----------------------------------------------------------------------
    # 6. Comparison
    # -----------------------------------------------------------------------

    print("COMPARISON TO LESSON 01")
    print("-" * 64)
    print("""
    Lesson 01 (Bellman):            Lesson 02 (Barto/Sutton):
      Knew the rules completely       Knew nothing about physics
      Computed values by reasoning    Learned by trial and error
      Guaranteed optimal solution     Found a good policy
      Only works with a model         Works with any environment

    The price of not having a model: the agent must explore, fail,
    and learn from failures. But the payoff is generality.

    The TD error -- the critic's surprise signal -- is the concept
    to carry forward from here. In Lesson 03, we study it in
    isolation as temporal-difference learning. In Lesson 04, we
    use it for Q-learning. The actor-critic split returns in Lesson
    06 with neural networks instead of single neurons.
    """)

    # -----------------------------------------------------------------------
    # 7. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots: sample episodes at intervals
    snapshots = []
    # More frames early (where the action is), fewer later
    early_episodes = list(range(0, min(15, num_episodes)))
    later_episodes = list(range(15, num_episodes, 5))
    sample_episodes = sorted(set(early_episodes + later_episodes + [num_episodes - 1]))

    for ep_idx in sample_episodes:
        if ep_idx < num_episodes:
            snapshots.append(BartoSuttonSnapshot(
                episode=ep_idx,
                total_reward=float(lengths[ep_idx]),
                angles=all_angles[ep_idx],
                actions=all_actions[ep_idx],
                episode_length=lengths[ep_idx],
                episode_num=ep_idx,
            ))

    # --- Artifact 1: Animation with pole visualization ---

    fig, axes = create_lesson_figure(
        "Lesson 02: Actor-Critic Balance",
        subtitle="Barto/Sutton/Anderson (1983) -- chaotic motion becomes control",
    )

    real_lengths = list(lengths)

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: draw the pole at a representative moment
        axes["env"].clear()
        # Show the pole at the midpoint of the episode (or last step if short)
        mid = min(len(snap.angles) - 1, len(snap.angles) // 2)
        angle = snap.angles[mid]
        action = snap.actions[mid] if mid < len(snap.actions) else None
        draw_pole(axes["env"], angle, action=action)
        status = "balanced!" if snap.episode_length >= 500 else f"fell at step {snap.episode_length}"
        axes["env"].set_title(f"Episode {snap.episode_num} -- {status}", fontsize=10)

        # Top-right: angle trajectory over time
        axes["algo"].clear()
        steps = list(range(len(snap.angles)))
        axes["algo"].plot(steps, snap.angles, color=TEAL, linewidth=0.8, alpha=0.8)
        axes["algo"].axhline(0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
        axes["algo"].axhline(-0.3, color="red", linestyle="--", linewidth=0.8, alpha=0.4)
        axes["algo"].axhline(0.0, color=DARK_GRAY, linestyle="-", linewidth=0.5, alpha=0.3)
        axes["algo"].set_xlim(0, 500)
        axes["algo"].set_ylim(-0.35, 0.35)
        axes["algo"].set_xlabel("Step", fontsize=8)
        axes["algo"].set_ylabel("Angle", fontsize=8)
        axes["algo"].set_title("Angle Trajectory", fontsize=10)

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

    # --- Artifact 2: Poster (evaluation episode, no exploration noise) ---

    # Run a clean evaluation episode with noise disabled to show the
    # trained policy's true behavior, not a noisy training rollout.
    from policywerk.actors.barto_sutton import train_episode
    from policywerk.primitives.random import create_rng as _create_rng
    eval_rng = _create_rng(0)
    eval_env = Balance()
    eval_ep, eval_angles, eval_actions = train_episode(
        eval_env, ace, ase, eval_rng, 36,
        gamma=0.95, alpha=0.0, beta=0.0,  # no learning during eval
        trace_decay=0.5, noise_std=0.0,    # no exploration noise
    )
    eval_length = len(eval_ep)

    fig2, axes2 = create_lesson_figure(
        "Lesson 02: Actor-Critic Balance (Trained)",
        subtitle="Barto/Sutton/Anderson (1983)",
    )

    def update_poster(frame_idx):
        # Show the pole at its midpoint angle from the eval episode
        mid = min(len(eval_angles) - 1, len(eval_angles) // 2)
        draw_pole(axes2["env"], eval_angles[mid])
        eval_status = f"{eval_length} steps" if eval_length >= 500 else f"fell at step {eval_length}"
        axes2["env"].set_title(f"Evaluation ({eval_status})", fontsize=10)

        axes2["algo"].clear()
        axes2["algo"].axis("off")
        info = (
            f"Learned in ~{first_success or '?'} episodes\n"
            f"36 boxes (6x6 discretization)\n"
            f"2 neurons: actor + critic\n\n"
            f"Evaluation: {eval_status}\n"
            f"(no exploration noise)\n\n"
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
    print("      02_barto_sutton_artifact.gif  pole balance across training")
    print("      02_barto_sutton_artifact.pdf  PDF storyboard of every frame")
    print("      02_barto_sutton_poster.png    evaluation episode (no noise)")
    print("      02_barto_sutton_trace.png     episode length over training")

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
