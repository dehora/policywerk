"""Lesson 1: The Bellman Equation (Bellman, 1957).

In 1957, Richard Bellman formalized how to make optimal decisions
in sequential problems. This lesson implements his two dynamic
programming algorithms — value iteration and policy iteration —
on a small gridworld, and animates the process of value propagation.

Run: uv run python lessons/01_bellman.py
"""

import os
from dataclasses import dataclass

from policywerk.world.gridworld import GridWorld
from policywerk.actors.bellman import value_iteration, policy_iteration, extract_policy
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.values import draw_value_heatmap, draw_policy_arrows, draw_grid_overlay
from policywerk.viz.traces import plot_training_traces

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot — extends FrameSnapshot with per-sweep data
# ---------------------------------------------------------------------------

@dataclass
class BellmanSnapshot(FrameSnapshot):
    values: list[list[float]]       # 5×5 grid of current V values
    max_change: float               # convergence metric
    policy: dict[str, int] | None   # greedy policy (final frames only)
    sweep: int


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_grid_values(env: GridWorld, values: list[list[float]]) -> None:
    """Print the value function as a formatted grid."""
    for r in range(len(values)):
        cells = []
        for c in range(len(values[0])):
            val = values[r][c]
            if env._grid[r][c] == 1:  # WALL
                cells.append("  ██  ")
            else:
                cells.append(f"{val:+.3f}")
        print("    " + "  ".join(cells))


def print_policy(policy: dict[str, int], rows: int, cols: int) -> None:
    """Print the policy as a grid of directional arrows."""
    arrows = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    for r in range(rows):
        cells = []
        for c in range(cols):
            label = f"{r},{c}"
            if label in policy:
                cells.append(f"  {arrows[policy[label]]}   ")
            else:
                cells.append("  ·   ")
        print("    " + " ".join(cells))


def update_sweep_trace(ax, values, label="", color=TEAL):
    """Update the convergence trace pane — like update_trace_axes but with
    'Sweep' on the x-axis and integer ticks."""
    ax.clear()
    # x values are sweep numbers starting at 0
    xs = list(range(len(values)))
    ax.plot(xs, values, color=color, linewidth=1.5, label=label, alpha=0.8,
            marker="o", markersize=4)
    ax.set_xlabel("Sweep", fontsize=9)
    ax.set_xticks(xs)
    if label:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Lesson 01: The Bellman Equation (1957)")
    print("  Richard Bellman — 'A Markovian Decision Process'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. The environment
    # -----------------------------------------------------------------------

    print("THE GRIDWORLD")
    print("-" * 64)
    print("""
    The agent lives in a 5×5 grid. It can move North, East, South,
    or West. Hitting a wall or boundary leaves it in place.

    Layout:
      . . . . G     G = goal (+1 reward, episode ends)
      . W . X .     W = wall (can't enter)
      . . . . .     X = pit (-1 reward, episode ends)
      . . . . .     . = empty (-0.04 per step)
      S . . . .     S = start

    The step cost of -0.04 encourages the agent to reach the goal
    quickly rather than wandering. The question is: from any cell,
    what's the best direction to go?
    """)

    env = GridWorld()
    gamma = 0.9  # discount: future rewards worth 90% of present ones
    theta = 0.001  # stop when values change by less than this

    # -----------------------------------------------------------------------
    # 2. Value iteration
    # -----------------------------------------------------------------------

    print("VALUE ITERATION")
    print("-" * 64)
    print("""
    Value iteration answers: "how good is each state?" It works by
    repeatedly sweeping through every state, asking:

      V(s) = best action's [expected reward + γ × value of next state]

    On the first sweep, only states next to the goal or pit get
    meaningful values. On the second sweep, their neighbors update.
    Gradually, information about rewards "ripples" backward through
    the grid until every state knows how good it is.
    """)

    V_vi, history = value_iteration(env, gamma=gamma, theta=theta)

    print(f"    Converged in {len(history)} sweeps.")
    print()

    # Show convergence progress with proportional bars
    print("    Sweep-by-sweep convergence (max value change):")
    max_initial = history[0]["max_change"] if history else 1.0
    for record in history:
        # Scale bars proportionally so the descent is visible
        fraction = record["max_change"] / max_initial if max_initial > 0 else 0
        bar_len = int(fraction * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"      Sweep {record['sweep']:2d}:  {record['max_change']:.6f}  [{bar}]")
    print()

    print(f"""    The grid is 5 columns wide, and information travels one column
    per sweep. After {len(history)} sweeps, the reward signal from the goal
    has reached every corner of the grid. The max change drops to
    zero when all values have fully stabilized.
    """)

    # Show final values
    print("    Final state values:")
    grid_values = env.grid_values(V_vi)
    print_grid_values(env, grid_values)
    print()

    # Derive and show the greedy policy
    policy_vi = extract_policy(env, V_vi, gamma=gamma)
    print("    Optimal policy (derived from values):")
    print_policy(policy_vi, 5, 5)
    print()

    print("""    The arrows show the greedy policy: at each state, go in the
    direction that leads to the highest expected value. Notice how
    the arrows point toward the goal and away from the pit — the
    values encode the consequences of each path.
    """)

    # -----------------------------------------------------------------------
    # 3. Policy iteration
    # -----------------------------------------------------------------------

    print("POLICY ITERATION")
    print("-" * 64)
    print("""
    Policy iteration takes a different approach. Instead of finding
    the optimal values directly, it alternates between two steps:

      1. Policy evaluation: "how good is my current plan?"
         Compute V(s) for the current policy until stable.

      2. Policy improvement: "can I do better?"
         For each state, switch to the best action under the new values.

    Repeat until the policy stops changing.
    """)

    V_pi, policy_pi, iterations = policy_iteration(env, gamma=gamma, theta=theta)

    print(f"    Converged in {iterations} evaluate/improve cycles.")
    print()
    print(f"""    Each cycle includes a full policy evaluation (many inner sweeps
    until values stabilize under the current policy), so {iterations} cycles
    is more total work than it appears. On larger problems, policy
    iteration often needs fewer cycles than value iteration needs
    sweeps — but on this small grid, both are nearly instant.
    """)

    # Show final values
    print("    Final state values:")
    grid_values_pi = env.grid_values(V_pi)
    print_grid_values(env, grid_values_pi)
    print()

    print("    Optimal policy:")
    print_policy(policy_pi, 5, 5)
    print()

    # -----------------------------------------------------------------------
    # 4. Comparison
    # -----------------------------------------------------------------------

    print("COMPARISON")
    print("-" * 64)

    # Check that both methods agree
    values_match = True
    for state in env.states():
        if not env.is_terminal(state):
            diff = abs(V_vi.get(state.label) - V_pi.get(state.label))
            if diff > 0.01:
                values_match = False

    policies_match = policy_vi == policy_pi

    print(f"    Values match:   {'yes' if values_match else 'NO'}")
    print(f"    Policies match: {'yes' if policies_match else 'NO'}")
    print()
    print(f"    Both methods find the same optimal policy. Value iteration")
    print(f"    took {len(history)} sweeps. Policy iteration took {iterations} evaluate/improve")
    print(f"    cycles. They arrive at the same answer because there's only")
    print(f"    one optimal value function for a given MDP and discount")
    print(f"    factor — the algorithms just search for it differently.")
    print()
    print(f"    This is the key insight: planning is repeated local backup")
    print(f"    until distant consequences become visible. Both methods do")
    print(f"    this — value iteration by sweeping values, policy iteration")
    print(f"    by alternating evaluation and improvement.")
    print()

    # -----------------------------------------------------------------------
    # 5. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ANIMATION")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots from value iteration history
    # Start with an initial frame showing all zeros (before any sweeps)
    initial_grid = [[0.0] * 5 for _ in range(5)]
    snapshots = [BellmanSnapshot(
        episode=0, total_reward=0.0,
        values=initial_grid, max_change=0.0,
        policy=None, sweep=0,
    )]

    for record in history:
        # Reconstruct the grid values from the stored value dict
        grid = [[0.0] * 5 for _ in range(5)]
        for label, val in record["values"].items():
            parts = label.split(",")
            r, c = int(parts[0]), int(parts[1])
            grid[r][c] = val

        is_final = (record == history[-1])
        snapshots.append(BellmanSnapshot(
            episode=record["sweep"],
            total_reward=0.0,
            values=grid,
            max_change=record["max_change"],
            policy=policy_vi if is_final else None,
            sweep=record["sweep"],
        ))

    # Add extra frames at the end showing the converged state with arrows
    for _ in range(5):
        snapshots.append(BellmanSnapshot(
            episode=history[-1]["sweep"],
            total_reward=0.0,
            values=snapshots[-1].values,
            max_change=0.0,
            policy=policy_vi,
            sweep=history[-1]["sweep"],
        ))

    # Cells where value text should be suppressed (overlay markers go here instead)
    special_cells = set(env.walls + env.pits + env.goals)

    # Determine value range for consistent heatmap coloring
    all_vals = [v for row in snapshots[-1].values for v in row]
    vmin = min(all_vals) - 0.1
    vmax = max(all_vals) + 0.1

    # Create the figure
    fig, axes = create_lesson_figure(
        "Lesson 01: Value Iteration",
        subtitle="Bellman (1957) — reward ripples backward through the grid",
    )

    # Collect max_change values for the trace pane
    max_changes = [s.max_change for s in snapshots]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: value heatmap + grid overlay
        draw_value_heatmap(axes["env"], snap.values, vmin=vmin, vmax=vmax,
                           skip_cells=special_cells)
        draw_grid_overlay(axes["env"], 5, 5,
                          walls=env.walls, pits=env.pits, goals=env.goals)
        if snap.policy:
            draw_policy_arrows(axes["env"], snap.policy, 5, 5)
        sweep_label = "Initial" if snap.sweep == 0 else f"Sweep {snap.sweep}"
        axes["env"].set_title(f"State Values — {sweep_label}", fontsize=10)

        # Top-right: convergence info
        axes["algo"].clear()
        axes["algo"].axis("off")
        if snap.sweep == 0:
            info_lines = [
                "All values start at 0",
                "",
                "The agent knows nothing yet.",
                "Sweeping will propagate",
                "reward information from",
                "the goal backward through",
                "the grid.",
            ]
        else:
            info_lines = [
                f"Sweep: {snap.sweep}",
                f"Max change: {snap.max_change:.6f}",
                f"Converged: {'yes' if snap.max_change < theta else 'no'}",
                "",
                f"Discount (γ): {gamma}",
                f"Threshold (θ): {theta}",
            ]
            if snap.policy:
                info_lines.append("")
                info_lines.append("→ Policy arrows shown")
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title("Algorithm State", fontsize=10)

        # Bottom: convergence trace
        trace_data = max_changes[:frame_idx + 1]
        update_sweep_trace(axes["trace"], trace_data,
                           label="Max value change", color=TEAL)
        axes["trace"].set_ylabel("Max Δ", fontsize=9)
        axes["trace"].set_title("Convergence", fontsize=10)

    # Save animation (slow enough to watch values propagate)
    save_animation(fig, update, len(snapshots), "output/01_bellman_artifact.gif", fps=2)
    print("    Saved: output/01_bellman_artifact.gif")

    # Poster frame — the final converged state with arrows
    fig2, axes2 = create_lesson_figure(
        "Lesson 01: Value Iteration (Converged)",
        subtitle="Bellman (1957)",
    )

    def update_poster(frame_idx):
        snap = snapshots[-1]
        draw_value_heatmap(axes2["env"], snap.values, vmin=vmin, vmax=vmax,
                           skip_cells=special_cells)
        draw_grid_overlay(axes2["env"], 5, 5,
                          walls=env.walls, pits=env.pits, goals=env.goals)
        draw_policy_arrows(axes2["env"], policy_vi, 5, 5)
        axes2["env"].set_title("Optimal Values + Policy", fontsize=10)

        axes2["algo"].clear()
        axes2["algo"].axis("off")
        info = (
            f"Converged in {len(history)} sweeps\n"
            f"γ = {gamma}  θ = {theta}\n\n"
            f"Value iteration and policy\n"
            f"iteration both find the same\n"
            f"optimal policy."
        )
        axes2["algo"].text(0.1, 0.9, info, transform=axes2["algo"].transAxes,
                           fontsize=10, verticalalignment="top",
                           fontfamily="monospace", color=DARK_GRAY)
        axes2["algo"].set_title("Summary", fontsize=10)

        sweep_changes = [h["max_change"] for h in history]
        update_sweep_trace(axes2["trace"], sweep_changes,
                           label="Max value change", color=TEAL)
        axes2["trace"].set_ylabel("Max Δ", fontsize=9)
        axes2["trace"].set_title("Convergence", fontsize=10)

    save_poster(fig2, update_poster, 0, "output/01_bellman_poster.png")
    plt.close(fig2)
    print("    Saved: output/01_bellman_poster.png")

    # Standalone convergence trace
    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    sweep_nums = list(range(1, len(history) + 1))
    ax3.plot(sweep_nums, [h["max_change"] for h in history],
             color=TEAL, linewidth=1.5, marker="o", markersize=5, alpha=0.8,
             label="Max value change")
    ax3.set_xlabel("Sweep")
    ax3.set_ylabel("Max Δ")
    ax3.set_title("Value Iteration Convergence")
    ax3.set_xticks(sweep_nums)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    save_figure(fig3, "output/01_bellman_trace.png")
    print("    Saved: output/01_bellman_trace.png")

    print()
    print("=" * 64)
    print("  Lesson 01 complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
