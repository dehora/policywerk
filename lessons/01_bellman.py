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
from policywerk.viz.traces import update_trace_axes, plot_training_traces

import matplotlib
matplotlib.use("Agg")


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

ACTION_NAMES = {0: "N", 1: "E", 2: "S", 3: "W"}


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

    # Show convergence progress
    print("    Sweep-by-sweep convergence (max value change):")
    for record in history:
        bar_len = min(50, int(record["max_change"] * 200))
        bar = "█" * bar_len
        print(f"      Sweep {record['sweep']:3d}:  {record['max_change']:.6f}  {bar}")
    print()

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
    the arrows all eventually point toward the goal and away from
    the pit.
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

    Repeat until the policy stops changing. This typically converges
    in very few iterations — often faster than value iteration.
    """)

    V_pi, policy_pi, iterations = policy_iteration(env, gamma=gamma, theta=theta)

    print(f"    Converged in {iterations} policy improvement iterations.")
    print()

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
    print(f"""
    Both methods find the same optimal policy. Value iteration took
    {len(history)} sweeps. Policy iteration took {iterations} evaluate/improve
    cycles. They arrive at the same answer because there's only one
    optimal value function for a given MDP and discount factor — the
    algorithms just search for it differently.
    """)

    # -----------------------------------------------------------------------
    # 5. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ANIMATION")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots from value iteration history
    snapshots = []
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

    # Add a few extra frames at the end showing the final state with arrows
    for _ in range(5):
        snapshots.append(BellmanSnapshot(
            episode=history[-1]["sweep"],
            total_reward=0.0,
            values=snapshots[-1].values,
            max_change=0.0,
            policy=policy_vi,
            sweep=history[-1]["sweep"],
        ))

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
        draw_value_heatmap(axes["env"], snap.values, vmin=vmin, vmax=vmax)
        draw_grid_overlay(axes["env"], 5, 5,
                          walls=env.walls, pits=env.pits, goals=env.goals)
        if snap.policy:
            draw_policy_arrows(axes["env"], snap.policy, 5, 5)
        axes["env"].set_title(f"State Values — Sweep {snap.sweep}", fontsize=10)

        # Top-right: convergence info
        axes["algo"].clear()
        axes["algo"].axis("off")
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
            info_lines.append("Policy arrows shown")
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title("Algorithm State", fontsize=10)

        # Bottom: convergence trace
        trace_data = max_changes[:frame_idx + 1]
        update_trace_axes(axes["trace"], trace_data, label="Max value change", color=TEAL)
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
        draw_value_heatmap(axes2["env"], snap.values, vmin=vmin, vmax=vmax)
        draw_grid_overlay(axes2["env"], 5, 5,
                          walls=env.walls, pits=env.pits, goals=env.goals)
        draw_policy_arrows(axes2["env"], policy_vi, 5, 5)
        axes2["env"].set_title("Optimal Values + Policy", fontsize=10)

        axes2["algo"].clear()
        axes2["algo"].axis("off")
        info = f"Converged in {len(history)} sweeps\nγ = {gamma}  θ = {theta}"
        axes2["algo"].text(0.1, 0.9, info, transform=axes2["algo"].transAxes,
                           fontsize=10, verticalalignment="top",
                           fontfamily="monospace", color=DARK_GRAY)
        axes2["algo"].set_title("Summary", fontsize=10)

        update_trace_axes(axes2["trace"], max_changes[:len(history)],
                          label="Max value change", color=TEAL)
        axes2["trace"].set_ylabel("Max Δ", fontsize=9)
        axes2["trace"].set_title("Convergence", fontsize=10)

    save_poster(fig2, update_poster, 0, "output/01_bellman_poster.png")
    import matplotlib.pyplot as plt
    plt.close(fig2)
    print("    Saved: output/01_bellman_poster.png")

    # Standalone convergence trace
    trace_fig = plot_training_traces(
        {"Max value change": [h["max_change"] for h in history]},
        title="Value Iteration Convergence",
    )
    save_figure(trace_fig, "output/01_bellman_trace.png")
    print("    Saved: output/01_bellman_trace.png")

    print()
    print("=" * 64)
    print("  Lesson 01 complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
