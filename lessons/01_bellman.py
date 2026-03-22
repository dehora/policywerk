"""Lesson 1: The Bellman Equation (Bellman, 1957).

In 1957, Richard Bellman formalized how to make optimal decisions
in sequential problems. This lesson implements his two dynamic
programming algorithms — value iteration and policy iteration —
on a small gridworld, and animates the process of value propagation.

Run: uv run python lessons/01_bellman.py
"""

import os
from dataclasses import dataclass

from policywerk.world.gridworld import GridWorld, WALL, GOAL, PIT
from policywerk.actors.bellman import value_iteration, policy_iteration, extract_policy
from policywerk.primitives.progress import Spinner
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.values import draw_value_heatmap, draw_policy_arrows, draw_grid_overlay

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot — extends FrameSnapshot with per-sweep data
# ---------------------------------------------------------------------------

@dataclass
class BellmanSnapshot(FrameSnapshot):
    values: list[list[float]]       # 5x5 grid of current V values
    max_change: float               # convergence metric
    policy: dict[str, int] | None   # greedy policy (final frames only)
    sweep: int


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

# Labels for special cells in the terminal grid
_CELL_LABELS = {GOAL: "   G  ", WALL: "  ##  ", PIT: "   X  "}


def print_grid_values(env: GridWorld, values: list[list[float]]) -> None:
    """Print the value function as a formatted grid with labeled special cells."""
    for r in range(len(values)):
        cells = []
        for c in range(len(values[0])):
            cell_type = env._grid[r][c]
            if cell_type in _CELL_LABELS:
                cells.append(_CELL_LABELS[cell_type])
            else:
                cells.append(f"{values[r][c]:+.3f}")
        print("    " + "  ".join(cells))


def print_policy(policy: dict[str, int], env: GridWorld, rows: int, cols: int) -> None:
    """Print the policy as a grid of directional arrows with labeled special cells."""
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    for r in range(rows):
        cells = []
        for c in range(cols):
            cell_type = env._grid[r][c]
            if cell_type == GOAL:
                cells.append("  G   ")
            elif cell_type == PIT:
                cells.append("  X   ")
            elif cell_type == WALL:
                cells.append("  #   ")
            else:
                label = f"{r},{c}"
                if label in policy:
                    cells.append(f"  {arrows[policy[label]]}   ")
                else:
                    cells.append("  .   ")
        print("    " + " ".join(cells))


def update_sweep_trace(ax, values, label="", color=TEAL, x_values=None):
    """Update the convergence trace pane with integer sweep ticks."""
    ax.clear()
    if x_values is not None:
        xs = x_values
    else:
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
    print("  Richard Bellman, 'A Markovian Decision Process'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. The environment
    # -----------------------------------------------------------------------

    print("THE GRIDWORLD")
    print("-" * 64)
    print("""
    The agent lives in a 5x5 grid. It can move North, East, South,
    or West. Hitting a wall or boundary leaves it in place.

    Layout:
      . . . . G     G = goal (+1 reward, episode ends)
      . # . X .     # = wall (impassable)
      . . . . .     X = pit (-1 reward, episode ends)
      . . . . .     . = empty (-0.04 per step)
      S . . . .     S = start

    The step cost of -0.04 encourages the agent to reach the goal
    quickly rather than wandering. The question is: from any cell,
    what is the best direction to go?
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
    Value iteration answers "how good is each state?" by assigning
    a number to every cell. This number is called V (the value) and
    it represents how much total future reward the agent can expect
    from that cell, assuming it plays optimally from there.

    At the start, all values are zero — the agent knows nothing.
    Then it sweeps through every cell, updating each one:

      "Try all four actions. For each action, look at the immediate
       reward and the value of wherever that action leads. Multiply
       the future value by gamma (0.9) because one step of distance
       makes it worth a bit less. Keep the best score."

    In math: V(s) = max over actions of [reward + gamma * V(next state)]

    Here is a concrete example. Consider cell (0,3), directly west
    of the goal:

      Move East  -> reach goal, get +1.  Value: +1.0 + 0.9 * 0 = +1.0
      Move North -> hit boundary, stay.  Value: -0.04 + 0.9 * 0 = -0.04
      Move South -> cell (1,3) is pit.   Value: -1.0 + 0.9 * 0 = -1.0
      Move West  -> cell (0,2), empty.   Value: -0.04 + 0.9 * 0 = -0.04

      Best action: East. New V(0,3) = +1.0

    On the first sweep, only cells next to the goal or pit get
    meaningful values. On the second sweep, their neighbors update.
    Gradually, information about rewards "ripples" backward through
    the grid until every cell knows how good it is.

    Note: in this grid, each action has exactly one outcome — move
    East and you always land one cell to the right. The formula
    above uses "expected value" language because later lessons will
    have environments where outcomes are uncertain.
    """)

    V_vi, history = value_iteration(env, gamma=gamma, theta=theta)

    print(f"    Converged in {len(history)} sweeps.")
    print()

    # Show convergence progress with proportional bars
    print("    Sweep-by-sweep convergence (max value change):")
    max_initial = history[0]["max_change"] if history else 1.0
    for record in history:
        fraction = record["max_change"] / max_initial if max_initial > 0 else 0
        bar_len = int(fraction * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)
        print(f"      Sweep {record['sweep']:2d}:  {record['max_change']:.6f}  [{bar}]")
    print()

    print(f"""    With synchronous updates, information travels exactly one cell
    per sweep — each state reads the previous sweep's values:

      Sweep 1: cells next to the goal or pit update — they can see
               the reward directly.
      Sweep 2: their neighbors update, using sweep 1's values.
      Sweep 3-7: the wave continues one step per sweep, reaching
               the far corners of the grid.
      Sweep 8: the last cell updates, values nearly stable.
      Sweep 9: nothing changes — convergence.

    The start cell (bottom-left) is 8 steps from the goal, so
    it takes 8 sweeps for goal information to reach it.
    """)

    # Show final values
    print("    Final state values (G=goal, X=pit, #=wall):")
    grid_values = env.grid_values(V_vi)
    print_grid_values(env, grid_values)
    print()

    print("""    Reading the grid: the highest values are near the goal
    (top-right). The lowest are at the start (bottom-left), furthest
    away. Values decrease smoothly with distance — each step away
    from the goal costs roughly a factor of gamma (0.9).

    The start cell has value +0.270. That is the goal's +1 reward,
    discounted over the 8 steps it takes to get there, minus the
    step costs paid along the way. Each step of distance reduces
    the reward by a factor of gamma (0.9) and adds a -0.04 cost.
    The green-to-red gradient in the animation is literally the
    discount factor at work.
    """)

    # Derive and show the greedy policy
    policy_vi = extract_policy(env, V_vi, gamma=gamma)
    print("    Optimal policy (G=goal, X=pit, #=wall):")
    print_policy(policy_vi, env, 5, 5)
    print()

    print("""    Starting at S (bottom-left), follow the arrows: up, up, up,
    up to the top row, then right, right, right, right to the goal.
    Eight steps — the shortest path. The wall and pit are naturally
    avoided because their neighbors have lower values.

    The arrows point toward higher-valued states. The agent does not
    need to "see" the goal — it just follows the gradient uphill.
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

    For example, suppose the initial policy says "go North from every
    cell." After evaluation, the top-row cells get terrible values —
    they keep hitting the boundary and paying -0.04 each step with no
    way to reach the goal. Improvement switches those cells to East.
    The next evaluation confirms that East is better. Each cycle fixes
    the worst decisions in the current policy.
    """)

    V_pi, policy_pi, iterations = policy_iteration(env, gamma=gamma, theta=theta)

    print(f"    Converged in {iterations} evaluate/improve cycles.")
    print()
    print(f"""    Each cycle includes a full policy evaluation (many inner sweeps
    until values stabilize under the current policy), so {iterations} cycles
    is more total work than it appears. On larger problems, policy
    iteration often needs fewer cycles than value iteration needs
    sweeps, but on this small grid both are nearly instant.
    """)

    # Show final values
    print("    Final state values (G=goal, X=pit, #=wall):")
    grid_values_pi = env.grid_values(V_pi)
    print_grid_values(env, grid_values_pi)
    print()

    print("    Optimal policy (G=goal, X=pit, #=wall):")
    print_policy(policy_pi, env, 5, 5)
    print()

    # -----------------------------------------------------------------------
    # 4. Comparison
    # -----------------------------------------------------------------------

    print("COMPARISON")
    print("-" * 64)

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
    print(f"    took {len(history)} sweeps through the entire state space. Policy")
    print(f"    iteration took {iterations} evaluate/improve cycles, each containing")
    print(f"    many inner sweeps.")
    print()
    print(f"    They arrive at the same answer because there is only one")
    print(f"    optimal value function for a given MDP and discount factor.")
    print(f"    The algorithms search for it differently:")
    print()
    print(f"      Value iteration:  sweep all states, always pick the best")
    print(f"                        action. Fast per sweep, many sweeps.")
    print()
    print(f"      Policy iteration: fix a policy, fully evaluate it, then")
    print(f"                        improve. Fewer outer cycles, but each")
    print(f"                        cycle does a full evaluation.")
    print()
    print(f"    The key insight is the same in both: planning is repeated")
    print(f"    local backup until distant consequences become visible.")
    print(f"    Each state learns about the goal not by seeing it directly,")
    print(f"    but by looking one step ahead at a neighbor that already")
    print(f"    knows. Information propagates through the grid like a wave.")
    print()

    # -----------------------------------------------------------------------
    # 5. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots from value iteration history
    initial_grid = [[0.0] * 5 for _ in range(5)]
    snapshots = [BellmanSnapshot(
        episode=0, total_reward=0.0,
        values=initial_grid, max_change=0.0,
        policy=None, sweep=0,
    )]

    for record in history:
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

    # Hold on the final converged state with arrows
    for _ in range(5):
        snapshots.append(BellmanSnapshot(
            episode=history[-1]["sweep"],
            total_reward=0.0,
            values=snapshots[-1].values,
            max_change=0.0,
            policy=policy_vi,
            sweep=history[-1]["sweep"],
        ))

    # Cells where value text is suppressed (overlay markers instead)
    special_cells = set(env.walls + env.pits + env.goals)

    # Consistent heatmap coloring across all frames
    all_vals = [v for row in snapshots[-1].values for v in row]
    vmin = min(all_vals) - 0.1
    vmax = max(all_vals) + 0.1

    # --- Artifact 1: Animation (GIF + PDF storyboard) ---

    fig, axes = create_lesson_figure(
        "Lesson 01: Value Iteration",
        subtitle="Bellman (1957) — reward ripples backward through the grid",
    )
    # Build real convergence data from history only (no fake initial/hold frames)
    real_changes = [h["max_change"] for h in history]
    real_sweeps = [h["sweep"] for h in history]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        draw_value_heatmap(axes["env"], snap.values, vmin=vmin, vmax=vmax,
                           skip_cells=special_cells)
        draw_grid_overlay(axes["env"], 5, 5,
                          walls=env.walls, pits=env.pits, goals=env.goals)
        if snap.policy:
            draw_policy_arrows(axes["env"], snap.policy, 5, 5)
        sweep_label = "Initial" if snap.sweep == 0 else f"Sweep {snap.sweep}"
        axes["env"].set_title(f"State Values — {sweep_label}", fontsize=10)

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
                f"Discount (gamma): {gamma}",
                f"Threshold (theta): {theta}",
            ]
            if snap.policy:
                info_lines.append("")
                info_lines.append(">> Policy arrows shown")
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title("Algorithm State", fontsize=10)

        # Clamp trace to real sweep data only (no fake initial/hold points)
        n = min(frame_idx, len(real_changes))
        trace_data = real_changes[:n]
        trace_xs = real_sweeps[:n]
        update_sweep_trace(axes["trace"], trace_data,
                           label="Max value change", color=TEAL,
                           x_values=trace_xs)
        axes["trace"].set_ylabel("Max change", fontsize=9)
        axes["trace"].set_title("Convergence", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/01_bellman_artifact.gif", fps=2)

    # --- Artifact 2: Poster frame ---

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
            f"gamma = {gamma}  theta = {theta}\n\n"
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
                           label="Max value change", color=TEAL,
                           x_values=list(range(1, len(history) + 1)))
        axes2["trace"].set_ylabel("Max change", fontsize=9)
        axes2["trace"].set_title("Convergence", fontsize=10)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/01_bellman_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Convergence trace ---

    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    sweep_nums = list(range(1, len(history) + 1))
    ax3.plot(sweep_nums, [h["max_change"] for h in history],
             color=TEAL, linewidth=1.5, marker="o", markersize=5, alpha=0.8,
             label="Max value change")
    ax3.set_xlabel("Sweep")
    ax3.set_ylabel("Max change")
    ax3.set_title("Value Iteration Convergence")
    ax3.set_xticks(sweep_nums)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    with Spinner("Generating trace"):
        save_figure(fig3, "output/01_bellman_trace.png")

    print()
    print("    Artifacts saved to output/:")
    print("      artifact.gif  animated value propagation, one frame per sweep")
    print("      artifact.pdf  PDF storyboard of every frame")
    print("      poster.png    final converged state with policy arrows")
    print("      trace.png     convergence curve (max value change per sweep)")

    # -----------------------------------------------------------------------
    # Closing
    # -----------------------------------------------------------------------

    print()
    print("=" * 64)
    print("  Lesson 01 complete.")
    print()
    print("  Both algorithms found the optimal policy by reasoning about")
    print("  the environment's rules. But what if the agent does not know")
    print("  the rules? In Lesson 02, the Barto/Sutton actor-critic learns")
    print("  to balance a pole through trial and error alone.")
    print("=" * 64)


if __name__ == "__main__":
    main()
