"""Visualization: Value functions, policies, and bar charts.

Heatmaps for state values, directional arrows for greedy policies,
grouped bar charts for value comparisons, Q-value displays.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — renders to files
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from policywerk.viz.animate import TEAL, ORANGE, LIGHT_GRAY, DARK_GRAY

Vector = list[float]
Matrix = list[list[float]]

# Arrow directions for grid actions: N, E, S, W
# Values are how far the arrow extends, in grid-cell units
_ARROW_DX = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
_ARROW_DY = {0: 0.3, 1: 0, 2: -0.3, 3: 0}


def draw_value_heatmap(
    ax: plt.Axes,
    values: Matrix,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """Draw colored cells representing state values on a grid.

    Red = low values (bad states), green = high values (good states).
    Each cell shows its numeric value as text.

    values: rows × cols matrix of floats.
    """
    ax.clear()
    rows = len(values)
    cols = len(values[0]) if values else 0

    # imshow displays a matrix as a colored image
    # cmap="RdYlGn" = Red-Yellow-Green gradient
    # origin="upper" = row 0 at the top (matching grid convention)
    # aspect="equal" = cells are square, not stretched
    im = ax.imshow(
        values, cmap="RdYlGn", vmin=vmin, vmax=vmax,
        origin="upper", aspect="equal",
    )

    # Annotate cells with numeric values
    for r in range(rows):
        for c in range(cols):
            val = values[r][c]
            # White text on dark cells, black text on light cells
            color = "white" if abs(val) > (vmax - vmin) * 0.4 else "black"
            # ha/va = horizontal/vertical alignment
            ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold")

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)


def draw_policy_arrows(
    ax: plt.Axes,
    policy: dict[str, int],
    rows: int,
    cols: int,
) -> None:
    """Overlay directional arrows showing which way the agent would go.

    policy: maps "r,c" -> action (0=N, 1=E, 2=S, 3=W).
    """
    for label, action in policy.items():
        parts = label.split(",")
        r, c = int(parts[0]), int(parts[1])
        dx = _ARROW_DX.get(action, 0)
        dy = _ARROW_DY.get(action, 0)
        # imshow's y-axis is inverted (row 0 at top), so flip dy
        ax.annotate("", xy=(c + dx, r - dy), xytext=(c, r),
                     arrowprops=dict(arrowstyle="->", color=DARK_GRAY, lw=1.5))


def draw_grid_overlay(
    ax: plt.Axes,
    rows: int,
    cols: int,
    walls: list[tuple[int, int]] | None = None,
    pits: list[tuple[int, int]] | None = None,
    goals: list[tuple[int, int]] | None = None,
) -> None:
    """Mark special cells on a grid — walls, pits, goals."""
    walls = walls or []
    pits = pits or []
    goals = goals or []

    for r, c in walls:
        # imshow centers each cell at integer coordinates, so the cell
        # at (r,c) spans from (c-0.5) to (c+0.5). Fill exactly one cell.
        rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                  facecolor=DARK_GRAY, alpha=0.8)
        ax.add_patch(rect)
        ax.text(c, r, "W", ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")

    for r, c in pits:
        ax.text(c, r, "X", ha="center", va="center", color="red",
                fontsize=12, fontweight="bold")

    for r, c in goals:
        ax.text(c, r, "G", ha="center", va="center", color="green",
                fontsize=12, fontweight="bold")


def draw_value_bars(
    ax: plt.Axes,
    estimated: Vector,
    true_values: Vector,
    labels: list[str],
) -> None:
    """Grouped bar chart comparing estimated vs true state values.

    Shows how close the agent's predictions are to reality.
    Used by L03 TD learning to visualize prediction convergence.
    """
    ax.clear()
    n = len(labels)
    width = 0.35
    x = list(range(n))
    x_est = [xi - width / 2 for xi in x]
    x_true = [xi + width / 2 for xi in x]

    ax.bar(x_est, estimated, width, label="Estimated", color=TEAL, alpha=0.8)
    ax.bar(x_true, true_values, width, label="True", color=ORANGE, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def draw_q_bars(
    ax: plt.Axes,
    q_values: Vector,
    action_labels: list[str],
) -> None:
    """Bar chart of Q-values (action scores) for the current state.

    The best action is highlighted in teal — this is the one
    the agent would choose.
    """
    ax.clear()
    n = len(action_labels)
    colors = [TEAL if q == max(q_values) else LIGHT_GRAY for q in q_values]

    ax.bar(range(n), q_values, color=colors, alpha=0.8, edgecolor=DARK_GRAY)
    ax.set_xticks(range(n))
    ax.set_xticklabels(action_labels, fontsize=8)
    ax.set_ylabel("Q-value", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
