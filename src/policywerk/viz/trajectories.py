"""Visualization: Trajectories, agents, pixel rendering.

Agent paths through environments, marker drawing, pixel-grid
rendering for DQN/Dreamer, policy distribution curves, and
real-vs-imagined split-screen for world models.
"""

import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — renders to files
import matplotlib.pyplot as plt

from policywerk.viz.animate import TEAL, ORANGE, LIGHT_GRAY, DARK_GRAY

Vector = list[float]
Matrix = list[list[float]]


def draw_trajectory(
    ax: plt.Axes,
    positions: list[tuple[float, float]],
    color: str = TEAL,
    alpha: float = 0.6,
    linewidth: float = 1.5,
) -> None:
    """Draw the path an agent took through a sequence of (x, y) positions."""
    if len(positions) < 2:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)


def draw_agent(
    ax: plt.Axes,
    position: tuple[float, float],
    color: str = TEAL,
    size: float = 80,
) -> None:
    """Draw an agent marker at the given position."""
    # zorder=10 ensures the agent is drawn on top of grid lines and trajectories
    ax.scatter([position[0]], [position[1]], c=color, s=size,
               zorder=10, edgecolors=DARK_GRAY, linewidths=1)


def draw_target(
    ax: plt.Axes,
    position: tuple[float, float],
    color: str = ORANGE,
    size: float = 100,
) -> None:
    """Draw a target marker (star) at the given position."""
    ax.scatter([position[0]], [position[1]], c=color, s=size,
               marker="*", zorder=10, edgecolors=DARK_GRAY, linewidths=0.5)


def draw_chain(
    ax: plt.Axes,
    labels: list[str],
    values: list[float] | None = None,
    path: list[str] | None = None,
    outcome: str | None = None,
) -> None:
    """Draw a chain of states (e.g. the random walk A-B-C-D-E).

    labels: state names in order (left to right).
    values: optional value estimates for each state (shown as color fill).
    path: optional list of state labels visited this episode (drawn as line).
    outcome: "left" or "right" — which terminal was reached.
    """
    n = len(labels)
    node_y = 0.5
    spacing = 1.0

    ax.clear()
    ax.set_xlim(-1.5, n * spacing + 0.5)
    ax.set_ylim(-0.3, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw terminal zones
    ax.text(-1.0, node_y, "[0]", ha="center", va="center", fontsize=11,
            fontweight="bold", color="red" if outcome == "left" else DARK_GRAY)
    ax.text(n * spacing, node_y, "[+1]", ha="center", va="center", fontsize=11,
            fontweight="bold", color="green" if outcome == "right" else DARK_GRAY)

    # Draw connecting lines between nodes
    for i in range(n - 1):
        ax.plot([i * spacing + 0.2, (i + 1) * spacing - 0.2], [node_y, node_y],
                color=LIGHT_GRAY, linewidth=2, zorder=1)
    # Lines to terminals
    ax.plot([-0.6, -0.2], [node_y, node_y], color=LIGHT_GRAY, linewidth=2, zorder=1)
    ax.plot([(n - 1) * spacing + 0.2, n * spacing - 0.4], [node_y, node_y],
            color=LIGHT_GRAY, linewidth=2, zorder=1)

    # Draw the path if provided (line through visited nodes)
    if path and len(path) > 1:
        path_xs = []
        path_ys = []
        for label in path:
            if label in labels:
                idx = labels.index(label)
                path_xs.append(idx * spacing)
                path_ys.append(node_y)
        # Add terminal position
        if outcome == "left":
            path_xs.append(-1.0)
            path_ys.append(node_y)
        elif outcome == "right":
            path_xs.append(n * spacing)
            path_ys.append(node_y)
        # Offset path slightly below nodes so it's visible
        path_ys_offset = [y - 0.12 for y in path_ys]
        ax.plot(path_xs, path_ys_offset, color=TEAL, linewidth=1.5, alpha=0.5, zorder=2)

    # Draw nodes
    for i, label in enumerate(labels):
        x = i * spacing
        # Color node by value if provided
        if values is not None:
            val = values[i]
            # Map value 0-1 to color intensity (white to teal)
            r = int(255 - val * (255 - 92))   # 92 = 0x5C (TEAL red)
            g = int(255 - val * (255 - 184))   # 184 = 0xB8 (TEAL green)
            b = int(255 - val * (255 - 178))   # 178 = 0xB2 (TEAL blue)
            color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            color = "white"

        circle = plt.Circle((x, node_y), 0.18, facecolor=color,
                             edgecolor=DARK_GRAY, linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, node_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK_GRAY, zorder=6)

        # Show value below node
        if values is not None:
            ax.text(x, node_y - 0.28, f"{values[i]:.2f}", ha="center",
                    va="top", fontsize=7, color=DARK_GRAY)


def draw_pole(
    ax: plt.Axes,
    angle: float,
    action: int | None = None,
    pole_length: float = 0.8,
) -> None:
    """Draw a simple inverted pendulum (pole on a pivot).

    The pole is a line from a fixed pivot point, tilted by the
    current angle. The pivot is drawn as a small triangle base.
    Optionally shows the applied force direction.
    """
    # Pivot at bottom center
    pivot_x, pivot_y = 0.0, 0.0

    # Pole tip: angle=0 is straight up, positive = right tilt
    tip_x = pivot_x + pole_length * math.sin(angle)
    tip_y = pivot_y + pole_length * math.cos(angle)

    # Draw the pole
    ax.plot([pivot_x, tip_x], [pivot_y, tip_y],
            color=TEAL, linewidth=4, solid_capstyle="round", zorder=5)

    # Draw the pivot base (small triangle)
    base_w = 0.15
    ax.fill([pivot_x - base_w, pivot_x + base_w, pivot_x],
            [pivot_y - 0.05, pivot_y - 0.05, pivot_y + 0.02],
            color=DARK_GRAY, zorder=6)

    # Draw the tip as a circle
    ax.scatter([tip_x], [tip_y], c=TEAL, s=60, zorder=7,
               edgecolors=DARK_GRAY, linewidths=0.5)

    # Show force direction arrow if action is given
    if action is not None:
        arrow_y = -0.08
        arrow_dx = 0.2 if action == 1 else -0.2
        ax.annotate("", xy=(pivot_x + arrow_dx, arrow_y),
                     xytext=(pivot_x, arrow_y),
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))

    # Ground line
    ax.plot([-0.5, 0.5], [-0.05, -0.05], color=LIGHT_GRAY, linewidth=1, zorder=1)

    # Danger zones (where the pole would fall)
    danger_angle = 0.3
    for sign in [1, -1]:
        dx = pole_length * math.sin(sign * danger_angle)
        dy = pole_length * math.cos(sign * danger_angle)
        ax.plot([pivot_x, pivot_x + dx], [pivot_y, pivot_y + dy],
                color="red", linewidth=1, linestyle="--", alpha=0.3, zorder=2)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.15, pole_length + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_pixel_env(
    ax: plt.Axes,
    frame: Matrix,
) -> None:
    """Display a pixel-grid environment (e.g. 16×16) as an image.

    frame: rows × cols matrix of floats (0.0 = empty, higher = objects).
    Uses reversed grayscale: 0.0 = white (empty), 1.0 = black (agent).
    """
    ax.clear()
    # interpolation="nearest" keeps pixels as sharp squares (no blurring)
    ax.imshow(frame, cmap="gray_r", interpolation="nearest",
              vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_policy_gaussian(
    ax: plt.Axes,
    mean: float,
    std: float,
    action_range: tuple[float, float] = (-2.0, 2.0),
    num_points: int = 100,
) -> None:
    """Draw the agent's action distribution as a bell curve.

    Shows how likely each action value is. A tall narrow curve means
    the agent is confident; a short wide curve means uncertain.
    Used by L06 PPO to show policy smoothing over training.
    """
    ax.clear()
    lo, hi = action_range
    step = (hi - lo) / num_points
    xs = [lo + i * step for i in range(num_points + 1)]

    # Bell curve formula: (1 / (std × √(2π))) × exp(-0.5 × ((x - mean) / std)²)
    inv_norm = 1.0 / (std * math.sqrt(2.0 * math.pi))
    ys = []
    for x in xs:
        z = (x - mean) / std
        ys.append(inv_norm * math.exp(-0.5 * z * z))

    # fill_between shades the area under the curve
    ax.fill_between(xs, ys, alpha=0.3, color=TEAL)
    ax.plot(xs, ys, color=TEAL, linewidth=1.5)
    # axvline draws a vertical line marking the mean action
    ax.axvline(mean, color=ORANGE, linestyle="--", linewidth=1, label=f"mean={mean:.2f}")
    ax.set_xlabel("Action", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_real_vs_imagined(
    ax: plt.Axes,
    real_frame: Matrix,
    imagined_frame: Matrix,
) -> None:
    """Split-screen: real observation on the left, world-model prediction on the right.

    Both frames are combined into one image so they stay perfectly
    aligned, making frame-by-frame comparison immediate. A gray gap
    column separates them visually.

    Used by L07 Dreamer to show imagination tracking reality, then diverging.
    """
    ax.clear()

    rows_r = len(real_frame)
    cols_r = len(real_frame[0]) if real_frame else 0
    rows_i = len(imagined_frame)
    cols_i = len(imagined_frame[0]) if imagined_frame else 0

    # Combine side by side with a 1-pixel gray gap as separator
    gap = 1
    combined_cols = cols_r + gap + cols_i
    combined = [[0.5] * combined_cols for _ in range(max(rows_r, rows_i))]

    for r in range(rows_r):
        for c in range(cols_r):
            combined[r][c] = real_frame[r][c]
    for r in range(rows_i):
        for c in range(cols_i):
            combined[r][cols_r + gap + c] = imagined_frame[r][c]

    ax.imshow(combined, cmap="gray_r", interpolation="nearest",
              vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(cols_r / 2, -1.5, "Real", ha="center", fontsize=8, color=TEAL)
    ax.text(cols_r + gap + cols_i / 2, -1.5, "Imagined", ha="center",
            fontsize=8, color=ORANGE)
