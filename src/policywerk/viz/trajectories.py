"""Visualization: Trajectories, agents, pixel rendering.

Agent paths through environments, marker drawing, pixel-grid
rendering for DQN/Dreamer, policy distribution curves, and
real-vs-imagined split-screen for world models.
"""

import math

import matplotlib
matplotlib.use("Agg")
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
    """Draw a path through a sequence of (x, y) positions."""
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


def draw_pixel_env(
    ax: plt.Axes,
    frame: Matrix,
) -> None:
    """Render a pixel-grid environment (e.g. 16x16) using imshow.

    frame: rows x cols matrix of floats (0.0 = empty, higher = objects).
    """
    ax.clear()
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
    """Draw a Gaussian policy distribution curve.

    Shows the probability density of continuous actions,
    useful for L06 PPO to visualize policy smoothing.
    """
    ax.clear()
    lo, hi = action_range
    step = (hi - lo) / num_points
    xs = [lo + i * step for i in range(num_points + 1)]

    # Gaussian PDF: (1 / (σ√2π)) * exp(-0.5 * ((x-μ)/σ)²)
    inv_norm = 1.0 / (std * math.sqrt(2.0 * math.pi))
    ys = []
    for x in xs:
        z = (x - mean) / std
        ys.append(inv_norm * math.exp(-0.5 * z * z))

    ax.fill_between(xs, ys, alpha=0.3, color=TEAL)
    ax.plot(xs, ys, color=TEAL, linewidth=1.5)
    ax.axvline(mean, color=ORANGE, linestyle="--", linewidth=1, label=f"μ={mean:.2f}")
    ax.set_xlabel("Action", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_real_vs_imagined(
    ax: plt.Axes,
    real_frame: Matrix,
    imagined_frame: Matrix,
) -> None:
    """Split-screen comparison: real observation vs world-model prediction.

    Draws both frames side by side within a single axes.
    Used by L07 Dreamer to show imagination tracking/drifting.
    """
    ax.clear()

    rows_r = len(real_frame)
    cols_r = len(real_frame[0]) if real_frame else 0
    rows_i = len(imagined_frame)
    cols_i = len(imagined_frame[0]) if imagined_frame else 0

    # Combine side by side with a 1-pixel gap
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

    # Labels
    ax.text(cols_r / 2, -1.5, "Real", ha="center", fontsize=8, color=TEAL)
    ax.text(cols_r + gap + cols_i / 2, -1.5, "Imagined", ha="center",
            fontsize=8, color=ORANGE)
