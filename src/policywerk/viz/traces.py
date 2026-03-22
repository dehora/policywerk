"""Visualization: Training trace plots.

Reward curves, loss curves, episode length — the bottom pane
of every lesson animation, and standalone trace.png exports.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from policywerk.viz.animate import TEAL, ORANGE, DARK_GRAY, DPI, save_figure

Vector = list[float]


def plot_training_traces(
    metrics: dict[str, Vector],
    title: str = "Training Progress",
    figsize: tuple[float, float] = (10, 3),
) -> plt.Figure:
    """Create a standalone training trace figure.

    Args:
        metrics: dict of metric_name -> list of values over time.
        title: figure title.
    """
    colors = [TEAL, ORANGE, DARK_GRAY, "#7B68EE"]
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    for i, (name, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        ax.plot(values, color=color, linewidth=1.5, label=name, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def update_trace_axes(
    ax: plt.Axes,
    values: Vector,
    label: str = "",
    color: str = TEAL,
) -> None:
    """Update a trace axes in-place for animation frames.

    Clears and redraws — simpler than artist management
    for line plots with growing data.
    """
    ax.clear()
    ax.plot(values, color=color, linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=9)
    if label:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
