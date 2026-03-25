"""Visualization: Training trace plots.

Reward curves, loss curves, episode length—the bottom pane
of every lesson animation, and standalone trace.png exports.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend—renders to files
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
        metrics: dict of metric_name to list of values over time.
            e.g. {"reward": [0.1, 0.5, 0.8, ...], "loss": [5.0, 3.2, ...]}
        title: figure title.

    Returns a Figure—caller is responsible for saving/closing it.
    """
    colors = [TEAL, ORANGE, DARK_GRAY, "#7B68EE"]
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)

    for i, (name, values) in enumerate(metrics.items()):
        color = colors[i % len(colors)]
        # alpha=0.8 makes lines slightly translucent so overlapping data stays visible
        ax.plot(values, color=color, linewidth=1.5, label=name, alpha=0.8)

    ax.set_xlabel("Episode")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()  # adjust spacing so labels don't overlap
    return fig


def update_trace_axes(
    ax: plt.Axes,
    values: Vector,
    label: str = "",
    color: str = TEAL,
) -> None:
    """Update the trace pane within an animation frame.

    Unlike plot_training_traces (which creates a standalone figure),
    this updates an existing axes in-place—used for the bottom
    pane of each animation frame.

    Wipes the plot and draws everything from scratch each frame.
    Inefficient but simple—the alternative (updating line data
    in place) adds complexity we don't need at these frame rates.
    """
    ax.clear()
    ax.plot(values, color=color, linewidth=1.5, label=label, alpha=0.8)
    ax.set_xlabel("Episode", fontsize=9)
    if label:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
