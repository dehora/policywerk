"""Visualization: Animation skeleton.

Shared infrastructure for lesson animations. Provides the consistent
three-pane layout (environment, algorithm internals, training trace),
frame recording during training, and export to GIF/MP4.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field

# House style — consistent across all seven lessons
TEAL = "#5CB8B2"
ORANGE = "#E8915C"
LIGHT_GRAY = "#E0E0E0"
DARK_GRAY = "#4A4A4A"
DPI = 150


def create_lesson_figure(
    title: str,
    subtitle: str = "",
    figsize: tuple[float, float] = (12, 7),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Create the standard three-pane lesson layout.

    Returns (fig, axes_dict) where axes_dict has keys:
      "env"   — top-left, environment / trajectory view
      "algo"  — top-right, algorithm-specific internal view
      "trace" — bottom, spanning full width, training trace
    """
    fig = plt.figure(figsize=figsize, dpi=DPI)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

    ax_env = fig.add_subplot(gs[0, 0])
    ax_algo = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color=DARK_GRAY)

    axes = {"env": ax_env, "algo": ax_algo, "trace": ax_trace}
    return fig, axes


@dataclass
class FrameSnapshot:
    """Base snapshot captured during training.

    Lessons extend this with algorithm-specific fields.
    The recorder stores these and the animation update function
    reads them back frame by frame.
    """
    episode: int
    total_reward: float


class FrameRecorder:
    """Accumulates training snapshots at configurable intervals.

    Pull model: the training loop checks should_record() and calls
    record() — the recorder doesn't inject itself into the loop.
    """

    def __init__(self, record_interval: int = 1):
        self.record_interval = record_interval
        self.snapshots: list[FrameSnapshot] = []

    def should_record(self, episode: int) -> bool:
        """True if this episode should be recorded."""
        return episode % self.record_interval == 0

    def record(self, snapshot: FrameSnapshot) -> None:
        """Store a snapshot."""
        self.snapshots.append(snapshot)

    @property
    def frame_count(self) -> int:
        return len(self.snapshots)


def save_animation(
    fig: plt.Figure,
    update_fn,
    frame_count: int,
    path: str,
    fps: int = 10,
) -> None:
    """Export an animation to GIF or MP4.

    Args:
        fig: the matplotlib Figure to animate.
        update_fn: callable(frame_index: int) that updates artists.
        frame_count: total number of frames.
        path: output file path (.gif or .mp4).
        fps: frames per second.
    """
    anim = FuncAnimation(fig, update_fn, frames=frame_count, blit=False)

    if path.endswith(".mp4"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
    else:
        writer = PillowWriter(fps=fps)

    anim.save(path, writer=writer, dpi=DPI)
    plt.close(fig)


def save_poster(
    fig: plt.Figure,
    update_fn,
    frame_index: int,
    path: str,
) -> None:
    """Save a single frame as a static image (the poster frame)."""
    update_fn(frame_index)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")


def save_figure(fig: plt.Figure, path: str) -> None:
    """Save a figure and close it."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
