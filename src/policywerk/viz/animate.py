"""Visualization: Animation skeleton.

Shared infrastructure for lesson animations. Provides the consistent
three-pane layout (environment, algorithm internals, training trace),
frame recording during training, and export to GIF/MP4.

How animation works:
  1. During training, the lesson records snapshots at intervals
     (e.g. every 10 episodes) using FrameRecorder.
  2. After training, the lesson creates an update function that
     reads each snapshot and redraws the figure for that frame.
  3. save_animation() calls the update function once per frame,
     captures each redraw, and writes them as a GIF.

Each frame redraws from scratch (clear + redraw) rather than
updating individual elements—simpler and sufficient for our
frame rates.
"""

import matplotlib
# Non-interactive backend—renders to files, not to a window.
# Required when running on servers or generating files without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field

# House style—consistent across all seven lessons.
# TEAL: primary data (agent, current estimates, learned values)
# ORANGE: secondary/reference data (true values, targets, comparisons)
# LIGHT_GRAY: de-emphasized or background elements
# DARK_GRAY: labels, borders, text
TEAL = "#5CB8B2"
ORANGE = "#E8915C"
LIGHT_GRAY = "#E0E0E0"
DARK_GRAY = "#4A4A4A"
# Dots per inch—controls image resolution.
# 150 is a good balance between quality and file size.
DPI = 150


def create_lesson_figure(
    title: str,
    subtitle: str = "",
    figsize: tuple[float, float] = (12, 7),
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Create the standard three-pane lesson layout.

    Returns (fig, axes_dict) where axes_dict has keys:
      "env"  —top-left, environment / trajectory view
      "algo" —top-right, algorithm-specific internal view
      "trace"—bottom, spanning full width, training trace

    figsize is in inches (matplotlib convention).
    """
    fig = plt.figure(figsize=figsize, dpi=DPI)
    # GridSpec divides the figure into a 2×2 grid.
    # height_ratios=[3, 1] makes the top row 3× taller than the bottom.
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1],
                  hspace=0.3, wspace=0.3)  # spacing between panes

    ax_env = fig.add_subplot(gs[0, 0])
    ax_algo = fig.add_subplot(gs[0, 1])
    ax_trace = fig.add_subplot(gs[1, :])  # bottom spans both columns

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    if subtitle:
        fig.text(0.5, 0.94, subtitle, ha="center", fontsize=10, color=DARK_GRAY)

    axes = {"env": ax_env, "algo": ax_algo, "trace": ax_trace}
    return fig, axes


@dataclass
class FrameSnapshot:
    """A frozen picture of the training state at one moment.

    Lessons extend this with algorithm-specific fields (e.g. value
    tables, trajectories, Q-values). The recorder stores a list of
    these, and the animation update function reads them back one
    per frame.
    """
    episode: int
    total_reward: float


class FrameRecorder:
    """Accumulates training snapshots at configurable intervals.

    The training loop decides when to record—the recorder doesn't
    inject itself into the loop.

    Usage:
        recorder = FrameRecorder(record_interval=10)
        for episode in range(1000):
            ... train ...
            if recorder.should_record(episode):
                recorder.record(MySnapshot(episode=episode, ...))

        # Later, build animation from recorder.snapshots
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
    pdf: bool = True,
) -> None:
    """Export an animation to GIF or MP4, optionally with a PDF storyboard.

    Args:
        fig: the matplotlib Figure to animate.
        update_fn: callable(frame_index: int) that redraws the
            figure for one frame. Called once per frame for the GIF,
            and again for each frame of the PDF storyboard. Must be
            idempotent (pure redraw, no side effects).
        frame_count: total number of frames.
        path: output file path (.gif or .mp4).
        fps: frames per second.
        pdf: if True, also export a multi-page PDF (one page per frame)
            alongside the animation. The PDF path is derived from the
            animation path by replacing the extension. Note: when pdf
            is True, update_fn is called again for each frame of the
            PDF storyboard, so it must be idempotent (pure redraw, no
            side effects).
    """
    # FuncAnimation calls update_fn(0), update_fn(1), ..., update_fn(frame_count-1),
    # and the writer captures each resulting figure as one frame.
    # blit=False means we redraw the entire figure each frame (simpler).
    anim = FuncAnimation(fig, update_fn, frames=frame_count, blit=False)

    if path.endswith(".mp4"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=fps)
    else:
        # PillowWriter uses the Pillow library (a matplotlib dependency)
        # to write GIF files—no external tools needed.
        writer = PillowWriter(fps=fps)

    anim.save(path, writer=writer, dpi=DPI)

    # Export a PDF storyboard—one page per frame, useful for review
    # and for embedding individual frames in documentation.
    if pdf:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_path = path.rsplit(".", 1)[0] + ".pdf"
        with PdfPages(pdf_path) as pp:
            for frame_idx in range(frame_count):
                update_fn(frame_idx)
                pp.savefig(fig, dpi=DPI, bbox_inches="tight")

    plt.close(fig)


def save_poster(
    fig: plt.Figure,
    update_fn,
    frame_index: int,
    path: str,
) -> None:
    """Save a single representative frame as a static image.

    Like a movie poster—a thumbnail that represents the animation.
    """
    update_fn(frame_index)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")  # tight crops whitespace


def save_figure(fig: plt.Figure, path: str) -> None:
    """Save a figure and close it."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
