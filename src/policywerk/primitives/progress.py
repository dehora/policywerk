"""Level 0: Training progress display.

Terminal progress bar for long-running training loops. Updates a
single line in place using carriage return.
"""

import sys

from policywerk.primitives import scalar


def progress_bar(
    epoch: int,
    total: int,
    loss: float,
    width: int = 30,
    stream=sys.stderr,
) -> None:
    """Display a training progress bar that updates in place."""
    fraction = scalar.multiply(float(epoch), scalar.inverse(float(total)))
    filled = int(scalar.multiply(fraction, float(width)))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = int(scalar.multiply(fraction, 100.0))
    line = f"\r  Training: epoch {epoch}/{total}  loss={loss:.4f}  [{bar}] {pct}%"
    stream.write(line)
    stream.flush()


def progress_done(stream=sys.stderr) -> None:
    """Finalize the progress bar with a newline."""
    stream.write("\n")
    stream.flush()
