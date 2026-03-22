"""Level 0: Training progress display.

Terminal progress bar for long-running training loops and a braille
spinner for background work (animation export, file generation).
"""

import sys
import threading
import time

from policywerk.primitives import scalar

# Braille spinner characters — smooth rotation effect in the terminal
_BRAILLE = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


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


class Spinner:
    """Braille spinner for indicating background work.

    Usage:
        with Spinner("Generating animation"):
            save_animation(...)
        # prints: ⠋ Generating animation... done.
    """

    def __init__(self, message: str, stream=sys.stdout):
        self._message = message
        self._stream = stream
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join()
        self._stream.write(f"\r    {_BRAILLE[0]} {self._message}... done.\n")
        self._stream.flush()

    def _spin(self):
        idx = 0
        while not self._stop.is_set():
            char = _BRAILLE[idx % len(_BRAILLE)]
            self._stream.write(f"\r    {char} {self._message}...")
            self._stream.flush()
            idx += 1
            time.sleep(0.1)
