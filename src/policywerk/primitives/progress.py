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

# ASCII fallback when the stream can't encode braille
_ASCII_SPINNER = "|/-\\"


def _spinner_chars(stream) -> str:
    """Return the best spinner character set the stream can handle."""
    try:
        _BRAILLE[0].encode(getattr(stream, "encoding", "utf-8") or "utf-8")
        return _BRAILLE
    except (UnicodeEncodeError, LookupError):
        return _ASCII_SPINNER


def _is_tty(stream) -> bool:
    """Return True if *stream* is connected to a terminal."""
    return hasattr(stream, "isatty") and stream.isatty()


def progress_bar(
    step: int,
    total: int,
    info: str = "",
    width: int = 30,
    stream=sys.stderr,
) -> None:
    """Display a training progress bar that updates in place.

    step: current step (1-indexed).
    total: total number of steps.
    info: caller-formatted status string (e.g. "reward=0.94", "loss=0.003").
    """
    fraction = scalar.multiply(float(step), scalar.inverse(float(total)))
    filled = int(scalar.multiply(fraction, float(width)))
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = int(scalar.multiply(fraction, 100.0))
    info_part = f"  {info}" if info else ""
    pad = len(str(total))
    line = f"\r    Training: {step:{pad}d}/{total}{info_part}  [{bar}] {pct:3d}%"
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
        self._chars = _spinner_chars(stream)
        self._tty = _is_tty(stream)

    def __enter__(self):
        self._stop.clear()
        # Only animate on TTY streams — non-TTY (pipes, files) get
        # a single clean "done." line with no spinner frames.
        if self._tty:
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread:
            self._thread.join()
        status = "failed." if exc_type is not None else "done."
        done_text = f"    {self._message}... {status}"
        if self._tty:
            # Overwrite the spinner line — pad with spaces to clear
            # any residual characters from the spinning text
            padding = " " * 10
            self._stream.write(f"\r{done_text}{padding}\n")
        else:
            # Non-TTY: single clean line, no \r, no spinner residue
            self._stream.write(f"{done_text}\n")
        self._stream.flush()

    def _spin(self):
        idx = 0
        while not self._stop.is_set():
            char = self._chars[idx % len(self._chars)]
            self._stream.write(f"\r    {char} {self._message}...")
            self._stream.flush()
            idx += 1
            time.sleep(0.1)
