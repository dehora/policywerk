"""Level 1: Training metrics logging.

Simple logger that accumulates scalar metrics during training
and can summarize them for display.
"""

from dataclasses import dataclass, field


@dataclass
class MetricLog:
    """Accumulates values for a single metric over time."""
    name: str
    values: list[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        self.values.append(value)

    @property
    def last(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

    def recent_mean(self, n: int = 100) -> float:
        """Mean of the last n values."""
        window = self.values[-n:]
        if not window:
            return 0.0
        return sum(window) / len(window)


class TrainingLog:
    """Collection of named metrics."""

    def __init__(self):
        self._metrics: dict[str, MetricLog] = {}

    def record(self, name: str, value: float) -> None:
        if name not in self._metrics:
            self._metrics[name] = MetricLog(name=name)
        self._metrics[name].record(value)

    def get(self, name: str) -> MetricLog:
        return self._metrics.get(name, MetricLog(name=name))

    def summary(self, recent_n: int = 100) -> str:
        """One-line summary of all metrics (recent means)."""
        parts = []
        for name, metric in self._metrics.items():
            parts.append(f"{name}={metric.recent_mean(recent_n):.4f}")
        return "  ".join(parts)
