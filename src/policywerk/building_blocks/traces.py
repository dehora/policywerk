"""Level 1: Eligibility traces.

Credit assignment mechanism that bridges Monte Carlo (full episode)
and TD(0) (single step). A trace marks recently visited states so
the TD error can update them proportionally to recency.
"""

from policywerk.primitives import scalar


class EligibilityTrace:
    """Trace that remembers which states were recently visited.

    Used by TD(λ) and the Barto/Sutton ACE/ASE architecture.
    """

    def __init__(self, gamma: float, lam: float):
        self._traces: dict[str, float] = {}
        self._gamma = gamma
        self._lam = lam

    def get(self, state_label: str) -> float:
        return self._traces.get(state_label, 0.0)

    def visit(self, state_label: str) -> None:
        """Mark a state as visited — accumulating trace."""
        self._traces[state_label] = scalar.add(self.get(state_label), 1.0)

    def replace(self, state_label: str) -> None:
        """Mark a state as visited — replacing trace (set to 1)."""
        self._traces[state_label] = 1.0

    def decay(self) -> None:
        """Decay all traces by γλ."""
        factor = scalar.multiply(self._gamma, self._lam)
        for key in self._traces:
            self._traces[key] = scalar.multiply(self._traces[key], factor)

    def reset(self) -> None:
        """Clear all traces (start of new episode)."""
        self._traces.clear()

    def all_traces(self) -> dict[str, float]:
        return dict(self._traces)
