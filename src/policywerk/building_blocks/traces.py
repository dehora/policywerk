"""Level 1: Eligibility traces.

When something good or bad happens, which past states deserve credit or blame?
A trace is a fading memory of recently visited states. The more recently a
state was visited, the more it gets updated when the agent learns something new.
"""

from policywerk.primitives import scalar


class EligibilityTrace:
    """Trace that remembers which states were recently visited.

    Used by TD(λ) and the Barto/Sutton ACE/ASE architecture.
    """

    def __init__(self, gamma: float, lam: float):
        """
        gamma: discount factor — how much to care about future versus present.
        lam: trace decay rate — how far back to spread credit.
        """
        self._traces: dict[str, float] = {}
        self._gamma = gamma
        self._lam = lam

    def get(self, state_label: str) -> float:
        return self._traces.get(state_label, 0.0)

    def visit(self, state_label: str) -> None:
        """Mark a state as visited — accumulating trace.

        The trace grows each revisit, stacking credit.
        """
        self._traces[state_label] = scalar.add(self.get(state_label), 1.0)

    def replace(self, state_label: str) -> None:
        """Mark a state as visited — replacing trace (resets to 1, doesn't stack)."""
        self._traces[state_label] = 1.0

    def decay(self) -> None:
        """Decay all traces by gamma * lambda."""
        factor = scalar.multiply(self._gamma, self._lam)
        for key in self._traces:
            self._traces[key] = scalar.multiply(self._traces[key], factor)

    def reset(self) -> None:
        """Clear all traces (start of new episode)."""
        self._traces.clear()

    def all_traces(self) -> dict[str, float]:
        return dict(self._traces)
