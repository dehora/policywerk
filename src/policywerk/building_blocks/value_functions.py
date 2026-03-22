"""Level 1: Tabular value functions.

A value function answers 'how good is this situation?' It stores, for each
state, the agent's estimate of how much total future reward it can expect
from there.

Dict-based V(s) and Q(s,a) tables used by the first four lessons.
Once we move to neural function approximation in L05, these
are replaced by network outputs.
"""


class TabularV:
    """State value function V(s) — how good is this state?

    Maps each state to the agent's estimate of total future reward from
    that point.
    """

    def __init__(self, default: float = 0.0):
        self._values: dict[str, float] = {}
        self._default = default

    def get(self, state_label: str) -> float:
        return self._values.get(state_label, self._default)

    def set(self, state_label: str, value: float) -> None:
        self._values[state_label] = value

    def update(self, state_label: str, delta: float) -> None:
        """Add delta to current value: V(s) += delta.

        delta: amount to add (not TD error — just a number to adjust the value by).
        """
        self._values[state_label] = self.get(state_label) + delta

    def all_values(self) -> dict[str, float]:
        return dict(self._values)

    def max_change(self, other: "TabularV") -> float:
        """Maximum absolute difference between two value functions.

        Used to check convergence — when the biggest change is tiny,
        the values have stabilized.
        """
        max_diff = 0.0
        all_keys = set(self._values.keys()) | set(other._values.keys())
        for key in all_keys:
            diff = abs(self.get(key) - other.get(key))
            if diff > max_diff:
                max_diff = diff
        return max_diff


class TabularQ:
    """Action-value function Q(s,a) — how good is this action in this state?

    Maps each (state, action) pair to an estimated value.
    """

    def __init__(self, default: float = 0.0):
        self._values: dict[tuple[str, int], float] = {}
        self._default = default

    def get(self, state_label: str, action: int) -> float:
        return self._values.get((state_label, action), self._default)

    def set(self, state_label: str, action: int, value: float) -> None:
        self._values[(state_label, action)] = value

    def update(self, state_label: str, action: int, delta: float) -> None:
        """Add delta to current value: Q(s,a) += delta.

        delta: amount to add (not TD error — just a number to adjust the value by).
        """
        key = (state_label, action)
        self._values[key] = self._values.get(key, self._default) + delta

    def best_action(self, state_label: str, num_actions: int) -> int:
        """Return the action with highest Q-value for this state."""
        best_a = 0
        best_val = self.get(state_label, 0)
        for a in range(1, num_actions):
            val = self.get(state_label, a)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def max_value(self, state_label: str, num_actions: int) -> float:
        """Return max_a Q(s, a)."""
        best = self.get(state_label, 0)
        for a in range(1, num_actions):
            val = self.get(state_label, a)
            if val > best:
                best = val
        return best
