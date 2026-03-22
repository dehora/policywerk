"""Level 2: Random walk environment.

Sutton's 5-state chain for demonstrating TD learning.
States A through E in a line. Start at C. Terminal at both
ends: left terminal gives reward 0, right terminal gives reward 1.
All other transitions give reward 0.

The true state values under a uniform random policy are known
analytically: [1/6, 2/6, 3/6, 4/6, 5/6].
"""

from policywerk.building_blocks.mdp import Environment, State

Vector = list[float]

_LABELS = ["A", "B", "C", "D", "E"]
_TRUE_VALUES = [1.0 / 6, 2.0 / 6, 3.0 / 6, 4.0 / 6, 5.0 / 6]


class RandomWalk(Environment):
    """5-state random walk.

    Actions: 0 = left, 1 = right.
    """

    TRUE_VALUES = list(_TRUE_VALUES)
    LABELS = list(_LABELS)

    def __init__(self):
        self._position = 2  # start at C

    def reset(self) -> State:
        self._position = 2
        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        """Move left (0) or right (1). Terminal at edges."""
        if action == 0:
            self._position -= 1
        else:
            self._position += 1

        # Left terminal (fell off left side)
        if self._position < 0:
            terminal_state = State(features=[-1.0], label="LEFT_TERMINAL")
            return terminal_state, 0.0, True

        # Right terminal (fell off right side)
        if self._position > 4:
            terminal_state = State(features=[5.0], label="RIGHT_TERMINAL")
            return terminal_state, 1.0, True

        return self._make_state(), 0.0, False

    def num_actions(self) -> int:
        return 2

    def _make_state(self) -> State:
        return State(
            features=[float(self._position)],
            label=_LABELS[self._position],
        )
