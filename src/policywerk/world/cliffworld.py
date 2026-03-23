"""Level 2: Cliff walking environment.

4x12 grid. Start at bottom-left (3,0), goal at bottom-right (3,11).
The cliff runs along the bottom row from (3,1) to (3,10).
Stepping on the cliff gives -100 reward and teleports to start.
Normal steps cost -1. Reaching the goal gives 0 and terminates.

Layout:
  . . . . . . . . . . . .
  . . . . . . . . . . . .
  . . . . . . . . . . . .
  S C C C C C C C C C C G    S=start, C=cliff(-100), G=goal

This is the classic testbed for comparing Q-learning (risky optimal
path along the cliff edge) vs SARSA (safer path away from the cliff).

Actions: 0=North, 1=East, 2=South, 3=West.
"""

from policywerk.building_blocks.mdp import Environment, State

Vector = list[float]

_ROWS = 4
_COLS = 12
_START = (3, 0)
_GOAL = (3, 11)

# Action deltas: N, E, S, W
_ROW_DELTA = [-1, 0, 1, 0]
_COL_DELTA = [0, 1, 0, -1]


class CliffWorld(Environment):
    """4x12 cliff walking grid."""

    ROWS = _ROWS
    COLS = _COLS
    START = _START
    GOAL = _GOAL
    # List, not set — `in` is a linear scan of 10 elements on every step.
    # A set would be O(1) in production code, but for 10 cells the
    # difference is negligible and the list preserves insertion order
    # for iteration in viz code.
    CLIFF = [(3, c) for c in range(1, 11)]

    def __init__(self):
        self._pos = _START

    def reset(self) -> State:
        self._pos = _START
        return self._make_state(self._pos)

    def step(self, action: int) -> tuple[State, float, bool]:
        r, c = self._pos
        nr = r + _ROW_DELTA[action]
        nc = c + _COL_DELTA[action]

        # Boundary clamp
        nr = max(0, min(_ROWS - 1, nr))
        nc = max(0, min(_COLS - 1, nc))

        # Cliff: -100 and back to start.
        # The agent isn't eliminated — it's sent back to the start, which is
        # costly because it has to walk all the way again.
        if (nr, nc) in self.CLIFF:
            self._pos = _START
            return self._make_state(self._pos), -100.0, False

        self._pos = (nr, nc)

        # Goal — the agent isn't rewarded for reaching the goal; it's trying
        # to stop accumulating step penalties (-1 per step).
        if self._pos == _GOAL:
            return self._make_state(self._pos), 0.0, True

        return self._make_state(self._pos), -1.0, False

    def num_actions(self) -> int:
        return 4

    def _make_state(self, pos: tuple[int, int]) -> State:
        r, c = pos
        return State(features=[float(r), float(c)], label=f"{r},{c}")
