"""Level 2: Deterministic gridworld.

5x5 grid with a goal (+1 reward), a pit (-1 reward), and walls.
Implements StochasticMDP so planning algorithms can ask 'what would
happen if I take action A from state S?' without actually taking the step.

Default layout:
  . . . . G     G = goal (+1, terminal)
  . W . X .     W = wall (impassable)
  . . . . .     X = pit (-1, terminal)
  . . . . .     . = empty cell (-0.04 step cost)
  S . . . .     S = start

Actions: 0=North, 1=East, 2=South, 3=West.
Hitting a wall or boundary leaves the agent in place.
"""

from policywerk.building_blocks.mdp import Environment, StochasticMDP, State

Vector = list[float]

# Cell types
EMPTY = 0
WALL = 1
GOAL = 2
PIT = 3

_DEFAULT_GRID = [
    [EMPTY, EMPTY, EMPTY, EMPTY, GOAL],
    [EMPTY, WALL,  EMPTY, PIT,   EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
    [EMPTY, EMPTY, EMPTY, EMPTY, EMPTY],
]

# Action deltas: N, E, S, W
_ROW_DELTA = [-1, 0, 1, 0]
_COL_DELTA = [0, 1, 0, -1]

# A small penalty for each step encourages the agent to reach the goal
# quickly rather than wandering.
STEP_COST = -0.04
GOAL_REWARD = 1.0
PIT_REWARD = -1.0


class GridWorld(StochasticMDP):
    """5x5 deterministic gridworld with known transition dynamics."""

    def __init__(self, grid: list[list[int]] | None = None, start: tuple[int, int] = (4, 0)):
        self._grid = [list(row) for row in (grid or _DEFAULT_GRID)]
        self._rows = len(self._grid)
        self._cols = len(self._grid[0])
        self._start = start
        self._pos = start

        # Precompute walls, goals, pits for viz
        self.walls: list[tuple[int, int]] = []
        self.goals: list[tuple[int, int]] = []
        self.pits: list[tuple[int, int]] = []
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] == WALL:
                    self.walls.append((r, c))
                elif self._grid[r][c] == GOAL:
                    self.goals.append((r, c))
                elif self._grid[r][c] == PIT:
                    self.pits.append((r, c))

    def reset(self) -> State:
        self._pos = self._start
        return self._make_state(self._pos)

    def step(self, action: int) -> tuple[State, float, bool]:
        r, c = self._pos
        nr = r + _ROW_DELTA[action]
        nc = c + _COL_DELTA[action]

        # Boundary or wall → stay in place
        if not (0 <= nr < self._rows and 0 <= nc < self._cols):
            nr, nc = r, c
        if self._grid[nr][nc] == WALL:
            nr, nc = r, c

        self._pos = (nr, nc)
        cell = self._grid[nr][nc]

        if cell == GOAL:
            return self._make_state(self._pos), GOAL_REWARD, True
        if cell == PIT:
            return self._make_state(self._pos), PIT_REWARD, True

        return self._make_state(self._pos), STEP_COST, False

    def num_actions(self) -> int:
        return 4

    def states(self) -> list[State]:
        """All non-wall states, including terminal (goal/pit) states.

        Terminal states are included because dynamic programming algorithms
        need to know about them — their value is defined (goal=+1, pit=-1)
        and they must appear in the state space for value iteration to work.
        """
        result = []
        for r in range(self._rows):
            for c in range(self._cols):
                if self._grid[r][c] != WALL:
                    result.append(self._make_state((r, c)))
        return result

    def transition_probs(
        self, state: State, action: int
    ) -> list[tuple[State, float, float]]:
        """Deterministic: one outcome with probability 1.0.

        step() moves the agent. transition_probs() answers the same question
        hypothetically — used by planning algorithms that reason about all
        possibilities.

        Terminal states (goal/pit) are absorbing: any action from a terminal
        state loops back to itself with zero reward.
        """
        r, c = self._parse_label(state.label)

        # Terminal states are absorbing — no escape, no further reward
        if self._grid[r][c] in (GOAL, PIT):
            return [(self._make_state((r, c)), 1.0, 0.0)]

        nr = r + _ROW_DELTA[action]
        nc = c + _COL_DELTA[action]

        if not (0 <= nr < self._rows and 0 <= nc < self._cols):
            nr, nc = r, c
        if self._grid[nr][nc] == WALL:
            nr, nc = r, c

        cell = self._grid[nr][nc]
        if cell == GOAL:
            reward = GOAL_REWARD
        elif cell == PIT:
            reward = PIT_REWARD
        else:
            reward = STEP_COST

        next_state = self._make_state((nr, nc))
        return [(next_state, 1.0, reward)]

    def is_terminal(self, state: State) -> bool:
        """True if state is a goal or pit."""
        r, c = self._parse_label(state.label)
        return self._grid[r][c] in (GOAL, PIT)

    def grid_values(self, value_fn) -> list[list[float]]:
        """Converts the agent's internal value estimates into a grid for heatmap visualization.

        value_fn: anything with a .get(label) method (TabularV).
        Walls get 0.0.
        """
        result = []
        for r in range(self._rows):
            row = []
            for c in range(self._cols):
                if self._grid[r][c] == WALL:
                    row.append(0.0)
                else:
                    row.append(value_fn.get(f"{r},{c}"))
            result.append(row)
        return result

    def _make_state(self, pos: tuple[int, int]) -> State:
        r, c = pos
        return State(features=[float(r), float(c)], label=f"{r},{c}")

    @staticmethod
    def _parse_label(label: str) -> tuple[int, int]:
        parts = label.split(",")
        return int(parts[0]), int(parts[1])
