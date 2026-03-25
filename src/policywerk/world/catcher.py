"""Level 2: Pixel gridworld (Catcher).

Grid-based environment where the agent navigates to collect reward
items and avoid hazards. Instead of knowing its grid coordinates,
the agent sees only a pixel image. It must figure out where it is
and where items are directly from the pixels.

State.features: flattened floats (grid_size × grid_size pixels)
  0.0 = empty, 0.3 = hazard, 0.7 = reward item, 1.0 = agent
  Different brightness levels encode different objects. The neural
  network learns to distinguish them.

Actions: 0=North, 1=East, 2=South, 3=West.
"""

import random as _random

from policywerk.building_blocks.mdp import Environment, State
from policywerk.primitives.random import create_rng

Vector = list[float]
Matrix = list[list[float]]

SIZE = 16
EMPTY = 0.0
HAZARD = 0.3
REWARD_ITEM = 0.7
AGENT = 1.0


class Catcher(Environment):
    """Pixel gridworld with collectible rewards and hazards.

    grid_size controls the observation dimensions. Default 16 (16×16 = 256
    pixel inputs). Smaller grids (e.g. 8×8 = 64 inputs) are easier to learn
    with small networks in pure Python.
    """

    def __init__(
        self,
        seed: int = 42,
        num_rewards: int = 3,
        num_hazards: int = 2,
        max_steps: int = 200,
        grid_size: int = SIZE,
    ):
        if grid_size < 1:
            raise ValueError(
                f"grid_size must be >= 1, got {grid_size}"
            )
        total_objects = 1 + num_rewards + num_hazards  # agent + items
        capacity = grid_size * grid_size
        if total_objects > capacity:
            raise ValueError(
                f"Grid {grid_size}x{grid_size} ({capacity} cells) cannot fit "
                f"{total_objects} objects (1 agent + {num_rewards} rewards + "
                f"{num_hazards} hazards)"
            )
        self._rng = create_rng(seed)
        self._num_rewards = num_rewards
        self._num_hazards = num_hazards
        self._max_steps = max_steps
        self._grid_size = grid_size
        self._agent_pos = (0, 0)
        self._reward_positions: list[tuple[int, int]] = []
        self._hazard_positions: list[tuple[int, int]] = []
        self._step_count = 0
        self._collected = 0

    def reset(self) -> State:
        self._step_count = 0
        self._collected = 0
        gs = self._grid_size

        # Place agent at center
        self._agent_pos = (gs // 2, gs // 2)

        # Place reward items and hazards at random positions
        occupied = {self._agent_pos}
        self._reward_positions = []
        self._hazard_positions = []

        for _ in range(self._num_rewards):
            pos = self._random_free_pos(occupied)
            self._reward_positions.append(pos)
            occupied.add(pos)

        for _ in range(self._num_hazards):
            pos = self._random_free_pos(occupied)
            self._hazard_positions.append(pos)
            occupied.add(pos)

        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        r, c = self._agent_pos
        gs = self._grid_size
        # N, E, S, W
        dr = [-1, 0, 1, 0]
        dc = [0, 1, 0, -1]

        nr = max(0, min(gs - 1, r + dr[action]))
        nc = max(0, min(gs - 1, c + dc[action]))
        self._agent_pos = (nr, nc)
        self._step_count += 1

        # Check reward item
        if self._agent_pos in self._reward_positions:
            self._reward_positions.remove(self._agent_pos)
            self._collected += 1
            # All collected → done
            if not self._reward_positions:
                return self._make_state(), 1.0, True
            return self._make_state(), 1.0, False

        # Check hazard
        if self._agent_pos in self._hazard_positions:
            return self._make_state(), -1.0, True

        # Max steps
        if self._step_count >= self._max_steps:
            return self._make_state(), 0.0, True

        # Small step cost encourages collecting items quickly rather than wandering
        return self._make_state(), -0.01, False

    def num_actions(self) -> int:
        return 4

    def render_frame(self) -> Matrix:
        """Return the current state as a grid_size × grid_size pixel grid."""
        gs = self._grid_size
        frame = [[EMPTY] * gs for _ in range(gs)]
        for r, c in self._reward_positions:
            frame[r][c] = REWARD_ITEM
        for r, c in self._hazard_positions:
            frame[r][c] = HAZARD
        ar, ac = self._agent_pos
        frame[ar][ac] = AGENT
        return frame

    def _make_state(self) -> State:
        # The agent's observation is the raw pixel grid flattened into a
        # list — no direct access to position coordinates.
        frame = self.render_frame()
        features = []
        for row in frame:
            features.extend(row)
        r, c = self._agent_pos
        return State(features=features, label=f"{r},{c}")

    def _random_free_pos(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find a random unoccupied position."""
        gs = self._grid_size
        while True:
            r = self._rng.randint(0, gs - 1)
            c = self._rng.randint(0, gs - 1)
            if (r, c) not in occupied:
                return (r, c)
