"""Level 2: Pixel gridworld (Catcher).

16x16 binary grid. The agent navigates to collect reward items
and avoid hazards. Observations are raw pixel frames — the state
that DQN must learn features from.

State.features: flattened 256 floats (the pixel grid)
  0.0 = empty, 0.3 = hazard, 0.7 = reward item, 1.0 = agent

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
    """16x16 pixel gridworld with collectible rewards and hazards."""

    def __init__(
        self,
        seed: int = 42,
        num_rewards: int = 3,
        num_hazards: int = 2,
        max_steps: int = 200,
    ):
        self._rng = create_rng(seed)
        self._num_rewards = num_rewards
        self._num_hazards = num_hazards
        self._max_steps = max_steps
        self._agent_pos = (0, 0)
        self._reward_positions: list[tuple[int, int]] = []
        self._hazard_positions: list[tuple[int, int]] = []
        self._step_count = 0
        self._collected = 0

    def reset(self) -> State:
        self._step_count = 0
        self._collected = 0

        # Place agent at center
        self._agent_pos = (SIZE // 2, SIZE // 2)

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
        # N, E, S, W
        dr = [-1, 0, 1, 0]
        dc = [0, 1, 0, -1]

        nr = max(0, min(SIZE - 1, r + dr[action]))
        nc = max(0, min(SIZE - 1, c + dc[action]))
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

        return self._make_state(), -0.01, False  # small step cost

    def num_actions(self) -> int:
        return 4

    def render_frame(self) -> Matrix:
        """Return the current state as a 16x16 pixel grid."""
        frame = [[EMPTY] * SIZE for _ in range(SIZE)]
        for r, c in self._reward_positions:
            frame[r][c] = REWARD_ITEM
        for r, c in self._hazard_positions:
            frame[r][c] = HAZARD
        ar, ac = self._agent_pos
        frame[ar][ac] = AGENT
        return frame

    def _make_state(self) -> State:
        frame = self.render_frame()
        features = []
        for row in frame:
            features.extend(row)
        r, c = self._agent_pos
        return State(features=features, label=f"{r},{c}")

    def _random_free_pos(self, occupied: set[tuple[int, int]]) -> tuple[int, int]:
        """Find a random unoccupied position."""
        while True:
            r = self._rng.randint(0, SIZE - 1)
            c = self._rng.randint(0, SIZE - 1)
            if (r, c) not in occupied:
                return (r, c)
