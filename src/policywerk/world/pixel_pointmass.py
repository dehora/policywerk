"""Level 2: Pixel-observed point-mass.

This class reuses PointMass for physics and just changes how the agent
perceives the state—pixels instead of coordinates.

Wraps the PointMass environment with 16x16 pixel observations.
The agent and target are rendered as small markers on a grayscale grid.
Same physics as pointmass.py, but observations are images.

Used by Dreamer (L07) to demonstrate learning a world model
from pixel observations.
"""

import math

from policywerk.building_blocks.mdp import Environment, State
from policywerk.primitives import scalar
from policywerk.world.pointmass import PointMass

Vector = list[float]
Matrix = list[list[float]]

SIZE = 16


class PixelPointMass(Environment):
    """Pixel-observed wrapper around PointMass."""

    def __init__(self, **kwargs):
        # All PointMass parameters (target, force_scale, damping, etc.) are passed through.
        self._inner = PointMass(**kwargs)

    def reset(self) -> State:
        self._inner.reset()
        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        _, reward, done = self._inner.step(action)
        return self._make_state(), reward, done

    def step_continuous(self, force: Vector) -> tuple[State, float, bool]:
        """Continuous action passthrough to inner PointMass."""
        _, reward, done = self._inner.step_continuous(force)
        return self._make_state(), reward, done

    @property
    def position(self) -> tuple[float, float]:
        """Current agent position in continuous coordinates."""
        return self._inner.position

    def num_actions(self) -> int:
        return self._inner.num_actions()

    def render_frame(self) -> Matrix:
        """Render the current state as a 16x16 pixel grid.

        Agent = 1.0, target = 0.7, empty = 0.0.
        Positions are mapped from continuous space to grid coordinates.
        """
        frame = [[0.0] * SIZE for _ in range(SIZE)]

        # Map continuous position to grid cell.
        # Grid rows map to y (vertical), columns map to x (horizontal).
        bounds = self._inner.bounds
        ax, ay = self._inner.position
        tx, ty = self._inner.target

        ar = self._to_grid(ay, bounds)
        ac = self._to_grid(ax, bounds)
        tr = self._to_grid(ty, bounds)
        tc = self._to_grid(tx, bounds)

        # Draw target first, agent on top
        frame[tr][tc] = 0.7
        frame[ar][ac] = 1.0

        return frame

    def _make_state(self) -> State:
        frame = self.render_frame()
        features = []
        for row in frame:
            features.extend(row)

        ax, ay = self._inner.position
        return State(features=features, label=f"{ax:.2f},{ay:.2f}")

    @staticmethod
    def _to_grid(val: float, bounds: float) -> int:
        """Map a continuous value in [-bounds, bounds] to [0, SIZE-1].

        Shift from [-bounds, bounds] to [0, 1], then scale to pixel coordinates.
        """
        normalized = (val + bounds) / (2.0 * bounds)
        idx = int(normalized * (SIZE - 1))
        return max(0, min(SIZE - 1, idx))
