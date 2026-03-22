"""Level 2: Point-mass continuous control.

A point mass in 2D space must reach a target position.
Physics: simple Newtonian — force → acceleration → velocity → position,
with damping to prevent oscillation.

State: (x, y, x_dot, y_dot, target_x, target_y)
Actions (discrete): 8 directions + stay = 9 actions
Actions (continuous): (force_x, force_y) via step_continuous()

Reward: negative distance to target (closer = less negative).
Terminal: within threshold of target, or max steps.

PPO (L06) uses step_continuous() for continuous policy optimization.
"""

import math

from policywerk.building_blocks.mdp import Environment, State
from policywerk.primitives import scalar

Vector = list[float]

# 9 discrete actions: 8 compass directions + stay
_FORCE_MAP = [
    (0.0, -1.0),   # 0: N
    (1.0, -1.0),   # 1: NE
    (1.0, 0.0),    # 2: E
    (1.0, 1.0),    # 3: SE
    (0.0, 1.0),    # 4: S
    (-1.0, 1.0),   # 5: SW
    (-1.0, 0.0),   # 6: W
    (-1.0, -1.0),  # 7: NW
    (0.0, 0.0),    # 8: stay
]


class PointMass(Environment):
    """2D point-mass reaching a target."""

    def __init__(
        self,
        target: tuple[float, float] = (0.8, 0.8),
        force_scale: float = 0.5,
        damping: float = 0.95,
        dt: float = 0.1,
        reach_threshold: float = 0.1,
        max_steps: int = 200,
        bounds: float = 2.0,
    ):
        self._target = target
        self._force_scale = force_scale
        self._damping = damping
        self._dt = dt
        self._threshold = reach_threshold
        self._max_steps = max_steps
        self._bounds = bounds
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._step_count = 0

    def reset(self) -> State:
        self._x = 0.0
        self._y = 0.0
        self._vx = 0.0
        self._vy = 0.0
        self._step_count = 0
        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        """Discrete action: apply one of 9 force directions."""
        fx, fy = _FORCE_MAP[action]
        return self._apply_force(
            scalar.multiply(fx, self._force_scale),
            scalar.multiply(fy, self._force_scale),
        )

    def step_continuous(self, force: Vector) -> tuple[State, float, bool]:
        """Continuous action: apply (force_x, force_y) directly.

        Used by PPO and Dreamer for continuous policy optimization.
        """
        fx = scalar.clamp(force[0], -1.0, 1.0) * self._force_scale
        fy = scalar.clamp(force[1], -1.0, 1.0) * self._force_scale
        return self._apply_force(fx, fy)

    def num_actions(self) -> int:
        return 9

    @property
    def position(self) -> tuple[float, float]:
        return (self._x, self._y)

    @property
    def target(self) -> tuple[float, float]:
        return self._target

    def _apply_force(self, fx: float, fy: float) -> tuple[State, float, bool]:
        # Newtonian: a = F (unit mass), v += a*dt, x += v*dt
        self._vx = scalar.multiply(scalar.add(self._vx, scalar.multiply(fx, self._dt)), self._damping)
        self._vy = scalar.multiply(scalar.add(self._vy, scalar.multiply(fy, self._dt)), self._damping)
        self._x = scalar.add(self._x, scalar.multiply(self._vx, self._dt))
        self._y = scalar.add(self._y, scalar.multiply(self._vy, self._dt))

        # Clamp to bounds
        self._x = scalar.clamp(self._x, -self._bounds, self._bounds)
        self._y = scalar.clamp(self._y, -self._bounds, self._bounds)

        self._step_count += 1

        # Distance to target
        dx = scalar.subtract(self._x, self._target[0])
        dy = scalar.subtract(self._y, self._target[1])
        dist = math.sqrt(scalar.add(scalar.multiply(dx, dx), scalar.multiply(dy, dy)))

        # Reward: negative distance
        reward = scalar.negate(dist)

        # Terminal: reached target or max steps
        done = dist < self._threshold or self._step_count >= self._max_steps

        return self._make_state(), reward, done

    def _make_state(self) -> State:
        return State(
            features=[self._x, self._y, self._vx, self._vy,
                      self._target[0], self._target[1]],
            label=f"{self._x:.2f},{self._y:.2f}",
        )
