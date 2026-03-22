"""Level 2: Simplified 1D balance environment.

Imagine balancing a broomstick on your fingertip — the agent can only
push left or right and must prevent the pole from tipping over.

A pole hinged at a point — the agent applies left or right torque
to keep it upright. Simpler than full cart-pole: no cart position,
just angle and angular velocity.

State: (angle, angular_velocity)
Actions: 0 = left torque, 1 = right torque
Reward: +1 per step of survival
Terminal: |angle| > max_angle

The continuous angle and velocity are grouped into coarse ranges so
the agent can use a simple lookup table. 6 angle bins x 6 velocity
bins = 36 possible discrete states.
"""

import math

from policywerk.building_blocks.mdp import Environment, State

Vector = list[float]

# Discretization boundaries
_ANGLE_BINS = [-0.2, -0.1, 0.0, 0.1, 0.2]
_VEL_BINS = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _discretize(value: float, bins: list[float]) -> int:
    """Map a continuous value to a bin index."""
    for i, boundary in enumerate(bins):
        if value < boundary:
            return i
    return len(bins)


class Balance(Environment):
    """Simplified 1D balance (inverted pendulum without cart).

    Physics:
        angular_accel = gravity * sin(angle) + torque
        angular_vel += angular_accel * dt
        angle += angular_vel * dt
    """

    def __init__(
        self,
        gravity: float = 9.8,
        length: float = 1.0,
        torque_magnitude: float = 5.0,
        dt: float = 0.02,
        max_angle: float = 0.3,
        max_steps: int = 500,
    ):
        self._gravity = gravity
        self._length = length
        self._torque_mag = torque_magnitude
        self._dt = dt  # Time step: how much simulated time passes per action
        self._max_angle = max_angle  # (about 17 degrees)
        self._max_steps = max_steps
        self._angle = 0.0
        self._vel = 0.0
        self._step_count = 0

    def reset(self) -> State:
        # Small initial tilt so the pole starts slightly off-balance
        self._angle = 0.01
        self._vel = 0.0
        self._step_count = 0
        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        # Apply torque: 0 = left, 1 = right
        torque = -self._torque_mag if action == 0 else self._torque_mag

        # Gravity pulls the pole down — the further it tilts, the harder
        # gravity pulls. The agent's torque pushes against gravity.
        angular_accel = (self._gravity / self._length) * math.sin(self._angle) + torque
        self._vel += angular_accel * self._dt
        self._angle += self._vel * self._dt
        self._step_count += 1

        done = abs(self._angle) > self._max_angle or self._step_count >= self._max_steps
        # Reward 1 for each step of survival.
        # Reward 0 if the pole fell.
        # Reward 1 if survived all max_steps (success).
        reward = 0.0 if done and abs(self._angle) > self._max_angle else 1.0

        return self._make_state(), reward, done

    def num_actions(self) -> int:
        return 2

    def _make_state(self) -> State:
        # Discretized label for tabular methods
        a_bin = _discretize(self._angle, _ANGLE_BINS)
        v_bin = _discretize(self._vel, _VEL_BINS)
        return State(
            features=[self._angle, self._vel],
            label=f"{a_bin},{v_bin}",
        )
