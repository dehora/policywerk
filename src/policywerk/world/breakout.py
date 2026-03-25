"""Level 2: Mini Breakout.

8×10 pixel grid. The agent controls a paddle at the bottom and must
bounce a ball into bricks at the top. The agent sees the pixel grid
(80 values) plus ball velocity (2 values, vertical and horizontal
direction). Velocity is included because a single frame is ambiguous:
the same pixel layout can occur with different ball directions.

Layout:
  row 0: . B B B B B B .
  row 1: . B B B B B B .
  row 2: . . . . . . . .
  row 3: . . . . o . . .   ball starts here, moving down-right
  row 4-8: empty
  row 9: . . . = = = . .   paddle (width 3), starts centered

Pixel encoding:
  0.0 = empty
  0.5 = brick
  0.7 = ball
  1.0 = paddle

Actions: 0=left, 1=stay, 2=right.
"""

from policywerk.building_blocks.mdp import Environment, State

Vector = list[float]
Matrix = list[list[float]]

ROWS = 10
COLS = 8
PADDLE_WIDTH = 3

EMPTY = 0.0
BRICK = 0.5
BALL = 0.7
PADDLE = 1.0

# Brick layout: rows 0-1, columns 1-6
_BRICK_ROWS = [0, 1]
_BRICK_COLS = range(1, 7)


class Breakout(Environment):
    """8×10 mini Breakout with deterministic ball start."""

    def __init__(self, max_steps: int = 200):
        self._max_steps = max_steps
        self._paddle_col = 0
        self._ball_r = 0
        self._ball_c = 0
        self._ball_dr = 0
        self._ball_dc = 0
        self._bricks: set[tuple[int, int]] = set()
        self._steps = 0
        self._score = 0
        self._done = False

    def reset(self) -> State:
        self._steps = 0
        self._score = 0
        self._done = False

        # Paddle centered
        self._paddle_col = (COLS - PADDLE_WIDTH) // 2

        # Ball at row 3, col 4, moving down-right
        self._ball_r = 3
        self._ball_c = 4
        self._ball_dr = 1
        self._ball_dc = 1

        # Place bricks
        self._bricks = set()
        for r in _BRICK_ROWS:
            for c in _BRICK_COLS:
                self._bricks.add((r, c))

        return self._make_state()

    def step(self, action: int) -> tuple[State, float, bool]:
        if self._done:
            return self._make_state(), 0.0, True

        # Move paddle
        if action == 0:  # left
            self._paddle_col = max(0, self._paddle_col - 1)
        elif action == 2:  # right
            self._paddle_col = min(COLS - PADDLE_WIDTH, self._paddle_col + 1)
        # action == 1: stay

        # Move ball
        next_r = self._ball_r + self._ball_dr
        next_c = self._ball_c + self._ball_dc

        # Wall bounces (left/right)
        if next_c < 0:
            next_c = 0
            self._ball_dc = -self._ball_dc
        elif next_c >= COLS:
            next_c = COLS - 1
            self._ball_dc = -self._ball_dc

        # Top wall bounce
        if next_r < 0:
            next_r = 0
            self._ball_dr = -self._ball_dr

        reward = -0.01  # step cost

        # Brick collision
        if (next_r, next_c) in self._bricks:
            self._bricks.remove((next_r, next_c))
            self._ball_dr = -self._ball_dr
            # Don't move into the brick cell — bounce back
            next_r = self._ball_r
            self._score += 1
            reward = 1.0

            # All bricks cleared
            if not self._bricks:
                self._ball_r = next_r
                self._ball_c = next_c
                self._done = True
                self._steps += 1
                return self._make_state(), reward, True

        # Paddle bounce: ball reaches paddle row
        if next_r >= ROWS - 1:
            paddle_cells = range(self._paddle_col, self._paddle_col + PADDLE_WIDTH)
            if next_c in paddle_cells:
                # Bounce up
                next_r = ROWS - 2
                self._ball_dr = -self._ball_dr
            else:
                # Miss — ball passed the paddle
                self._ball_r = next_r
                self._ball_c = next_c
                self._done = True
                self._steps += 1
                return self._make_state(), -1.0, True

        self._ball_r = next_r
        self._ball_c = next_c
        self._steps += 1

        # Max steps
        if self._steps >= self._max_steps:
            self._done = True
            return self._make_state(), reward, True

        return self._make_state(), reward, False

    def num_actions(self) -> int:
        return 3

    def score(self) -> int:
        """Number of bricks destroyed so far."""
        return self._score

    def bricks_remaining(self) -> int:
        """Number of bricks still on the board."""
        return len(self._bricks)

    def render_frame(self) -> Matrix:
        """Return current state as a 10×8 pixel grid."""
        frame = [[EMPTY] * COLS for _ in range(ROWS)]

        # Bricks
        for r, c in self._bricks:
            frame[r][c] = BRICK

        # Ball
        if 0 <= self._ball_r < ROWS and 0 <= self._ball_c < COLS:
            frame[self._ball_r][self._ball_c] = BALL

        # Paddle
        for dc in range(PADDLE_WIDTH):
            pc = self._paddle_col + dc
            if 0 <= pc < COLS:
                frame[ROWS - 1][pc] = PADDLE

        return frame

    def render_color_frame(self) -> list[list[list[float]]]:
        """Return an RGB frame for display (not for the network).

        Colors match classic Breakout:
          row 0 bricks: red, row 1 bricks: orange,
          ball: white, paddle: blue, background: black.
        """
        # RGB tuples
        bg = [0.0, 0.0, 0.0]
        brick_colors = {
            0: [0.9, 0.2, 0.2],   # red
            1: [0.9, 0.6, 0.2],   # orange
        }
        ball_rgb = [1.0, 1.0, 1.0]
        paddle_rgb = [0.3, 0.5, 0.9]

        frame = [
            [list(bg) for _ in range(COLS)]
            for _ in range(ROWS)
        ]

        for r, c in self._bricks:
            frame[r][c] = list(brick_colors.get(r, [0.7, 0.7, 0.7]))

        if 0 <= self._ball_r < ROWS and 0 <= self._ball_c < COLS:
            frame[self._ball_r][self._ball_c] = list(ball_rgb)

        for dc in range(PADDLE_WIDTH):
            pc = self._paddle_col + dc
            if 0 <= pc < COLS:
                frame[ROWS - 1][pc] = list(paddle_rgb)

        return frame

    def _make_state(self) -> State:
        frame = self.render_frame()
        features: list[float] = []
        for row in frame:
            features.extend(row)
        # Append ball velocity so the observation is Markov: the same
        # pixel frame can occur with different ball directions, and
        # without velocity the agent cannot distinguish them.
        features.append(float(self._ball_dr))
        features.append(float(self._ball_dc))
        return State(
            features=features,
            label=f"p{self._paddle_col},b{self._ball_r},{self._ball_c}",
        )
