"""Visualization: Trajectories, agents, pixel rendering.

Agent paths through environments, marker drawing, pixel-grid
rendering for DQN/Dreamer, policy distribution curves, and
real-vs-imagined split-screen for world models.
"""

import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend—renders to files
import matplotlib.pyplot as plt

from policywerk.viz.animate import TEAL, ORANGE, LIGHT_GRAY, DARK_GRAY

Vector = list[float]
Matrix = list[list[float]]


def draw_trajectory(
    ax: plt.Axes,
    positions: list[tuple[float, float]],
    color: str = TEAL,
    alpha: float = 0.6,
    linewidth: float = 1.5,
) -> None:
    """Draw the path an agent took through a sequence of (x, y) positions."""
    if len(positions) < 2:
        return
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)


def draw_agent(
    ax: plt.Axes,
    position: tuple[float, float],
    color: str = TEAL,
    size: float = 80,
) -> None:
    """Draw an agent marker at the given position."""
    # zorder=10 ensures the agent is drawn on top of grid lines and trajectories
    ax.scatter([position[0]], [position[1]], c=color, s=size,
               zorder=10, edgecolors=DARK_GRAY, linewidths=1)


def draw_target(
    ax: plt.Axes,
    position: tuple[float, float],
    color: str = ORANGE,
    size: float = 100,
) -> None:
    """Draw a target marker (star) at the given position."""
    ax.scatter([position[0]], [position[1]], c=color, s=size,
               marker="*", zorder=10, edgecolors=DARK_GRAY, linewidths=0.5)


def draw_cliff_grid(
    ax: plt.Axes,
    rows: int,
    cols: int,
    cliff: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
    policy: dict[str, int] | None = None,
    caption: str | None = None,
    agent_pos: tuple[int, int] | None = None,
    cell_values: dict[tuple[int, int], float] | None = None,
) -> None:
    """Draw a cliff walking grid with value-gradient cell coloring.

    agent_pos: if set, that cell is colored light blue.

    Cell colors are determined by cell_values if provided
    (mapping (row, col) -> float), using the same green gradient as
    L01's value heatmap. Otherwise cells are light gray.
    """
    ax.clear()
    import matplotlib.patches as patches

    cliff_set = set(cliff)

    # Determine value range for color mapping
    cell_vals = cell_values or {}
    if cell_vals:
        all_vals = list(cell_vals.values())
        vmin = min(all_vals)
        vmax = max(all_vals)
        if vmin == vmax:
            vmin, vmax = vmin - 1, vmax + 1
    else:
        vmin, vmax = -1, 0

    # Draw grid cells
    for r in range(rows):
        for c in range(cols):
            if (r, c) in cliff_set:
                color = "#cc3333"  # bold red for cliff
            elif agent_pos is not None and (r, c) == agent_pos:
                color = "#a8d8ea"  # light blue for agent position
            elif (r, c) in cell_vals:
                # Green gradient: higher value = more green (like L01)
                val = cell_vals[(r, c)]
                frac = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                # Blend: yellow(low) -> green(high), cap at 0.75 to keep darkest readable
                import matplotlib.cm as cm
                rgba = cm.YlGn(frac * 0.75)
                color = f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}"
            elif (r, c) == start:
                color = "#e8e8e8"
            elif (r, c) == goal:
                color = "#85e085"  # green for goal
            else:
                color = "#f0f0f0"  # light gray
            rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                      facecolor=color, edgecolor="#cccccc",
                                      linewidth=0.5, zorder=1)
            ax.add_patch(rect)

    # Row labels on the left (cartesian: bottom row = 0), column labels on the bottom
    for r in range(rows):
        cartesian_r = (rows - 1) - r
        ax.text(-0.85, r, str(cartesian_r), ha="center", va="center",
                fontsize=6, color="#999999")
    for c in range(cols):
        ax.text(c, rows - 0.35, str(c), ha="center", va="top",
                fontsize=6, color="#999999")

    # Label start and goal prominently (white text with dark outline for contrast)
    import matplotlib.patheffects as pe
    marker_effects = [pe.withStroke(linewidth=2.5, foreground=DARK_GRAY)]
    ax.text(start[1], start[0], "S", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=9,
            path_effects=marker_effects)
    ax.text(goal[1], goal[0], "G", ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=9,
            path_effects=marker_effects)

    # Draw policy arrows
    if policy:
        arrow_dx = {0: 0, 1: 0.3, 2: 0, 3: -0.3}
        arrow_dy = {0: -0.3, 1: 0, 2: 0.3, 3: 0}
        for label, action in policy.items():
            parts = label.split(",")
            r, c = int(parts[0]), int(parts[1])
            if (r, c) in cliff_set or (r, c) == goal:
                continue
            dx = arrow_dx.get(action, 0)
            dy = arrow_dy.get(action, 0)
            ax.annotate("", xy=(c + dx, r + dy), xytext=(c, r),
                         arrowprops=dict(arrowstyle="->", color=DARK_GRAY,
                                         lw=1.2, alpha=0.7),
                         zorder=4)

    # Agent marker (orange circle) at current position, skip at goal
    if agent_pos is not None and agent_pos != goal:
        ar, ac = agent_pos
        ax.scatter([ac], [ar], c=ORANGE, s=180, zorder=8,
                   edgecolors=DARK_GRAY, linewidths=1.5)

    ax.set_xlim(-1.2, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # invert y so row 0 is at top
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if caption:
        ax.text(cols / 2 - 0.5, rows + 0.1, caption,
                ha="center", va="top", fontsize=7, color=DARK_GRAY, style="italic")


def draw_chain(
    ax: plt.Axes,
    labels: list[str],
    values: list[float] | None = None,
    path: list[str] | None = None,
    outcome: str | None = None,
    agent_label: str | None = None,
    caption: str | None = None,
) -> None:
    """Draw a chain of states (e.g. the random walk A-B-C-D-E).

    labels: state names in order (left to right).
    values: optional value estimates for each state (shown as color fill).
    path: optional list of state labels visited this episode (drawn as line).
    outcome: "left" or "right"—which terminal was reached.
    agent_label: if set, draw an agent marker above this state node.
    caption: optional text displayed below the chain explaining the task.
    """
    n = len(labels)
    node_y = 0.5
    spacing = 1.0

    ax.clear()
    ax.set_xlim(-1.5, n * spacing + 0.5)
    ax.set_ylim(-0.6, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Optional caption below the chain
    if caption:
        mid_x = (n - 1) * spacing / 2
        ax.text(mid_x, -0.45, caption,
                ha="center", va="center", fontsize=7, color=DARK_GRAY, style="italic")

    # Draw terminal zones
    ax.text(-1.0, node_y, "[0]", ha="center", va="center", fontsize=11,
            fontweight="bold", color="red" if outcome == "left" else DARK_GRAY)
    ax.text(n * spacing, node_y, "[+1]", ha="center", va="center", fontsize=11,
            fontweight="bold", color="green" if outcome == "right" else DARK_GRAY)

    # Draw connecting lines between nodes
    for i in range(n - 1):
        ax.plot([i * spacing + 0.2, (i + 1) * spacing - 0.2], [node_y, node_y],
                color=LIGHT_GRAY, linewidth=2, zorder=1)
    # Lines to terminals
    ax.plot([-0.6, -0.2], [node_y, node_y], color=LIGHT_GRAY, linewidth=2, zorder=1)
    ax.plot([(n - 1) * spacing + 0.2, n * spacing - 0.4], [node_y, node_y],
            color=LIGHT_GRAY, linewidth=2, zorder=1)

    # Draw the path if provided (line through visited nodes)
    if path and len(path) > 1:
        path_xs = []
        path_ys = []
        for label in path:
            if label in labels:
                idx = labels.index(label)
                path_xs.append(idx * spacing)
                path_ys.append(node_y)
        # Add terminal position
        if outcome == "left":
            path_xs.append(-1.0)
            path_ys.append(node_y)
        elif outcome == "right":
            path_xs.append(n * spacing)
            path_ys.append(node_y)
        # Offset path slightly below nodes so it's visible
        path_ys_offset = [y - 0.12 for y in path_ys]
        ax.plot(path_xs, path_ys_offset, color=TEAL, linewidth=1.5, alpha=0.5, zorder=2)

    # Draw nodes
    for i, label in enumerate(labels):
        x = i * spacing
        # Color node by value if provided
        if values is not None:
            val = values[i]
            # Map value 0-1 to color intensity (white to teal)
            r = int(255 - val * (255 - 92))   # 92 = 0x5C (TEAL red)
            g = int(255 - val * (255 - 184))   # 184 = 0xB8 (TEAL green)
            b = int(255 - val * (255 - 178))   # 178 = 0xB2 (TEAL blue)
            color = f"#{r:02x}{g:02x}{b:02x}"
        else:
            color = "white"

        circle = plt.Circle((x, node_y), 0.18, facecolor=color,
                             edgecolor=DARK_GRAY, linewidth=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, node_y, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=DARK_GRAY, zorder=6)

        # Show value below node
        if values is not None:
            ax.text(x, node_y - 0.28, f"{values[i]:.2f}", ha="center",
                    va="top", fontsize=7, color=DARK_GRAY)

    # Draw agent marker above current position
    if agent_label is not None:
        if agent_label in labels:
            idx = labels.index(agent_label)
            agent_x = idx * spacing
        elif agent_label == "LEFT_TERMINAL":
            agent_x = -1.0
        elif agent_label == "RIGHT_TERMINAL":
            agent_x = n * spacing
        else:
            agent_x = None
        if agent_x is not None:
            # Orange triangle marker above the node
            ax.scatter([agent_x], [node_y + 0.32], marker="v", s=120,
                       color=ORANGE, edgecolors=DARK_GRAY, linewidths=0.5, zorder=8)


def draw_pole(
    ax: plt.Axes,
    angle: float,
    action: int | float | None = None,
    pole_length: float = 0.8,
) -> None:
    """Draw a simple inverted pendulum (pole on a pivot).

    The pole is a line from a fixed pivot point, tilted by the
    current angle. The pivot is drawn as a small triangle base.
    Optionally shows the applied force direction.

    action can be:
      int: 0=left, 1=right (L02 discrete)
      float: continuous torque in [-1, 1] (L06 continuous)
    """
    # Pivot at bottom center
    pivot_x, pivot_y = 0.0, 0.0

    # Pole tip: angle=0 is straight up, positive = right tilt
    tip_x = pivot_x + pole_length * math.sin(angle)
    tip_y = pivot_y + pole_length * math.cos(angle)

    # Draw the pole
    ax.plot([pivot_x, tip_x], [pivot_y, tip_y],
            color=TEAL, linewidth=4, solid_capstyle="round", zorder=5)

    # Draw the pivot base (small triangle)
    base_w = 0.15
    ax.fill([pivot_x - base_w, pivot_x + base_w, pivot_x],
            [pivot_y - 0.05, pivot_y - 0.05, pivot_y + 0.02],
            color=DARK_GRAY, zorder=6)

    # Draw the tip as a circle
    ax.scatter([tip_x], [tip_y], c=TEAL, s=60, zorder=7,
               edgecolors=DARK_GRAY, linewidths=0.5)

    # Show force direction arrow if action is given
    if action is not None:
        arrow_y = -0.08
        if isinstance(action, float):
            # Continuous torque: arrow length proportional to magnitude
            arrow_dx = 0.3 * action  # scale to fit the viz
        else:
            # Discrete action: fixed-length arrow
            arrow_dx = 0.2 if action == 1 else -0.2
        if abs(arrow_dx) > 0.01:  # skip tiny arrows
            ax.annotate("", xy=(pivot_x + arrow_dx, arrow_y),
                         xytext=(pivot_x, arrow_y),
                         arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))

    # Ground line
    ax.plot([-0.5, 0.5], [-0.05, -0.05], color=LIGHT_GRAY, linewidth=1, zorder=1)

    # Danger zones (where the pole would fall)
    danger_angle = 0.3
    for sign in [1, -1]:
        dx = pole_length * math.sin(sign * danger_angle)
        dy = pole_length * math.cos(sign * danger_angle)
        ax.plot([pivot_x, pivot_x + dx], [pivot_y, pivot_y + dy],
                color="red", linewidth=1, linestyle="--", alpha=0.3, zorder=2)

    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.15, pole_length + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_breakout_frame(
    ax: plt.Axes,
    frame_rgb: list[list[list[float]]],
    score: int | None = None,
) -> None:
    """Display a Breakout color frame.

    frame_rgb: rows × cols × 3 RGB array (0.0-1.0 per channel).
    score: if set, displayed as white text at the top of the frame.
    """
    ax.clear()
    ax.imshow(frame_rgb, interpolation="nearest", aspect="equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if score is not None:
        cols = len(frame_rgb[0]) if frame_rgb else 0
        ax.text(cols - 1, 0, str(score),
                ha="center", va="center", fontsize=12,
                fontweight="bold", color="white",
                fontfamily="monospace")


def _pixel_to_rgb(val: float, imagined: bool = False) -> list[float]:
    """Map a grayscale pixel value to RGB for display.

    Real frames (imagined=False):
      0.0-0.15  → dark background
      0.15-0.80 → orange (target at 0.7)
      0.80-1.0  → teal (agent at 1.0)

    Imagined frames (imagined=True):
      Lighter background, lower thresholds so the decoder's soft
      outputs (~0.3-0.6) appear as visible colored dots. The decoder
      can't push sigmoid output past ~0.7, so the agent threshold
      is lowered to 0.55 on the imagined side.
    """
    # ORANGE = #E8915C (0.91, 0.57, 0.36), TEAL = #5CB8B2 (0.36, 0.72, 0.70)
    if imagined:
        bg = [0.20, 0.20, 0.28]
        lo = 0.10           # lower threshold—decoder outputs are faint
        orange_hi = 0.55    # decoder target outputs ~0.3-0.5
    else:
        bg = [0.10, 0.10, 0.18]
        lo = 0.15
        orange_hi = 0.80    # real target is exactly 0.7

    if val < lo:
        return list(bg)
    elif val < orange_hi:
        t = (val - lo) / (orange_hi - lo)
        return [bg[0] + (0.91 - bg[0]) * t,
                bg[1] + (0.57 - bg[1]) * t,
                bg[2] + (0.36 - bg[2]) * t]
    else:
        t = min((val - orange_hi) / (1.0 - orange_hi), 1.0)
        return [bg[0] + (0.36 - bg[0]) * t,
                bg[1] + (0.72 - bg[1]) * t,
                bg[2] + (0.70 - bg[2]) * t]


def _frame_to_rgb(frame: Matrix, imagined: bool = False) -> list[list[list[float]]]:
    """Convert a grayscale pixel frame to RGB using the project color scheme.

    For imagined frames, the decoder's outputs are too soft to distinguish
    agent from target by value alone (sigmoid squashes the agent from 1.0
    to ~0.35). Instead, we find the two brightest pixels and force-color
    the brightest as teal (agent) and the second as orange (target). This
    is a display-level boost—the model's actual outputs are unchanged.
    """
    if not imagined:
        return [[_pixel_to_rgb(val) for val in row] for row in frame]

    # Imagined side: find the two most active pixels
    rows = len(frame)
    cols = len(frame[0]) if frame else 0
    bg = [0.20, 0.20, 0.28]

    # Collect all pixel values with positions
    pixels = []
    for r in range(rows):
        for c in range(cols):
            pixels.append((frame[r][c], r, c))
    pixels.sort(key=lambda x: -x[0])

    # The brightest pixel is the target (decoder reconstructs it best ~0.7),
    # the second brightest is the agent (decoder reaches ~0.35).
    # Require meaningful activation above the background median to avoid
    # coloring noise in blank or low-contrast frames.
    bg_vals = sorted([frame[r][c] for r in range(rows) for c in range(cols)])
    bg_median = bg_vals[len(bg_vals) // 2] if bg_vals else 0.0
    min_margin = 0.10  # must be this much above background median
    target_pos = None
    agent_pos = None
    if pixels and pixels[0][0] > bg_median + min_margin:
        target_pos = (pixels[0][1], pixels[0][2])
        # Agent must also be above background and not the same cell as target
        if len(pixels) > 1 and pixels[1][0] > bg_median + min_margin:
            p1r, p1c = pixels[1][1], pixels[1][2]
            if (p1r, p1c) != target_pos:  # distinct cell (adjacent is fine)
                agent_pos = (p1r, p1c)

    rgb = [[list(bg) for _ in range(cols)] for _ in range(rows)]

    # Color background pixels based on value (subtle glow for active areas)
    for r in range(rows):
        for c in range(cols):
            val = frame[r][c]
            if val > 0.02:
                t = min(val / 0.5, 1.0)
                rgb[r][c] = [bg[0] + 0.10 * t, bg[1] + 0.10 * t, bg[2] + 0.08 * t]

    # Force-color the top two pixels
    if target_pos:
        r, c = target_pos
        rgb[r][c] = [0.91, 0.57, 0.36]  # ORANGE
    if agent_pos:
        r, c = agent_pos
        rgb[r][c] = [0.36, 0.72, 0.70]  # TEAL

    return rgb


def _add_pixel_grid(ax: plt.Axes, rows: int, cols: int) -> None:
    """Overlay a subtle grid so individual pixels are visible on dark backgrounds."""
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#333333", linewidth=0.3, zorder=2)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#333333", linewidth=0.3, zorder=2)


def draw_pixel_env(
    ax: plt.Axes,
    frame: Matrix,
) -> None:
    """Display a pixel-grid environment (e.g. 16×16) as a color image.

    frame: rows × cols matrix of floats (0.0 = empty, 0.7 = target, 1.0 = agent).
    Colors: agent = teal, target = orange, empty = dark background.
    """
    ax.clear()
    rows = len(frame)
    cols = len(frame[0]) if frame else 0
    rgb = _frame_to_rgb(frame)
    ax.imshow(rgb, interpolation="nearest", aspect="equal")
    _add_pixel_grid(ax, rows, cols)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_policy_gaussian(
    ax: plt.Axes,
    mean: float,
    std: float,
    action_range: tuple[float, float] = (-2.0, 2.0),
    num_points: int = 100,
) -> None:
    """Draw the agent's action distribution as a bell curve.

    Shows how likely each action value is. A tall narrow curve means
    the agent is confident; a short wide curve means uncertain.
    Used by L06 PPO to show policy smoothing over training.
    """
    ax.clear()
    lo, hi = action_range
    step = (hi - lo) / num_points
    xs = [lo + i * step for i in range(num_points + 1)]

    # Bell curve formula: (1 / (std × √(2π))) × exp(-0.5 × ((x - mean) / std)²)
    inv_norm = 1.0 / (std * math.sqrt(2.0 * math.pi))
    ys = []
    for x in xs:
        z = (x - mean) / std
        ys.append(inv_norm * math.exp(-0.5 * z * z))

    # fill_between shades the area under the curve
    ax.fill_between(xs, ys, alpha=0.3, color=TEAL)
    ax.plot(xs, ys, color=TEAL, linewidth=1.5)
    # axvline draws a vertical line marking the mean action
    ax.axvline(mean, color=ORANGE, linestyle="--", linewidth=1, label=f"mean={mean:.2f}")
    ax.set_xlabel("Action", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_real_vs_imagined(
    ax: plt.Axes,
    real_frame: Matrix,
    imagined_frame: Matrix,
) -> None:
    """Split-screen: real observation on the left, world-model prediction on the right.

    Both frames are combined into one image so they stay perfectly
    aligned, making frame-by-frame comparison immediate. A gray gap
    column separates them visually.

    Used by L07 Dreamer to show imagination tracking reality, then diverging.
    """
    ax.clear()

    rows_r = len(real_frame)
    cols_r = len(real_frame[0]) if real_frame else 0
    rows_i = len(imagined_frame)
    cols_i = len(imagined_frame[0]) if imagined_frame else 0

    # Convert both frames to RGB, combine with a gray gap column.
    # The imagined side uses lighter background and lower thresholds.
    rgb_r = _frame_to_rgb(real_frame, imagined=False)
    rgb_i = _frame_to_rgb(imagined_frame, imagined=True)

    gap = 1
    total_rows = max(rows_r, rows_i)
    combined_cols = cols_r + gap + cols_i
    gray_pixel = [0.4, 0.4, 0.4]
    combined = [[list(gray_pixel) for _ in range(combined_cols)]
                for _ in range(total_rows)]

    for r in range(rows_r):
        for c in range(cols_r):
            combined[r][c] = rgb_r[r][c]
    for r in range(rows_i):
        for c in range(cols_i):
            combined[r][cols_r + gap + c] = rgb_i[r][c]

    ax.imshow(combined, interpolation="nearest", aspect="equal")
    _add_pixel_grid(ax, total_rows, combined_cols)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(cols_r / 2, total_rows + 0.8, "Real", ha="center", fontsize=8, color=DARK_GRAY)
    ax.text(cols_r + gap + cols_i / 2, total_rows + 0.8, "Reconstructed", ha="center",
            fontsize=8, color=DARK_GRAY)
