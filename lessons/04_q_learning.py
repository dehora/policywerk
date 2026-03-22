"""Lesson 4: Q-Learning (Watkins, 1989).

In 1989, Chris Watkins showed how an agent could learn the optimal
policy while exploring — without needing a model of the environment
and without being constrained to follow its current best guess.
This is Q-learning, and it extends TD learning from prediction
(Lesson 03) to control.

Run: uv run python lessons/04_q_learning.py
"""

import os
from dataclasses import dataclass

from policywerk.world.cliffworld import CliffWorld
from policywerk.actors.q_learner import q_learning, sarsa, extract_greedy_policy
from policywerk.primitives.progress import Spinner
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.trajectories import draw_cliff_grid
from policywerk.viz.traces import update_trace_axes

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot
# ---------------------------------------------------------------------------

@dataclass
class QLearningSnapshot(FrameSnapshot):
    path: list[tuple[int, int]]
    policy: dict[str, int]
    episode_num: int
    ep_reward: float
    method: str


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Lesson 04: Q-Learning (1989)")
    print("  Chris Watkins, 'Learning from Delayed Rewards'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. From prediction to control
    # -----------------------------------------------------------------------

    print("FROM PREDICTION TO CONTROL")
    print("-" * 64)
    print("""
    Lesson 03 taught the agent to PREDICT -- learning how good
    each state is. But prediction alone does not tell the agent
    what to DO. Knowing that state D is worth 0.667 does not say
    whether to go left or right.

    Q-learning solves this by learning action values: Q(s, a) is
    "how good is it to take action a in state s?" Once you know
    Q(s, a) for every action, the decision is trivial: pick the
    action with the highest Q-value.

    The key innovation is off-policy learning. The agent explores
    randomly (epsilon-greedy) but learns the OPTIMAL policy --
    as if it always picked the best action. This is possible
    because the update rule uses "max" over the next state's
    actions, not the action actually taken.
    """)

    # -----------------------------------------------------------------------
    # 2. The cliff world
    # -----------------------------------------------------------------------

    print("THE CLIFF WORLD")
    print("-" * 64)
    print("""
    A 4x12 grid. Start at the bottom-left, goal at the bottom-right.
    The cliff runs along the bottom edge between start and goal.

      . . . . . . . . . . . .
      . . . . . . . . . . . .
      . . . . . . . . . . . .
      S C C C C C C C C C C G    S=start, C=cliff(-100), G=goal

    Actions: N, E, S, W (0, 1, 2, 3)
    Normal steps cost -1. Stepping on the cliff costs -100 and
    teleports back to start (the episode does not end -- the agent
    must try again). Reaching the goal ends the episode with
    reward 0.

    The optimal path is along the cliff edge (bottom row, 13 steps,
    total reward -13). But with epsilon=0.1 exploration, the agent
    occasionally steps south into the cliff. This is where Q-learning
    and SARSA disagree about what to learn.
    """)

    env = CliffWorld()

    # -----------------------------------------------------------------------
    # 3. Q-learning explained
    # -----------------------------------------------------------------------

    print("Q-LEARNING: THE OFF-POLICY UPDATE")
    print("-" * 64)
    print("""
    The Q-learning update after each step (s, a, r, s'):

      Q(s, a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    This is the TD error from Lesson 03, but applied to action
    values. The crucial word is "max" -- the agent asks "what is
    the best I COULD do from s'?" regardless of what it actually
    does next.

    Concrete example. The agent is at (2, 1), just above the cliff.
    It goes East to (2, 2), getting reward -1.

      Q((2,1), East) += 0.5 * [-1 + 1.0 * max_a Q((2,2), a) - Q((2,1), East)]

    If max_a Q((2,2), a) = -10 and Q((2,1), East) = -12:
      TD error = -1 + (-10) - (-12) = 1
      Q((2,1), East) += 0.5 * 1 = +0.5 -> Q = -11.5

    The value went up because -10 is better than the old estimate
    of -12. The "max" means Q-learning learns what the optimal
    policy would do, even while the agent explores with epsilon-
    greedy. This is off-policy learning.
    """)

    # -----------------------------------------------------------------------
    # 4. SARSA for comparison
    # -----------------------------------------------------------------------

    print("SARSA: THE ON-POLICY ALTERNATIVE")
    print("-" * 64)
    print("""
    SARSA uses the action ACTUALLY taken next, not the best:

      Q(s, a) += alpha * [r + gamma * Q(s', a_next) - Q(s, a)]

    The name SARSA comes from the five values used in each update:
    State, Action, Reward, State', Action'.

    On the cliff world, this matters. With epsilon=0.1, the agent
    sometimes takes a random action. Near the cliff edge, a random
    action might step south into the cliff (-100 penalty). SARSA
    accounts for this risk because it uses the action actually
    taken (which might be random). Q-learning ignores the risk
    because it uses the best action (which would never step into
    the cliff).

    Result: Q-learning learns the cliff-edge path (optimal but
    risky during training). SARSA learns a safer path one row up
    (suboptimal but avoids the cliff penalty during exploration).
    """)

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------

    print("TRAINING: Q-LEARNING vs SARSA")
    print("-" * 64)

    num_episodes = 500
    alpha = 0.5
    gamma = 1.0
    epsilon = 0.1

    Q_ql, hist_ql = q_learning(env, num_episodes=num_episodes, alpha=alpha,
                                gamma=gamma, epsilon=epsilon, seed=42)
    Q_sa, hist_sa = sarsa(env, num_episodes=num_episodes, alpha=alpha,
                           gamma=gamma, epsilon=epsilon, seed=42)

    print(f"    Training {num_episodes} episodes with alpha={alpha}, epsilon={epsilon}")
    print()

    # Show reward progression
    window = 50
    print(f"    Average reward per {window} episodes:")
    for i in range(0, num_episodes, window):
        chunk_ql = [h["total_reward"] for h in hist_ql[i:i + window]]
        chunk_sa = [h["total_reward"] for h in hist_sa[i:i + window]]
        avg_ql = sum(chunk_ql) / len(chunk_ql)
        avg_sa = sum(chunk_sa) / len(chunk_sa)
        print(f"      Episodes {i:3d}-{i+window-1:3d}:  Q-learning {avg_ql:7.1f}   SARSA {avg_sa:7.1f}")
    print()

    # Extract and show policies
    policy_ql = extract_greedy_policy(Q_ql, env)
    policy_sa = extract_greedy_policy(Q_sa, env)

    # Show the key difference: which row does each policy use?
    ql_last_path = hist_ql[-1]["path"]
    sa_last_path = hist_sa[-1]["path"]
    ql_avg_row = sum(r for r, c in ql_last_path) / len(ql_last_path)
    sa_avg_row = sum(r for r, c in sa_last_path) / len(sa_last_path)

    print(f"    Q-learning final path: {len(ql_last_path)} steps, avg row {ql_avg_row:.1f}")
    print(f"    SARSA final path:      {len(sa_last_path)} steps, avg row {sa_avg_row:.1f}")
    print()
    print("""    Q-learning's path hugs the bottom row (row 3, the cliff edge)
    because it learned the optimal policy. SARSA's path stays higher
    (row 2) because it learned to avoid the cliff during exploration.

    Both are "correct" in different senses. Q-learning found the
    shortest path. SARSA found the path that actually works best
    when you are still exploring with epsilon=0.1.
    """)

    # -----------------------------------------------------------------------
    # 6. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots — sample Q-learning episodes
    snapshots = []
    sample_episodes = sorted(set(
        list(range(0, min(10, num_episodes))) +
        list(range(10, num_episodes, num_episodes // 20)) +
        [num_episodes - 1]
    ))

    for ep_idx in sample_episodes:
        if ep_idx < num_episodes:
            h = hist_ql[ep_idx]
            pol = extract_greedy_policy(Q_ql, env) if ep_idx == num_episodes - 1 else None
            snapshots.append(QLearningSnapshot(
                episode=ep_idx,
                total_reward=h["total_reward"],
                path=h["path"],
                policy=pol,
                episode_num=ep_idx,
                ep_reward=h["total_reward"],
                method="Q-learning",
            ))

    cliff_cells = list(CliffWorld.CLIFF)

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 04: Q-Learning",
        subtitle="Watkins (1989) -- policy arrows settle into a route",
    )

    ql_rewards = [h["total_reward"] for h in hist_ql]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: cliff grid with path
        draw_cliff_grid(axes["env"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        path=snap.path, policy=snap.policy,
                        caption="Cliff world: reach G from S, avoid C (-100)")
        reward_str = f"{snap.ep_reward:.0f}" if snap.ep_reward > -1000 else "< -1000"
        axes["env"].set_title(f"Episode {snap.episode_num} (reward: {reward_str})", fontsize=10)

        # Top-right: info
        axes["algo"].clear()
        axes["algo"].axis("off")
        info_lines = [
            f"Episode: {snap.episode_num}",
            f"Reward:  {reward_str}",
            f"Steps:   {len(snap.path) - 1}",
            "",
            f"Method: {snap.method}",
            f"alpha:   {alpha}",
            f"epsilon: {epsilon}",
        ]
        if snap.policy:
            info_lines.append("")
            info_lines.append(">> Policy arrows shown")
        text = "\n".join(info_lines)
        axes["algo"].text(0.1, 0.9, text, transform=axes["algo"].transAxes,
                          fontsize=10, verticalalignment="top",
                          fontfamily="monospace", color=DARK_GRAY)
        axes["algo"].set_title("Training State", fontsize=10)

        # Bottom: reward trace
        n = min(snap.episode_num + 1, len(ql_rewards))
        update_trace_axes(axes["trace"], ql_rewards[:n],
                          label="Episode reward", color=TEAL)
        axes["trace"].set_ylabel("Reward", fontsize=9)
        axes["trace"].set_xlabel("Episode", fontsize=9)
        axes["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/04_q_learning_artifact.gif", fps=3)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 04: Q-Learning vs SARSA",
        subtitle="Watkins (1989)",
    )

    def update_poster(frame_idx):
        # Show Q-learning's final path with policy
        draw_cliff_grid(axes2["env"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        path=ql_last_path, policy=policy_ql,
                        caption="Q-learning: optimal cliff-edge path")
        axes2["env"].set_title("Q-Learning Policy", fontsize=10)

        # Show SARSA's final path with policy
        draw_cliff_grid(axes2["algo"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        path=sa_last_path, policy=policy_sa,
                        caption="SARSA: safer path avoiding cliff edge")
        axes2["algo"].set_title("SARSA Policy", fontsize=10)

        # Reward comparison
        axes2["trace"].plot(range(num_episodes), ql_rewards,
                            color=TEAL, linewidth=0.5, alpha=0.3)
        sa_rewards = [h["total_reward"] for h in hist_sa]
        axes2["trace"].plot(range(num_episodes), sa_rewards,
                            color=ORANGE, linewidth=0.5, alpha=0.3)
        # Smoothed
        w = 20
        if num_episodes > w:
            ql_smooth = [sum(ql_rewards[max(0,i-w):i+1])/min(i+1,w) for i in range(num_episodes)]
            sa_smooth = [sum(sa_rewards[max(0,i-w):i+1])/min(i+1,w) for i in range(num_episodes)]
            axes2["trace"].plot(range(num_episodes), ql_smooth,
                                color=TEAL, linewidth=1.5, label="Q-learning")
            axes2["trace"].plot(range(num_episodes), sa_smooth,
                                color=ORANGE, linewidth=1.5, label="SARSA")
        axes2["trace"].set_ylabel("Reward", fontsize=9)
        axes2["trace"].set_xlabel("Episode", fontsize=9)
        axes2["trace"].set_title("Q-Learning vs SARSA", fontsize=10)
        axes2["trace"].legend(fontsize=8)
        axes2["trace"].grid(True, alpha=0.3)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/04_q_learning_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Comparison trace ---

    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    sa_rewards = [h["total_reward"] for h in hist_sa]
    ax3.plot(range(num_episodes), ql_rewards,
             color=TEAL, linewidth=0.3, alpha=0.3)
    ax3.plot(range(num_episodes), sa_rewards,
             color=ORANGE, linewidth=0.3, alpha=0.3)
    w = 20
    if num_episodes > w:
        ql_smooth = [sum(ql_rewards[max(0,i-w):i+1])/min(i+1,w) for i in range(num_episodes)]
        sa_smooth = [sum(sa_rewards[max(0,i-w):i+1])/min(i+1,w) for i in range(num_episodes)]
        ax3.plot(range(num_episodes), ql_smooth,
                 color=TEAL, linewidth=1.5, label="Q-learning (smoothed)")
        ax3.plot(range(num_episodes), sa_smooth,
                 color=ORANGE, linewidth=1.5, label="SARSA (smoothed)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Reward")
    ax3.set_title("Q-Learning vs SARSA: Episode Reward")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    with Spinner("Generating trace"):
        save_figure(fig3, "output/04_q_learning_trace.png")

    print()
    print("    Artifacts saved to output/:")
    print("      04_q_learning_artifact.gif  Q-learning trajectories over training")
    print("      04_q_learning_artifact.pdf  PDF storyboard of every frame")
    print("      04_q_learning_poster.png    Q-learning vs SARSA side by side")
    print("      04_q_learning_trace.png     reward comparison over training")

    # -----------------------------------------------------------------------
    # Closing
    # -----------------------------------------------------------------------

    print()
    print("=" * 64)
    print("  Lesson 04 complete.")
    print()
    print("  Q-learning uses tables: one entry per state-action pair.")
    print("  But what if the state space is too large to tabulate --")
    print("  like pixel observations? In Lesson 05, DQN replaces the")
    print("  table with a neural network: Q-learning at scale.")
    print("=" * 64)


if __name__ == "__main__":
    main()
