"""Lesson 4: Q-Learning (Watkins, 1989).

In 1989, Chris Watkins showed how an agent could learn the optimal
policy while exploring—without needing a model of the environment
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
    policy: dict[str, int] | None
    episode_num: int
    ep_reward: float
    method: str
    agent_pos: tuple[int, int] | None = None
    step_label: str | None = None
    completed: bool = True  # False for mid-episode step-by-step frames


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
    Lesson 03 taught the agent to predict how good each state is.
    But prediction alone does not tell the agent what to do.
    Knowing that state D is worth 0.667 does not say whether to
    go left or right.

    Q-learning solves this by learning action values. Instead of
    V(s) -- "how good is this state?" -- it learns Q(s, a) --
    "how good is it to take action a in state s?" Once you have
    a Q-value for every action, you can pick the one with the
    highest score.

    The second idea is off-policy learning. The agent needs to
    try random actions to discover what works. But it also needs
    to learn the best possible behavior, not just the behavior
    it happens to be trying. Q-learning does both at once. It
    tries random actions during training, but the update rule
    always assumes the best action will be taken next. This
    means the agent can learn the best strategy while still
    exploring.
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

    The optimal path runs just above the cliff (row 2), then drops
    to the goal: 13 steps, total reward -12 (12 steps at -1 each,
    plus the goal step at 0). But with epsilon=0.1 exploration
    (epsilon is the exploration rate—the probability that the agent
    picks a random action instead of its current best; with
    epsilon=0.1, one in ten actions is random), the agent
    occasionally steps south into the cliff. This is where
    Q-learning and SARSA disagree about what to learn.
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

    Result: Q-learning learns the cliff-edge path—optimal if the
    agent follows the greedy policy perfectly. SARSA learns a safer
    path one row up—optimal given that the agent is still exploring
    with epsilon=0.1. Both are correct for the question they answer.
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

    # Show early episode volatility
    early = [(h["episode"], h["total_reward"], h["steps"]) for h in hist_ql[:10]]
    print("    Early episodes (Q-learning):")
    for ep, r, s in early:
        r_str = f"{r:.0f}" if r > -1000 else "< -1000"
        print(f"      Episode {ep:2d}:  reward {r_str:>8s}   steps {s}")
    print()

    print("""    The early episodes are wild. The agent falls off the cliff
    repeatedly, getting teleported back to start, racking up huge
    penalties. By episode 3 it has mostly learned to avoid the
    cliff, but occasional relapses show that epsilon exploration
    near the edge is genuinely dangerous even after the agent
    knows what to do.

    Notice that Q-learning's average training reward stays worse
    than SARSA's even after hundreds of episodes. This is the
    on-policy/off-policy trade-off in a nutshell: the better the
    learned policy, the worse the training performance when
    exploration adds noise. Q-learning's cliff-edge path is
    optimal when executed perfectly, but maximally punished by
    random exploration. SARSA's safer path is suboptimal but
    more forgiving of mistakes.
    """)

    # Extract greedy policies: pick the action with the highest
    # Q-value in each state, converting the Q-table into a
    # deterministic strategy.
    policy_ql = extract_greedy_policy(Q_ql, env)
    policy_sa = extract_greedy_policy(Q_sa, env)

    # Run greedy evaluation episodes (no exploration) using the
    # extracted policies to show the learned behavior cleanly.
    from policywerk.actors.q_learner import _label_to_pos, eval_greedy
    ql_eval_path, ql_eval_reward, ql_eval_done = eval_greedy(policy_ql, CliffWorld())
    sa_eval_path, sa_eval_reward, sa_eval_done = eval_greedy(policy_sa, CliffWorld())

    ql_avg_row = sum(r for r, c in ql_eval_path) / len(ql_eval_path)
    sa_avg_row = sum(r for r, c in sa_eval_path) / len(sa_eval_path)

    print(f"    Greedy evaluation (no exploration):")
    print(f"      Q-learning: {len(ql_eval_path)-1} steps, reward {ql_eval_reward:.0f}, avg row {ql_avg_row:.1f}")
    print(f"      SARSA:      {len(sa_eval_path)-1} steps, reward {sa_eval_reward:.0f}, avg row {sa_avg_row:.1f}")
    print()
    print("""    Q-learning's greedy policy runs just above the cliff (row 2),
    the shortest safe route. SARSA's greedy policy may take a wider
    path because it learned to account for exploratory mistakes
    during training.

    Both are "correct" in different senses. Q-learning found the
    optimal path. SARSA found the path that works best when the
    agent is still exploring with epsilon=0.1.
    """)

    # -----------------------------------------------------------------------
    # 6. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots: show policy arrows evolving over training.
    # Each frame shows the current greedy policy as arrows in the grid,
    # so the viewer watches the arrows settle into a route.
    snapshots = []

    sample_episodes = sorted(set(
        list(range(0, min(10, num_episodes))) +
        list(range(10, num_episodes, max(1, num_episodes // 20))) +
        [num_episodes - 1]
    ))

    for ep_idx in sample_episodes:
        if ep_idx < num_episodes:
            h = hist_ql[ep_idx]
            ep_path = h["path"]
            last_pos = ep_path[-1] if ep_path else None
            reward_str = f"{h['total_reward']:.0f}" if h['total_reward'] > -1000 else "< -1000"
            snapshots.append(QLearningSnapshot(
                episode=ep_idx,
                total_reward=h["total_reward"],
                path=ep_path,
                policy=h.get("policy"),  # evolving policy arrows
                episode_num=ep_idx,
                ep_reward=h["total_reward"],
                method="Q-learning",
                agent_pos=last_pos,  # blue cell at agent's final position
                step_label=f"Episode {ep_idx} (reward: {reward_str})",
            ))

    cliff_cells = list(CliffWorld.CLIFF)

    # Append greedy replay frames: step through the learned optimal path,
    # looped 4 times so it lingers in the animation.
    final_policy = hist_ql[-1].get("policy")
    for _loop in range(4):
        for step_idx, pos in enumerate(ql_eval_path):
            snapshots.append(QLearningSnapshot(
                episode=num_episodes,
                total_reward=ql_eval_reward,
                path=ql_eval_path[:step_idx + 1],
                policy=final_policy,
                episode_num=num_episodes - 1,
                ep_reward=ql_eval_reward,
                method="Q-learning",
                agent_pos=pos,
                step_label=f"Learned policy (step {step_idx}/{len(ql_eval_path)-1})",
            ))

    # Append SARSA greedy replay frames: same idea, different path.
    # Use policy_sa directly (the extracted greedy policy).
    for _loop in range(4):
        for step_idx, pos in enumerate(sa_eval_path):
            snapshots.append(QLearningSnapshot(
                episode=num_episodes + 1,
                total_reward=sa_eval_reward,
                path=sa_eval_path[:step_idx + 1],
                policy=policy_sa,
                episode_num=num_episodes,
                ep_reward=sa_eval_reward,
                method="SARSA",
                agent_pos=pos,
                step_label=f"SARSA policy (step {step_idx}/{len(sa_eval_path)-1})",
            ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 04: Q-Learning",
        subtitle="Watkins (1989) -- policy arrows settle into a route",
        figsize=(14, 7),  # wider for the 4x12 grid
    )

    ql_rewards = [h["total_reward"] for h in hist_ql]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Compute cell values from the Q-value snapshot for this episode.
        # Use the correct history for the method being displayed.
        is_replay = snap.episode >= num_episodes
        if is_replay and snap.method == "SARSA":
            ep_hist = hist_sa[-1]
        else:
            ep_hist = hist_ql[min(snap.episode_num, len(hist_ql) - 1)]
        vals_dict = ep_hist.get("values", {})
        cv = {}
        for label, val in vals_dict.items():
            parts = label.split(",")
            cv[(int(parts[0]), int(parts[1]))] = val

        # Top-left: cliff grid with value-gradient coloring and policy arrows
        draw_cliff_grid(axes["env"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        policy=snap.policy,
                        caption="Cliff world: green = high value, red = cliff",
                        agent_pos=snap.agent_pos, cell_values=cv)
        title = snap.step_label if snap.step_label else f"Episode {snap.episode_num}"
        axes["env"].set_title(title, fontsize=10)

        # Top-right: info
        axes["algo"].clear()
        axes["algo"].axis("off")
        reward_str = f"{snap.ep_reward:.0f}" if snap.ep_reward > -1000 else "< -1000"
        if is_replay:
            eval_path = ql_eval_path if snap.method == "Q-learning" else sa_eval_path
            eval_reward = ql_eval_reward if snap.method == "Q-learning" else sa_eval_reward
            info_lines = [
                f"Greedy replay: {snap.method}",
                f"(no exploration)",
                "",
                f"Reward:  {eval_reward:.0f}",
                f"Steps:   {len(eval_path) - 1}",
                "",
                "Following learned policy",
            ]
        else:
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
        axes["algo"].set_title("Greedy Replay" if is_replay else "Training State",
                               fontsize=10)

        # Bottom: reward trace
        n = min(snap.episode_num + 1, len(ql_rewards))
        update_trace_axes(axes["trace"], ql_rewards[:n],
                          label="Episode reward", color=TEAL)
        axes["trace"].set_ylabel("Reward", fontsize=9)
        axes["trace"].set_xlabel("Episode", fontsize=9)
        axes["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/04_q_learning_artifact.gif", fps=2)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 04: Q-Learning vs SARSA",
        subtitle="Watkins (1989)",
        figsize=(14, 7),
    )

    def update_poster(frame_idx):
        # Show Q-learning's greedy eval path with policy
        # Q-learning final values as cell colors
        ql_final_vals = hist_ql[-1].get("values", {})
        ql_cv = {(int(l.split(",")[0]), int(l.split(",")[1])): v
                 for l, v in ql_final_vals.items()}
        draw_cliff_grid(axes2["env"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        policy=policy_ql, cell_values=ql_cv,
                        caption="Q-learning: green = high value")
        axes2["env"].set_title("Q-Learning Policy", fontsize=10)

        # SARSA final values as cell colors
        sa_final_vals = hist_sa[-1].get("values", {})
        sa_cv = {(int(l.split(",")[0]), int(l.split(",")[1])): v
                 for l, v in sa_final_vals.items()}
        draw_cliff_grid(axes2["algo"], CliffWorld.ROWS, CliffWorld.COLS,
                        cliff_cells, CliffWorld.START, CliffWorld.GOAL,
                        policy=policy_sa, cell_values=sa_cv,
                        caption="SARSA: green = high value")
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
            ql_smooth = [sum(ql_rewards[max(0,i-w+1):i+1])/len(ql_rewards[max(0,i-w+1):i+1]) for i in range(num_episodes)]
            sa_smooth = [sum(sa_rewards[max(0,i-w+1):i+1])/len(sa_rewards[max(0,i-w+1):i+1]) for i in range(num_episodes)]
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
        ql_smooth = [sum(ql_rewards[max(0,i-w+1):i+1])/len(ql_rewards[max(0,i-w+1):i+1]) for i in range(num_episodes)]
        sa_smooth = [sum(sa_rewards[max(0,i-w+1):i+1])/len(sa_rewards[max(0,i-w+1):i+1]) for i in range(num_episodes)]
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
