"""Lesson 3: Temporal-Difference Learning (Sutton, 1988).

In 1988, Richard Sutton formalized the idea that drove Lesson 02's
critic: you can learn to predict without waiting for the final
outcome. This lesson isolates TD learning on a simple random walk
where the true values are known, so we can see exactly how well
each method predicts.

Run: uv run python lessons/03_td_learning.py
"""

import os
from dataclasses import dataclass

from policywerk.world.random_walk import RandomWalk
from policywerk.actors.td_learner import td_zero, td_lambda, monte_carlo, rms_error
from policywerk.primitives.progress import Spinner
from policywerk.viz.animate import (
    create_lesson_figure, FrameSnapshot, save_animation,
    save_poster, save_figure, TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.values import draw_value_bars
from policywerk.viz.traces import update_trace_axes
from policywerk.viz.trajectories import draw_chain

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Animation snapshot
# ---------------------------------------------------------------------------

@dataclass
class TDSnapshot(FrameSnapshot):
    estimated: list[float]
    rms: float
    episode_num: int
    path: list[str] | None = None
    outcome: str | None = None
    agent_label: str | None = None  # current position for step-by-step frames
    step_label: str | None = None   # title override for step frames


# ---------------------------------------------------------------------------
# The lesson
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("  Lesson 03: Temporal-Difference Learning (1988)")
    print("  Richard Sutton, 'Learning to Predict by the")
    print("  Methods of Temporal Differences'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. Connecting to Lesson 02
    # -----------------------------------------------------------------------

    print("ISOLATING THE TD IDEA")
    print("-" * 64)
    print("""
    In Lesson 02, the ACE critic updated its predictions using:

      td_error = reward + gamma * prediction(now) - prediction(before)

    This worked, but we never explained why. Why should updating a
    prediction toward the NEXT prediction (which might also be wrong)
    lead to correct values?

    This lesson isolates the TD idea on a problem simple enough to
    verify: a random walk with known true values. We can measure
    exactly how accurate the predictions are.
    """)

    # -----------------------------------------------------------------------
    # 2. The random walk
    # -----------------------------------------------------------------------

    print("THE RANDOM WALK")
    print("-" * 64)
    print("""
    Five states in a chain. The agent starts at C and walks left
    or right at random (50/50). The episode ends when it falls off
    either end.

      [0] <-- A -- B -- C -- D -- E --> [+1]
       |                                  |
       lose (reward 0)        win (reward +1)

    All intermediate steps give reward 0. The only signal comes at
    the very end: +1 if the agent reaches the right side, 0 if left.

    The true values -- the probability of reaching +1 from each
    state -- are known exactly:

      A: 1/6 = 0.167    (far from +1, likely to lose)
      B: 2/6 = 0.333
      C: 3/6 = 0.500    (center, 50/50 chance)
      D: 4/6 = 0.667
      E: 5/6 = 0.833    (close to +1, likely to win)

    This is the test: can the agent learn these values from
    experience alone, without knowing they follow this pattern?
    """)

    env = RandomWalk()

    # -----------------------------------------------------------------------
    # 3. TD(0) explained
    # -----------------------------------------------------------------------

    print("TD(0): BOOTSTRAPPING FROM THE NEXT STATE")
    print("-" * 64)
    print("""
    TD(0) updates V(s) after every single step:

      V(s) += alpha * [reward + gamma * V(s') - V(s)]

    The term in brackets is the TD error -- the same surprise signal
    from Lesson 02. "reward + gamma * V(s')" is what actually
    happened (reward) plus what the agent now expects (V of the next
    state). "V(s)" is what the agent predicted before the step. The
    difference is the surprise.

    Here is a concrete walkthrough. Suppose all values start at 0.5
    (we know nothing, so we guess 50/50). The agent walks:

      C -> D -> E -> [+1]  (won!)

    With alpha=0.1 and gamma=1.0:

      Step 1: C -> D, reward=0
        TD error = 0 + 1.0 * V(D) - V(C) = 0 + 0.5 - 0.5 = 0
        V(C) unchanged  (no surprise -- C and D had the same value)

      Step 2: D -> E, reward=0
        TD error = 0 + 1.0 * V(E) - V(D) = 0 + 0.5 - 0.5 = 0
        V(D) unchanged  (still no surprise)

      Step 3: E -> [+1], reward=1
        TD error = 1 + 1.0 * 0 - V(E) = 1 - 0.5 = 0.5
        V(E) += 0.1 * 0.5 = 0.05 -> V(E) = 0.55

    Only E updated! The +1 reward is one step away from E, so E
    learns first. On the NEXT episode, if the agent passes through
    D and reaches E, then D will see V(E)=0.55 and update toward
    it. Information propagates backward one step per episode --
    the same "ripple" as Lesson 01, but learned from experience
    instead of computed from a model.

    The key insight: TD(0) does not wait for the episode to end.
    It updates V(C) based on V(D), even though V(D) might be
    wrong. This is called bootstrapping -- updating a guess from
    a guess. The surprise is that this converges to the right
    answer.
    """)

    # -----------------------------------------------------------------------
    # 4. Monte Carlo for comparison
    # -----------------------------------------------------------------------

    print("MONTE CARLO: WAITING FOR THE TRUTH")
    print("-" * 64)
    print("""
    Monte Carlo (MC) takes the opposite approach: wait for the
    episode to end, then update each state toward the actual
    discounted return from its first visit.

      V(s) += alpha * [G_t - V(s)]

    For the walk C -> D -> E -> [+1] (gamma=1.0):
      G from C = 0 + 0 + 1 = 1
      G from D = 0 + 1 = 1
      G from E = 1
      V(C) += 0.1 * (1 - 0.5) = 0.05 -> V(C) = 0.55
      V(D) += 0.1 * (1 - 0.5) = 0.05 -> V(D) = 0.55
      V(E) += 0.1 * (1 - 0.5) = 0.05 -> V(E) = 0.55

    MC waits for the outcome, then updates the first visit of each
    state. TD(0) updates after every step without waiting. MC is
    unbiased (it uses the real outcome) but high-variance (the
    next episode from C might go C -> B -> A -> [0], giving G=0
    instead of G=1). TD(0) is biased (it trusts V(s'), which
    might be wrong) but lower variance (one step of randomness
    instead of a whole episode).
    """)

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------

    print("TRAINING: TD(0) vs MONTE CARLO")
    print("-" * 64)

    num_episodes = 100
    alpha = 0.1
    gamma = 1.0

    V_td, hist_td = td_zero(env, num_episodes=num_episodes, alpha=alpha,
                             gamma=gamma, seed=42)
    V_mc, hist_mc = monte_carlo(env, num_episodes=num_episodes, alpha=alpha,
                                 gamma=gamma, seed=42)

    print(f"    Training {num_episodes} episodes with alpha={alpha}, gamma={gamma}")
    print()

    # Show RMS error progression
    print("    RMS error over training (lower = better predictions):")
    for i in range(0, num_episodes, 20):
        td_rms = hist_td[i]["rms"]
        mc_rms = hist_mc[i]["rms"]
        td_bar = "#" * int(td_rms * 40) + "." * (40 - int(td_rms * 40))
        mc_bar = "#" * int(mc_rms * 40) + "." * (40 - int(mc_rms * 40))
        print(f"      Episode {i:3d}:  TD(0) {td_rms:.3f}  [{td_bar[:20]}]")
        print(f"                   MC    {mc_rms:.3f}  [{mc_bar[:20]}]")
    print()

    # Final values comparison
    print("    Final value estimates vs true values:")
    print(f"    {'State':>7s}  {'True':>6s}  {'TD(0)':>6s}  {'MC':>6s}")
    print(f"    {'-----':>7s}  {'----':>6s}  {'-----':>6s}  {'--':>6s}")
    for i, label in enumerate(RandomWalk.LABELS):
        true = RandomWalk.TRUE_VALUES[i]
        td_val = V_td.get(label)
        mc_val = V_mc.get(label)
        print(f"    {label:>7s}  {true:6.3f}  {td_val:6.3f}  {mc_val:6.3f}")
    print()
    print(f"    Final RMS:  TD(0) = {hist_td[-1]['rms']:.4f}   MC = {hist_mc[-1]['rms']:.4f}")
    print()

    print("""    Both methods converge toward the true values. TD(0) typically
    reaches low error faster because it reuses information: each
    step updates one state, and that updated state helps its
    neighbors on the next episode. MC must wait for complete
    episodes and is noisier because different episodes from the
    same state can have very different outcomes.
    """)

    # -----------------------------------------------------------------------
    # 6. TD(lambda) -- the spectrum
    # -----------------------------------------------------------------------

    print("TD(LAMBDA): THE SPECTRUM FROM TD TO MC")
    print("-" * 64)
    print("""
    TD(lambda) blends TD(0) and Monte Carlo using eligibility traces.
    The lambda parameter controls the blend:

      lambda = 0.0 : pure TD(0), update only the last state
      lambda = 0.3 : mostly TD, with some backward credit
      lambda = 0.7 : mostly MC, but with bootstrapping
      lambda = 1.0 : most MC-like (update all visited states via traces,
                     though not exactly identical to MC due to online
                     accumulating traces)

    Eligibility traces remember which states were recently visited.
    When the TD error arrives, all traced states get updated -- not
    just the most recent one. This is the same mechanism the ACE
    used in Lesson 02.
    """)

    lambdas = [0.0, 0.3, 0.7, 1.0]
    lambda_results = {}
    for lam in lambdas:
        V_l, hist_l = td_lambda(env, num_episodes=num_episodes, alpha=alpha,
                                 gamma=gamma, lam=lam, seed=42)
        lambda_results[lam] = (V_l, hist_l)

    print("    Final RMS error for different lambda values:")
    for lam in lambdas:
        _, hist = lambda_results[lam]
        final_rms = hist[-1]["rms"]
        bar = "#" * int(final_rms * 80) + "." * max(0, 20 - int(final_rms * 80))
        print(f"      lambda={lam:.1f}:  RMS = {final_rms:.4f}  [{bar[:20]}]")
    print()

    # Describe what the results actually show
    best_lam = min(lambdas, key=lambda l: lambda_results[l][1][-1]["rms"])
    print(f"""    On this run, lambda={best_lam} gave the lowest final error.
    The best lambda depends on the problem, the learning rate,
    and the number of episodes. On larger problems, intermediate
    lambda values (0.3-0.7) often outperform the extremes because
    they combine TD's stability with MC's directness.

    Note: Sutton's 1988 paper found that lambda around 0.3 was
    optimal on this same random walk, but using "offline" batch
    updates (accumulate changes, apply at episode end, repeat
    over training sets). Our implementation uses online updates
    (apply after each step), which is the standard practical
    variant and what Lesson 02's critic already did. The online
    version favors lower lambda on small problems because each
    step immediately improves one value.
    """)

    # -----------------------------------------------------------------------
    # 7. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    # Build snapshots: step-by-step for first 3 episodes, then summaries
    snapshots = []
    STEP_BY_STEP_EPISODES = 3

    for ep_idx in range(min(STEP_BY_STEP_EPISODES, num_episodes)):
        h = hist_td[ep_idx]
        ep_path = h.get("path", [])
        ep_outcome = h.get("outcome")
        ep_steps = h.get("steps", [])

        # Step-by-step: each frame shows values/rms AFTER the corresponding
        # TD update. step_snapshots[0] = pre-episode, [1] = after first
        # transition, [2] = after second, etc.

        # First frame: agent at starting position (pre-episode values)
        step_data = ep_steps[0] if ep_steps else {"values": h["pre_values"], "rms": h["pre_rms"]}
        snapshots.append(TDSnapshot(
            episode=ep_idx,
            total_reward=0.0,
            estimated=step_data["values"],
            rms=step_data["rms"],
            episode_num=ep_idx,
            path=[ep_path[0]] if ep_path else [],
            outcome=None,
            agent_label=ep_path[0] if ep_path else None,
            step_label=f"Episode {ep_idx}, start at {ep_path[0] if ep_path else '?'}",
        ))
        # Subsequent steps: agent moves, values update after each transition
        for step_idx in range(1, len(ep_path)):
            pos = ep_path[step_idx]
            # Use per-step snapshot (after the transition that moved to this state)
            step_data = ep_steps[step_idx] if step_idx < len(ep_steps) else ep_steps[-1]
            snapshots.append(TDSnapshot(
                episode=ep_idx,
                total_reward=0.0,
                estimated=step_data["values"],
                rms=step_data["rms"],
                episode_num=ep_idx,
                path=ep_path[:step_idx + 1],
                outcome=None,
                agent_label=pos,
                step_label=f"Episode {ep_idx}, step {step_idx}",
            ))
        # Final frame: outcome with post-episode values
        step_data = ep_steps[-1] if ep_steps else {"values": h["values"], "rms": h["rms"]}
        snapshots.append(TDSnapshot(
            episode=ep_idx,
            total_reward=0.0,
            estimated=step_data["values"],
            rms=step_data["rms"],
            episode_num=ep_idx,
            path=ep_path,
            outcome=ep_outcome,
            agent_label=None,
            step_label=f"Episode {ep_idx} -> {'won' if ep_outcome == 'right' else 'lost'}",
        ))

    # Remaining episodes: one summary frame per sampled episode
    record_interval = max(1, (num_episodes - STEP_BY_STEP_EPISODES) // 20)
    for ep_idx in range(STEP_BY_STEP_EPISODES, num_episodes):
        if (ep_idx - STEP_BY_STEP_EPISODES) % record_interval == 0 or ep_idx == num_episodes - 1:
            h = hist_td[ep_idx]
            ep_path = h.get("path", [])
            ep_outcome = h.get("outcome")
            # Show agent at last position before terminal
            last_pos = ep_path[-1] if ep_path else None
            result = f"-> {'won' if ep_outcome == 'right' else 'lost'}" if ep_outcome else ""
            snapshots.append(TDSnapshot(
                episode=ep_idx,
                total_reward=0.0,
                estimated=h["values"],
                rms=h["rms"],
                episode_num=ep_idx,
                path=ep_path,
                outcome=ep_outcome,
                agent_label=last_pos,
                step_label=f"Episode {ep_idx} {result}",
            ))

    true_values = list(RandomWalk.TRUE_VALUES)
    labels = list(RandomWalk.LABELS)

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 03: TD(0) Learning",
        subtitle="Sutton (1988) -- estimates move toward true values",
    )

    real_rms = [h["rms"] for h in hist_td]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: chain diagram with agent position
        draw_chain(axes["env"], labels, values=snap.estimated,
                   path=snap.path, outcome=snap.outcome,
                   agent_label=snap.agent_label,
                   caption="Random walk: start at C, move left/right at random until terminal")
        title = snap.step_label if snap.step_label else f"Episode {snap.episode_num}"
        axes["env"].set_title(title, fontsize=10)

        # Top-right: bar chart of estimated vs true values
        draw_value_bars(axes["algo"], snap.estimated, true_values, labels)
        axes["algo"].set_ylim(0, 1.0)
        axes["algo"].set_title(f"Estimates vs True (RMS={snap.rms:.3f})", fontsize=10)

        # Bottom: RMS error trace — only show completed episodes
        # (don't advance the trace during mid-episode step frames)
        n = min(snap.episode_num, len(real_rms))
        if snap.outcome is not None:
            # This is an outcome frame — include this episode's RMS
            n = min(snap.episode_num + 1, len(real_rms))
        update_trace_axes(axes["trace"], real_rms[:n],
                          label="RMS error", color=TEAL)
        axes["trace"].set_ylabel("RMS Error", fontsize=9)
        axes["trace"].set_xlabel("Episode", fontsize=9)
        axes["trace"].set_title("Prediction Accuracy", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/03_td_learning_artifact.gif", fps=3)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 03: TD Learning (Converged)",
        subtitle="Sutton (1988)",
    )

    def update_poster(frame_idx):
        # Chain with final learned values
        draw_chain(axes2["env"], labels, values=hist_td[-1]["values"],
                   caption="Random walk: start at C, move left/right at random until terminal")
        axes2["env"].set_title("Learned Values (TD(0))", fontsize=10)

        # Bar chart comparison
        draw_value_bars(axes2["algo"], hist_td[-1]["values"], true_values, labels)
        axes2["algo"].set_ylim(0, 1.0)
        axes2["algo"].set_title(f"Estimates vs True (RMS={hist_td[-1]['rms']:.3f})", fontsize=10)

        # Show both TD and MC RMS curves
        axes2["trace"].plot(range(num_episodes), [h["rms"] for h in hist_td],
                            color=TEAL, linewidth=1.0, alpha=0.8, label="TD(0)")
        axes2["trace"].plot(range(num_episodes), [h["rms"] for h in hist_mc],
                            color=ORANGE, linewidth=1.0, alpha=0.8, label="MC")
        axes2["trace"].set_ylabel("RMS Error", fontsize=9)
        axes2["trace"].set_xlabel("Episode", fontsize=9)
        axes2["trace"].set_title("TD(0) vs Monte Carlo", fontsize=10)
        axes2["trace"].legend(fontsize=8)
        axes2["trace"].grid(True, alpha=0.3)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/03_td_learning_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Comparison trace ---

    fig3, ax3 = plt.subplots(figsize=(10, 3), dpi=150)
    ax3.plot(range(num_episodes), [h["rms"] for h in hist_td],
             color=TEAL, linewidth=1.0, alpha=0.8, label="TD(0)")
    ax3.plot(range(num_episodes), [h["rms"] for h in hist_mc],
             color=ORANGE, linewidth=1.0, alpha=0.8, label="Monte Carlo")
    for lam in [0.3, 0.7]:
        _, hist_l = lambda_results[lam]
        ax3.plot(range(num_episodes), [h["rms"] for h in hist_l],
                 linewidth=0.8, alpha=0.6, label=f"TD(lambda={lam})")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("RMS Error")
    ax3.set_title("Prediction Error: TD vs Monte Carlo vs TD(lambda)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()

    with Spinner("Generating trace"):
        save_figure(fig3, "output/03_td_learning_trace.png")

    print()
    print("    Artifacts saved to output/:")
    print("      03_td_learning_artifact.gif  value estimates converging")
    print("      03_td_learning_artifact.pdf  PDF storyboard of every frame")
    print("      03_td_learning_poster.png    final estimates vs true values")
    print("      03_td_learning_trace.png     TD vs MC vs TD(lambda) comparison")

    # -----------------------------------------------------------------------
    # Closing
    # -----------------------------------------------------------------------

    print()
    print("=" * 64)
    print("  Lesson 03 complete.")
    print()
    print("  TD learning predicts values by bootstrapping from the next")
    print("  state's estimate. But prediction is only half the problem.")
    print("  In Lesson 04, Watkins extends TD to CONTROL: learning which")
    print("  actions to take, not just how good states are. That is")
    print("  Q-learning.")
    print("=" * 64)


if __name__ == "__main__":
    main()
