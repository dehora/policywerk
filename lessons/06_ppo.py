"""Lesson 06: Proximal Policy Optimization (PPO).

Schulman et al. (2017), 'Proximal Policy Optimization Algorithms.'

Learns the policy directly instead of deriving it from Q-values.
The actor network outputs a probability distribution over actions,
and the agent samples from it. A clipped surrogate objective
prevents catastrophically large updates.

    uv run python lessons/06_ppo.py

Artifacts:
  output/06_ppo_artifact.gif   Training animation
  output/06_ppo_poster.png     Trained agent snapshot
  output/06_ppo_trace.png      Reward and entropy curves
"""

import os
import math
from dataclasses import dataclass

from policywerk.actors.ppo import ppo, balance_outcome
from policywerk.building_blocks.network import network_forward
from policywerk.building_blocks.distributions import Gaussian
from policywerk.world.balance import Balance
from policywerk.viz.animate import (
    FrameSnapshot, create_lesson_figure, save_animation, save_poster,
    TEAL, ORANGE, DARK_GRAY,
)
from policywerk.viz.traces import update_trace_axes
from policywerk.viz.trajectories import draw_pole, draw_policy_gaussian
from policywerk.primitives.progress import Spinner
from policywerk.primitives import scalar

import matplotlib.pyplot as plt


@dataclass
class PPOSnapshot(FrameSnapshot):
    angle: float
    torque: float       # clamped executed torque, not raw actor mean
    mean: float
    std: float
    step_label: str
    phase: str = ""     # "random", "training", "trained"
    ep_length: int = 0


def main():
    print("=" * 64)
    print("  Lesson 06: Proximal Policy Optimization (2017)")
    print("  Schulman et al., 'Proximal Policy Optimization Algorithms'")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. The big shift
    # -----------------------------------------------------------------------

    print("THE BIG SHIFT")
    print("-" * 64)
    print("""
    Every lesson so far has followed the same pattern: learn a
    value, use the value to pick actions. Bellman computed V(s).
    TD learning estimated V(s) from experience. Q-learning
    estimated Q(s, a). DQN approximated Q(s, a) with a neural
    network. In every case, the agent asked "how good is this
    state or action?" and then picked the action with the
    highest value.

    That pattern ends here.

    PPO does not learn values and derive a policy from them. It
    learns the policy directly. The neural network IS the policy.
    It takes the current state and outputs a probability
    distribution over actions. The agent samples an action from
    that distribution. Training adjusts the distribution so that
    good actions become more probable and bad actions become less
    probable.

    Side by side with DQN:

      DQN:  network(state) -> [Q(left), Q(stay), Q(right)]  -> pick highest
      PPO:  network(state) -> (mean=0.3, std=0.6)           -> sample 0.47

    DQN's network outputs three numbers and the agent picks the
    biggest. PPO's network outputs a bell curve and the agent
    draws a random number from it. This is a fundamentally
    different way to make decisions, and it is what makes
    continuous actions possible—you cannot take argmax over an
    infinite range, but you can sample from a bell curve centered
    anywhere.
    """)

    # -----------------------------------------------------------------------
    # 2. Revisiting Balance
    # -----------------------------------------------------------------------

    print("REVISITING BALANCE")
    print("-" * 64)
    print("""
    In Lesson 02, Barto, Sutton, and Anderson (1983) solved the
    inverted pendulum with an ACE/ASE actor-critic. The state was
    discretized into 36 boxes (6 angle bins x 6 velocity bins).
    The action was binary: push left or push right, full force.

    PPO solves the same physics with none of those constraints.
    The state is the raw angle and angular velocity—two
    continuous numbers, not a box index. The action is a
    continuous torque between -1 and +1—not a binary choice.
    The network learns smooth, proportional control: a small
    tilt gets a small correction, a large tilt gets a large one.

      State:   (angle, angular velocity)—two continuous numbers
      Action:  torque in [-1, +1]—any value, not just left/right
      Reward:  +1 per step survived, 0 when the pole falls
      Goal:    survive 500 steps

    Same environment, same goal, but 34 years of progress:

      L02 (1983): 2 discrete actions, 36 boxes, eligibility traces
      L06 (2017): continuous torque, raw state, neural policy gradients
    """)

    # -----------------------------------------------------------------------
    # 3. The simplest policy gradient
    # -----------------------------------------------------------------------

    print("THE SIMPLEST POLICY GRADIENT")
    print("-" * 64)
    print("""
    The core idea of policy gradient methods fits in one sentence:
    if an action turned out better than expected, adjust the
    network to make that action more likely next time. If it
    turned out worse than expected, make it less likely.

    The intuition. The pole is tilting right. The network
    outputs a bell curve centered at torque = 0.1. The agent
    samples torque = -0.3 from the tail of the curve. Over the
    next several steps, the pole recovers and the agent survives
    longer than the critic predicted. That sequence of events
    produces a positive advantage for torque = -0.3.

    The update: shift the bell curve so that torque = -0.3
    becomes more probable when the pole tilts right. The network
    adjusts its weights, and the bell curve's center moves
    toward -0.3. (The actual mechanism—advantages, the critic,
    and how "better than expected" is computed—comes in section
    6. For now, the key point is the direction of the update.)

    Compare this to DQN's update rule. DQN asked "what is this
    action worth?" and updated a value toward a target. PPO asks
    "should I do this action more or less often?" and adjusts a
    probability. There is no target in the DQN sense—there is
    only the direction: more likely or less likely, and by how
    much.

    This is the policy gradient. Every algorithm from here onward
    uses some version of it.
    """)

    # -----------------------------------------------------------------------
    # 4. The policy as a bell curve
    # -----------------------------------------------------------------------

    print("THE POLICY AS A BELL CURVE")
    print("-" * 64)
    print("""
    The "bell curve" is a Gaussian distribution, defined by two
    numbers: the mean (where the curve is centered) and the
    standard deviation, or std (how wide the curve is).

    The actor network takes the current state and outputs both:

      Actor:  state -> [mean, log_std]
      Policy: sample action from Gaussian(mean, exp(log_std))

    The output is log_std rather than std directly because the
    log can be any real number (positive or negative), while std
    must be positive. Taking exp() of the output guarantees a
    valid standard deviation.

    A wide bell curve (large std) means the agent is uncertain—
    it explores by sampling a broad range of torques. A narrow
    bell curve (small std) means the agent is confident—it
    applies nearly the same torque every time.

    The bell curve assigns a probability to every possible action.
    The code uses the log of this probability (log_prob) because
    logs are numerically more stable—the details are in the
    Gaussian class. What matters for the policy gradient is that
    log_prob tells the network "how likely was this action under
    the current policy?" If the action was good, training
    increases its log_prob. If bad, training decreases it.

    Concrete example. The network outputs mean=0.3, std=0.6. The
    agent samples torque=0.47 from the bell curve. This turned
    out well—the pole recovered. Training shifts the mean toward
    0.47 and might narrow the std, making the agent more likely
    to apply similar torque in this state next time.
    """)

    # -----------------------------------------------------------------------
    # 5. Why clipping
    # -----------------------------------------------------------------------

    print("WHY CLIPPING")
    print("-" * 64)
    print("""
    The policy gradient from section 3 has a problem: if one good
    action makes the network wildly more likely to repeat it, the
    policy can collapse. A single lucky sample could shift the
    entire bell curve to one side, destroying behavior that was
    working elsewhere.

    PPO's solution is the clipped surrogate objective. After each
    batch of experience, PPO computes a ratio for each action:
    how much more (or less) likely is this action under the
    updated policy compared to the policy that collected the data?

      ratio = new_probability / old_probability

    A ratio of 1.0 means no change. A ratio of 1.5 means the
    action is now 50% more likely.

    The clip limits how far the ratio can go:

      L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)

    Concrete example. The agent applied torque 0.3 when the pole
    was tilting right. This turned out well (advantage = +2.0).
    After one gradient step, the new policy wants to make this
    action more likely, pushing the ratio toward 1.5. But with
    epsilon = 0.2, the clip caps it at 1.2. Beyond that point,
    the gradient is zero—the update stops.

    This is the "proximal" in PPO: the new policy stays close
    to the old one. No single update can move the policy too
    far.
    """)

    # -----------------------------------------------------------------------
    # 6. The critic and advantages
    # -----------------------------------------------------------------------

    print("THE CRITIC AND ADVANTAGES")
    print("-" * 64)
    print("""
    The policy gradient says "make good actions more likely." But
    how does the agent know which actions were good? If the pole
    survived 100 steps, all 100 actions contributed—but some
    mattered more than others. Using raw rewards would make every
    action in a good episode look good and every action in a bad
    episode look bad.

    The solution is the advantage: how much better was this action
    than what was expected? A separate network—the critic—
    estimates the expected value of each state. The advantage is:

      advantage = what happened - what was expected

    Positive advantage means "better than the critic predicted."
    Negative means "worse." The policy gradient uses advantages
    instead of raw rewards, so only actions that were genuinely
    better (or worse) than usual get credit.

    This is the same actor-critic idea from Lesson 02. The ACE
    was the critic; the ASE was the actor. In PPO, both are
    neural networks instead of weight vectors, but the split is
    the same: the critic evaluates, the actor acts.

    Computing advantages is the same TD-vs-MC tradeoff from
    Lesson 03. GAE (Generalized Advantage Estimation) blends
    both using lambda:

      delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
      A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...

    Concrete example. Three steps with gamma=0.99:

      Step 0: reward=1.0, V(s_0)=10.0, V(s_1)=12.0
        delta_0 = 1.0 + 0.99*12.0 - 10.0 = 2.88

      Step 1: reward=1.0, V(s_1)=12.0, V(s_2)=11.0
        delta_1 = 1.0 + 0.99*11.0 - 12.0 = -0.11

      Step 2: reward=1.0, V(s_2)=11.0, V(s_3)=10.0
        delta_2 = 1.0 + 0.99*10.0 - 11.0 = -0.10

    With lambda=0 (pure TD): A_0 = delta_0 = 2.88.
    Only the one-step residual matters.

    With lambda=0.95: A_0 = 2.88 + 0.94*(-0.11) + 0.89*(-0.10)
    = 2.69. The positive delta_0 is partially offset by the
    negative deltas that follow—the critic was already too
    optimistic about V(s_1), so the advantage shrinks. This is
    GAE reducing variance: one lucky step doesn't inflate the
    advantage as much when subsequent steps bring the estimate
    back to earth.

    Lambda = 0.95 is the standard choice. With lambda = 1 on a
    complete episode, the advantage equals the full Monte Carlo
    return minus the baseline. In PPO's truncated batches, the
    final step still bootstraps from the critic, so it is never
    purely Monte Carlo in practice.
    """)

    # -----------------------------------------------------------------------
    # 7. Multiple epochs
    # -----------------------------------------------------------------------

    num_epochs = 3

    print("MULTIPLE EPOCHS")
    print("-" * 64)
    print(f"""
    PPO is on-policy: it collects a batch of experience, learns
    from it, then throws it away and collects fresh data. DQN
    (Lesson 05) solved data efficiency through experience
    replay—a buffer of past transitions, sampled repeatedly.
    PPO cannot reuse old data because the policy has changed
    since it was collected, making the old transitions off-policy.

    But PPO can reuse the current batch. After collecting T
    steps, PPO runs K passes (epochs) of gradient descent over
    the same data. The clipped surrogate is what makes this safe:
    even if the policy drifts during the K passes, the clip
    prevents it from moving too far from the policy that
    collected the data.

    This is a direct tradeoff. More epochs extract more learning
    per batch (better sample efficiency), but push the policy
    further from the collection policy (more staleness). K=3 to
    K=10 is typical; beyond that, the clip fires on most samples
    and learning stalls. We use K={num_epochs}.
    """)

    # -----------------------------------------------------------------------
    # 8. Training
    # -----------------------------------------------------------------------

    print("TRAINING")
    print("-" * 64)

    num_iterations = 250
    steps_per_iter = 500
    hidden_size = 64
    learning_rate_actor = 0.001
    learning_rate_critic = 0.003
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    entropy_coeff = 0.005

    print(f"""
    Each iteration collects {steps_per_iter} steps of experience, then
    runs {num_epochs} epochs of PPO updates over the whole batch.
    The actor and critic are separate networks, each with a
    {hidden_size}-neuron hidden layer. The hidden activation is
    tanh instead of DQN's ReLU: tanh's bounded output (-1 to +1)
    prevents activation explosion when inputs are continuous and
    potentially large. ReLU's unbounded positive range works well
    for pixel inputs (non-negative) but can cause instability
    with raw continuous state variables.

    Actor:  2 inputs (angle, velocity) -> {hidden_size} hidden (tanh) -> 2 outputs (mean, log_std)
    Critic: 2 inputs (angle, velocity) -> {hidden_size} hidden (tanh) -> 1 output (value)

    Hyperparameters:
      Iterations:     {num_iterations}
      Steps/iter:     {steps_per_iter}
      Epochs:         {num_epochs}
      Gamma:          {gamma}
      Lambda (GAE):   {lam}
      Clip epsilon:   {clip_epsilon}
      Actor LR:       {learning_rate_actor}
      Critic LR:      {learning_rate_critic}
      Entropy coeff:  {entropy_coeff}
    """)

    env = Balance()

    actor_net, critic_net, history = ppo(
        env,
        num_iterations=num_iterations,
        steps_per_iter=steps_per_iter,
        num_epochs=num_epochs,
        gamma=gamma,
        lam=lam,
        clip_epsilon=clip_epsilon,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        entropy_coeff=entropy_coeff,
        hidden_size=hidden_size,
        seed=42,
    )

    # Print training summary
    window = 30
    print()
    print(f"    Average per {window} iterations:")
    for start in range(0, num_iterations, window):
        end = min(start + window, num_iterations)
        avg_r = sum(h["avg_reward"] for h in history[start:end]) / (end - start)
        avg_std = sum(h["mean_std"] for h in history[start:end]) / (end - start)
        avg_ent = sum(h["entropy"] for h in history[start:end]) / (end - start)
        print(f"      Iterations {start:3d}-{end-1:3d}:  "
              f"reward {avg_r:7.1f}  std {avg_std:.3f}  entropy {avg_ent:.3f}")
    print()

    # -----------------------------------------------------------------------
    # 9. What PPO learned
    # -----------------------------------------------------------------------

    print("WHAT PPO LEARNED")
    print("-" * 64)
    print("""
    With the trained policy, the agent plays greedily: at each
    step it uses the mean of the Gaussian (no sampling noise).
    This is the deterministic policy—the best single action
    the network has learned for each state.
    """)

    # Greedy evaluation—use clamped torque (what the environment actually executes)
    eval_env = Balance()
    state = eval_env.reset()
    eval_steps = 0
    eval_reward = 0.0
    eval_angles = []
    eval_torques = []
    for _ in range(500):
        actor_out, _ = network_forward(actor_net, state.features)
        mean = actor_out[0]
        executed_torque = scalar.clamp(mean, -1.0, 1.0)
        state, reward, done = eval_env.step_continuous(executed_torque)
        eval_reward += reward
        eval_steps += 1
        eval_angles.append(state.features[0])
        eval_torques.append(executed_torque)
        if done:
            break

    print(f"    Greedy evaluation (deterministic policy):")
    print(f"      Steps survived:  {eval_steps}")
    print(f"      Total reward:    {eval_reward:.0f}")
    max_angle = max(abs(a) for a in eval_angles)
    avg_torque = sum(abs(t) for t in eval_torques) / len(eval_torques)
    print(f"      Max |angle|:     {max_angle:.4f} rad")
    print(f"      Avg |torque|:    {avg_torque:.4f}")
    print()

    # Policy at a few representative states—show clamped torque
    print("    Policy at representative states:")
    test_states = [
        (0.0, 0.0, "upright, still"),
        (0.1, 0.0, "tilting right"),
        (-0.1, 0.0, "tilting left"),
        (0.0, 0.5, "upright, rotating right"),
    ]
    for angle, vel, desc in test_states:
        out, _ = network_forward(actor_net, [angle, vel])
        m = out[0]
        executed = scalar.clamp(m, -1.0, 1.0)
        ls = scalar.clamp(out[1], -2.0, 2.0)
        s = scalar.exp(ls)
        print(f"      {desc:30s}  torque={executed:+.3f}  std={s:.3f}")
    print()

    eval_survived, _ = balance_outcome(eval_steps, reward, max_steps=500)
    if eval_survived:
        survival_msg = f"The agent survived {eval_steps} steps—the maximum."
    else:
        survival_msg = f"The agent survived {eval_steps} of 500 steps."
    print(f"""    {survival_msg} It kept
    the pole within {max_angle:.4f} radians of vertical, applying
    an average torque of {avg_torque:.4f}. Compare this to L02's
    discrete push-left / push-right: PPO applies smooth,
    proportional corrections that keep the pole nearly still.

    The policy at representative states shows what the network
    learned: when the pole tilts right, apply negative torque
    (push left). When tilting left, apply positive torque (push
    right). When rotating right, apply strong negative torque to
    counteract the momentum. The "upright, still" torque is not
    zero—the network learned to pre-compensate for the pole's
    initial rightward tilt of 0.01 radians.
    """)
    print()

    # -----------------------------------------------------------------------
    # 10. Animation
    # -----------------------------------------------------------------------

    print("GENERATING ARTIFACTS")
    print("-" * 64)

    os.makedirs("output", exist_ok=True)

    snapshots: list[PPOSnapshot] = []
    from policywerk.primitives.random import create_rng as _create_rng

    # --- Phase 1: Random policy rollout ---
    rand_rng = _create_rng(99)
    rand_env = Balance()
    state = rand_env.reset()
    rand_done = False
    rand_step = 0
    while not rand_done and rand_step < 200:
        torque = rand_rng.gauss(0.0, 1.0)
        torque = max(-1.0, min(1.0, torque))
        state, reward, rand_done = rand_env.step_continuous(torque)
        rand_step += 1
        snapshots.append(PPOSnapshot(
            episode=0,
            total_reward=float(rand_step),
            angle=state.features[0],
            torque=torque,
            mean=0.0,
            std=1.0,
            step_label=f"1/3  Random policy (step {rand_step})",
            phase="random",
            ep_length=rand_step,
        ))
    # Hold on final frame
    for _ in range(8):
        snapshots.append(PPOSnapshot(
            episode=0,
            total_reward=float(rand_step),
            angle=state.features[0],
            torque=0.0,
            mean=0.0,
            std=1.0,
            step_label=f"1/3  Fell after {rand_step} steps",
            phase="random",
            ep_length=rand_step,
        ))

    # --- Phase 2: Training progression ---
    sample_iters = sorted(set(
        list(range(0, num_iterations, max(1, num_iterations // 12))) +
        [num_iterations - 1]
    ))
    for idx in sample_iters:
        h = history[idx]
        snapshots.append(PPOSnapshot(
            episode=idx,
            total_reward=h["avg_reward"],
            angle=0.0,  # show pole upright as placeholder
            torque=0.0,
            mean=0.0,
            std=h["mean_std"],
            step_label=f"2/3  Training: iteration {idx}/{num_iterations}",
            phase="training",
            ep_length=int(h["avg_reward"]),
        ))

    # --- Phase 3: Trained policy rollout (loop twice) ---
    for _loop in range(2):
        t_env = Balance()
        state = t_env.reset()
        t_done = False
        t_step = 0
        while not t_done and t_step < 500:
            actor_out, _ = network_forward(actor_net, state.features)
            raw_mean = actor_out[0]
            executed = scalar.clamp(raw_mean, -1.0, 1.0)
            log_std = scalar.clamp(actor_out[1], -2.0, 2.0)
            std = scalar.exp(log_std)
            state, reward, t_done = t_env.step_continuous(executed)
            t_step += 1
            if t_step % 5 == 0 or t_done:
                snapshots.append(PPOSnapshot(
                    episode=num_iterations,
                    total_reward=float(t_step),
                    angle=state.features[0],
                    torque=executed,
                    mean=executed,
                    std=std,
                    step_label=f"3/3  Trained policy (step {t_step})",
                    phase="trained",
                    ep_length=t_step,
                ))
        survived, outcome_label = balance_outcome(t_step, reward, max_steps=500)
        end_label = f"3/3  {outcome_label}"
        hold_angle = 0.0 if survived else state.features[0]
        for _ in range(12):
            snapshots.append(PPOSnapshot(
                episode=num_iterations,
                total_reward=float(t_step),
                angle=hold_angle,
                torque=0.0,
                mean=executed,
                std=std,
                step_label=end_label,
                phase="trained",
                ep_length=t_step,
            ))

    # --- Artifact 1: Animation ---

    fig, axes = create_lesson_figure(
        "Lesson 06: PPO",
        subtitle="Schulman et al. (2017) | policy gradients replace Q-values",
        figsize=(12, 7),
    )

    rewards_list = [h["avg_reward"] for h in history]

    def update(frame_idx):
        snap = snapshots[frame_idx]

        # Top-left: pole
        axes["env"].clear()
        draw_pole(axes["env"], snap.angle, action=snap.torque)
        axes["env"].set_title(snap.step_label, fontsize=10)

        # Top-right: policy Gaussian
        axes["algo"].clear()
        if snap.phase == "random":
            draw_policy_gaussian(axes["algo"], 0.0, 1.0, action_range=(-2.0, 2.0))
            axes["algo"].set_title("Random Policy (std=1.0)", fontsize=10)
        elif snap.phase == "trained":
            draw_policy_gaussian(axes["algo"], snap.mean, snap.std, action_range=(-2.0, 2.0))
            axes["algo"].set_title(f"Trained Policy (std={snap.std:.2f})", fontsize=10)
        else:
            draw_policy_gaussian(axes["algo"], 0.0, snap.std, action_range=(-2.0, 2.0))
            axes["algo"].set_title(f"Policy at iter {snap.episode} (std={snap.std:.2f})", fontsize=10)

        # Bottom: reward trace
        axes["trace"].clear()
        if snap.phase == "random":
            axes["trace"].axis("off")
            axes["trace"].text(0.5, 0.5, "Training not started",
                               transform=axes["trace"].transAxes,
                               ha="center", va="center", fontsize=10,
                               color=DARK_GRAY, style="italic")
        else:
            n = min(snap.episode + 1, len(rewards_list))
            if n > 0:
                update_trace_axes(axes["trace"], rewards_list[:n],
                                  label="Avg reward", color=TEAL)
            axes["trace"].set_ylabel("Reward", fontsize=9)
            axes["trace"].set_xlabel("Iteration", fontsize=9)
            axes["trace"].set_title("Learning Progress", fontsize=10)

    with Spinner("Generating animation"):
        save_animation(fig, update, len(snapshots),
                       "output/06_ppo_artifact.gif", fps=8)

    # --- Artifact 2: Poster ---

    fig2, axes2 = create_lesson_figure(
        "Lesson 06: PPO",
        subtitle="Schulman et al. (2017)",
        figsize=(12, 7),
    )

    # Pick the frame where the pole is most upright (smallest |angle|)
    if eval_angles:
        best_idx = min(range(len(eval_angles)), key=lambda i: abs(eval_angles[i]))
        poster_angle = eval_angles[best_idx]
        poster_torque = eval_torques[best_idx]
    else:
        poster_angle = 0.0
        poster_torque = 0.0
    final_out, _ = network_forward(actor_net, [0.0, 0.0])
    poster_mean = final_out[0]
    poster_log_std = scalar.clamp(final_out[1], -2.0, 2.0)
    poster_std = scalar.exp(poster_log_std)

    def update_poster(frame_idx):
        draw_pole(axes2["env"], poster_angle, action=poster_torque)
        axes2["env"].set_title("Trained Agent (greedy)", fontsize=10)

        draw_policy_gaussian(axes2["algo"], poster_mean, poster_std, action_range=(-2.0, 2.0))
        axes2["algo"].set_title(f"Policy distribution (std={poster_std:.2f})", fontsize=10)

        axes2["trace"].plot(range(num_iterations), rewards_list,
                            color=TEAL, linewidth=0.5, alpha=0.3)
        w = 20
        if num_iterations > w:
            smooth = [sum(rewards_list[max(0, i - w + 1):i + 1]) /
                      len(rewards_list[max(0, i - w + 1):i + 1])
                      for i in range(num_iterations)]
            axes2["trace"].plot(range(num_iterations), smooth,
                                color=TEAL, linewidth=1.5, label="Reward (smoothed)")
        axes2["trace"].set_ylabel("Reward", fontsize=9)
        axes2["trace"].set_xlabel("Iteration", fontsize=9)
        axes2["trace"].set_title("Training Progress", fontsize=10)
        axes2["trace"].legend(fontsize=8)

    with Spinner("Generating poster"):
        save_poster(fig2, update_poster, 0, "output/06_ppo_poster.png")
    plt.close(fig2)

    # --- Artifact 3: Trace ---

    fig3, (ax_r, ax_e) = plt.subplots(2, 1, figsize=(10, 5), dpi=150, sharex=True)

    ax_r.plot(range(num_iterations), rewards_list,
              color=TEAL, linewidth=0.5, alpha=0.3, label="Raw reward")
    w = 20
    if num_iterations > w:
        smooth_r = [sum(rewards_list[max(0, i - w + 1):i + 1]) /
                    len(rewards_list[max(0, i - w + 1):i + 1])
                    for i in range(num_iterations)]
        ax_r.plot(range(num_iterations), smooth_r,
                  color=TEAL, linewidth=1.5, label="Reward (smoothed)")
    ax_r.set_ylabel("Avg Reward", fontsize=9)
    ax_r.set_title("PPO Training on Balance", fontsize=10)
    ax_r.legend(fontsize=8)
    ax_r.grid(True, alpha=0.3)

    entropy_list = [h["entropy"] for h in history]
    ax_e.plot(range(num_iterations), entropy_list,
              color=ORANGE, linewidth=1.0, label="Entropy")
    ax_e.set_ylabel("Entropy", fontsize=9)
    ax_e.set_xlabel("Iteration", fontsize=9)
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3)

    fig3.tight_layout()

    with Spinner("Generating trace"):
        fig3.savefig("output/06_ppo_trace.png")
    plt.close(fig3)

    print("    Generating animation... done.")
    print("    Generating poster... done.")
    print("    Generating trace... done.")
    print()
    print("    Artifacts saved to output/:")
    print("      06_ppo_artifact.gif  PPO training on Balance")
    print("      06_ppo_poster.png    Trained agent snapshot")
    print("      06_ppo_trace.png     Reward and entropy curves")
    print()

    print("=" * 64)
    print("  Lesson 06 complete.")
    print()
    print("  PPO closed the gap between discrete and continuous control.")
    print("  Lesson 05's DQN needed a finite set of actions to take")
    print("  argmax over. PPO replaced the value function with a policy")
    print("  network that outputs a probability distribution. Three ideas")
    print("  made it work: clipped surrogate (trust region without second-")
    print("  order optimization), GAE (advantage estimation with bias-")
    print("  variance tradeoff), and multiple epochs (reusing on-policy")
    print("  data for more learning per batch).")
    print()
    print("  But PPO still has a fundamental limitation: it learns from")
    print("  real experience only. Every training step requires actually")
    print("  running the environment. If the environment is expensive")
    print("  (a robot, a simulator, the real world), this is wasteful.")
    print()
    print("  In Lesson 07, DreamerV3 takes a different approach: learn a")
    print("  model of the environment, then train the policy in imagination.")
    print("  Instead of running the real environment for every gradient")
    print("  step, the agent dreams up synthetic experience and learns")
    print("  from it. Real data trains the world model; imagined data")
    print("  trains the policy.")
    print("=" * 64)


if __name__ == "__main__":
    main()
