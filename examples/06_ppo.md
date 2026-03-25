# Lesson 6: Proximal Policy Optimization (Schulman et al., 2017)

PPO learns the policy directly instead of deriving it from Q-values. The actor network outputs a Gaussian distribution over continuous actions, and the agent samples from it. A clipped surrogate objective prevents catastrophically large updates.

```
uv run python lessons/06_ppo.py
```

## From Values to Actions

DQN (Lesson 05) learned Q-values for three discrete actions and picked the highest. That works when actions come from a short list. But what if the action is a continuous torque—any real number between -1 and +1? You cannot take argmax over an infinite range.

PPO solves this by outputting a probability distribution. For continuous control, that distribution is a Gaussian (bell curve). The network outputs the mean and standard deviation, and the agent samples an action from the curve.

## The Policy as a Bell Curve

The actor network takes the state (angle, angular velocity) and outputs two numbers: the mean and log-standard-deviation of a Gaussian over torques.

```
Actor network:  state -> [mean, log_std]
Policy:         action ~ Gaussian(mean, exp(log_std))
```

A wide bell curve (large std) means exploration. A narrow bell curve means the agent is confident. Training makes the curve narrower and shifts it toward the right torque.

The log of the probability under the Gaussian:

```
log p(a) = -0.5 * ((a - mean) / std)^2 - log(std) - 0.5 * log(2pi)
```

Actions that led to high advantage get their log-probability increased; actions that led to low advantage get decreased. This is the policy gradient.

## The Surrogate Objective

Large policy updates can be catastrophic. PPO constrains the update using a clipped surrogate:

```
ratio = pi_new(a | s) / pi_old(a | s) = exp(log_prob_new - log_prob_old)
L = min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
```

If the advantage is positive (good action), the objective wants to increase the ratio. But the clip prevents it from going above 1 + epsilon (0.2). Beyond that point, the gradient is zero. This keeps the new policy close to the old one—the "proximal" in PPO.

## Generalized Advantage Estimation

GAE blends TD and Monte Carlo advantage estimation using lambda, the same tradeoff from Lesson 03:

```
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...
```

Lambda = 0.95 leans toward longer horizons while keeping variance manageable.

## Revisiting Balance

Lesson 02 solved this with Barto, Sutton, and Anderson's 1983 ACE/ASE: 2 discrete actions, 36 boxes, eligibility traces. PPO uses the raw angle and velocity directly, applies continuous torque, and learns through neural policy gradients. Same physics, 34 years of algorithmic progress.

## Training Results

```
Actor:  2 inputs -> 64 hidden (tanh) -> 2 outputs (mean, log_std)
Critic: 2 inputs -> 64 hidden (tanh) -> 1 output (value)

Iterations:     250
Steps/iter:     500
Epochs:         3
Gamma:          0.99
Lambda (GAE):   0.95
Clip epsilon:   0.2
Actor LR:       0.001
Critic LR:      0.003

Average per 30 iterations:
  Iterations   0- 29:  reward    52.7  std 1.058  entropy 1.470
  Iterations  30- 59:  reward   104.3  std 0.910  entropy 1.317
  Iterations  60- 89:  reward   249.8  std 0.765  entropy 1.145
  Iterations  90-119:  reward   311.9  std 0.612  entropy 0.926
  Iterations 120-149:  reward   425.8  std 0.530  entropy 0.783
  Iterations 150-179:  reward   381.0  std 0.556  entropy 0.827
  Iterations 180-209:  reward   386.4  std 0.603  entropy 0.911
  Iterations 210-239:  reward   473.0  std 0.655  entropy 0.994
```

The reward (average episode length) climbs from ~53 to 473, with a dip at iterations 150-209 before recovering. The standard deviation drops from ~1.06 to ~0.53 by iteration 150, then increases slightly as the agent explores further refinements. Entropy tracks the same trend.

The dip at iterations 150-209 is a common PPO pattern. The agent reaches near-optimal performance (reward ~426), then the policy std narrows enough that the advantages become noisier—the critic's estimates are calibrated for a wider exploration range. The recovery to 473 by iterations 210-239 shows the critic catching up. This is exactly the kind of instability that motivates the clipped surrogate: without the clip, this dip would be a collapse.

## What the Network Learned

With the deterministic policy (using the clamped mean, no sampling):

```
Greedy evaluation:
  Steps survived:  500
  Total reward:    500
  Max |angle|:     0.1620 rad
  Avg |torque|:    0.2928

Policy at representative states:
  upright, still          torque=+0.438  std=0.649
  tilting right           torque=-0.028  std=0.664
  tilting left            torque=+0.903  std=0.635
  rotating right          torque=-0.831  std=0.849
```

The agent survived the full 500 steps, keeping the pole within 0.16 radians of vertical with an average executed torque of 0.29. All reported torques are clamped to the environment's [-1, 1] action range.

The policy at representative states reveals the learned controller: tilting left produces strong positive torque (+0.903), rotating right produces strong negative torque (-0.831), and tilting right gets a small corrective nudge (-0.028). The "upright, still" bias (+0.438) reflects the pole's initial rightward tilt of 0.01 radians—the network learned to pre-compensate.

Compare this to L02's binary push-left/push-right: PPO applies smooth, proportional corrections that keep the pole nearly still.

## Artifacts

### Training Animation

![PPO training animation](img/06_ppo_artifact.gif)

Three phases: random policy (pole falls immediately), training progression (bell curve narrowing), and trained policy (pole stays balanced for 500 steps). The Gaussian distribution in the top-right pane shows the policy sharpening over training.

### Trained Agent Snapshot

![Trained agent](img/06_ppo_poster.png)

The trained pole nearly vertical, the narrow policy Gaussian, and the full training reward curve.

### Reward and Entropy Curves

![Reward and entropy](img/06_ppo_trace.png)

Top: average reward per iteration climbing from ~50 to 400+. Bottom: policy entropy declining as the agent commits to a strategy, with the recovery visible around iteration 200.

## Next

PPO learns from real experience only. Every training step requires actually running the environment. In Lesson 07, DreamerV3 learns a model of the environment and trains the policy in imagination—real data trains the world model, imagined data trains the policy.
