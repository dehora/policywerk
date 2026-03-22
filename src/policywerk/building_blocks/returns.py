"""Level 1: Return computation.

The "return" in RL (nothing to do with Python's return statement) is
the total reward an agent collects from the current moment until the
episode ends. It's the number the agent is ultimately trying to
maximize — not the immediate reward from one step, but the cumulative
payoff of an entire sequence of decisions.

The complication is that future rewards are uncertain and distant.
A reward right now is guaranteed, but a reward 10 steps from now
depends on what happens in between. Discounting handles this: each
step into the future multiplies the reward by gamma (γ), a number
between 0 and 1. With gamma=0.9, a reward 10 steps away is worth
0.9^10 ≈ 0.35 of its face value. This captures the intuition that
a bird in the hand is worth more than one in the bush.

The fundamental tradeoff in return estimation:

  Monte Carlo (discount_return): wait until the episode ends, then
    add up all the actual rewards with discounting. This gives the
    true return — no guessing — but it's noisy. Two episodes from
    the same state can give very different returns due to randomness.

  TD(0) / bootstrapping (n_step_return with n=1): don't wait. After
    one step, use the agent's own value estimate for the next state
    as a stand-in for all future rewards. This is biased (the estimate
    might be wrong) but low-variance (it doesn't depend on the
    randomness of an entire episode).

  TD(λ) / lambda-return: a weighted blend of all n-step returns.
    lambda=0 gives pure TD(0), lambda=1 gives pure Monte Carlo,
    and values in between trade off bias against variance.

  GAE (gae): a practical version of the same idea, computing
    per-step advantages (how much better was this action than
    average?) with the same lambda-controlled tradeoff. Used by
    PPO (L06).

This spectrum — from "wait and see" to "guess and go" — is one
of the central ideas in reinforcement learning. Every algorithm
in this project sits somewhere on it.
"""

from policywerk.primitives import scalar

Vector = list[float]


def discount_return(rewards: Vector, gamma: float) -> float:
    """Monte Carlo return: G = r_0 + γr_1 + γ²r_2 + ...

    Uses all rewards in the sequence. Accurate on average but noisy —
    needs many episodes for a stable estimate.

    gamma: discount factor — how much to devalue future rewards.
           gamma=0.9 means a reward 10 steps away is worth about 0.35
           of its face value.
    """
    g = 0.0
    for r in reversed(rewards):
        g = scalar.add(r, scalar.multiply(gamma, g))
    return g


def n_step_return(rewards: Vector, bootstrap_value: float, gamma: float) -> float:
    """N-step return: r_0 + γr_1 + ... + γⁿV(sₙ).

    Uses n actual rewards then bootstraps from a value estimate.
    Interpolates between TD(0) (n=1) and Monte Carlo (n=∞).

    bootstrap_value: an estimated value standing in for the unknown future —
                     the agent guesses how much reward will come from here.
    """
    g = bootstrap_value
    for r in reversed(rewards):
        g = scalar.add(r, scalar.multiply(gamma, g))
    return g


def lambda_return(rewards: Vector, values: Vector, next_value: float,
                  gamma: float, lam: float) -> float:
    """TD(λ) return from the first timestep.

    Convention (same as gae):
      rewards[k] = reward received at step k
      values[k]  = V(s_k), the estimated value of state k
      next_value = V(s_T), the bootstrap value after the last step

    The 1-step return from step k is: r_k + gamma * V(s_{k+1}).

    lambda=0: pure 1-step TD — bootstrap from the next state immediately.
    lambda=1: full Monte Carlo-like — use all actual rewards, bootstrap
              only from next_value at the end.
    Values in between blend short and long n-step returns.
    """
    t = len(rewards)
    if t == 0:
        return 0.0
    g_lambda = 0.0
    g_n = next_value
    for k in range(t - 1, -1, -1):
        # V(s_{k+1}): the value of the next state
        v_next = next_value if k == t - 1 else values[k + 1]
        # 1-step return: r_k + gamma * V(s_{k+1})
        g_1 = scalar.add(rewards[k], scalar.multiply(gamma, v_next))
        # n-step return (accumulates from the end)
        g_n = scalar.add(rewards[k], scalar.multiply(gamma, g_n))
        # Lambda-weighted blend of 1-step and n-step
        g_lambda = scalar.add(
            scalar.multiply(scalar.subtract(1.0, lam), g_1),
            scalar.multiply(lam, g_n),
        )
        g_n = g_lambda
    return g_lambda


def gae(rewards: Vector, values: Vector, next_value: float,
        gamma: float, lam: float) -> Vector:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Computes the advantage at each timestep — how much better was this action
    than average? Positive = better than expected, negative = worse.

      δ_t = r_t + γV(s_{t+1}) - V(s_t)
      A_t = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

    Returns a vector of advantages, one per timestep.
    """
    t = len(rewards)
    advantages = [0.0] * t
    gae_val = 0.0
    for i in range(t - 1, -1, -1):
        v_next = next_value if i == t - 1 else values[i + 1]
        delta = scalar.subtract(
            scalar.add(rewards[i], scalar.multiply(gamma, v_next)),
            values[i],
        )
        gae_val = scalar.add(delta, scalar.multiply(scalar.multiply(gamma, lam), gae_val))
        advantages[i] = gae_val
    return advantages
