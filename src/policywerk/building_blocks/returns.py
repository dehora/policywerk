"""Level 1: Return computation.

Different ways to estimate the return (cumulative discounted reward)
from a sequence of rewards. The spectrum from Monte Carlo to TD(0)
and everything in between.
"""

from policywerk.primitives import scalar

Vector = list[float]


def discount_return(rewards: Vector, gamma: float) -> float:
    """Monte Carlo return: G = r_0 + γr_1 + γ²r_2 + ...

    Uses all rewards in the sequence. Unbiased but high variance.
    """
    g = 0.0
    for r in reversed(rewards):
        g = scalar.add(r, scalar.multiply(gamma, g))
    return g


def n_step_return(rewards: Vector, bootstrap_value: float, gamma: float) -> float:
    """N-step return: r_0 + γr_1 + ... + γⁿV(sₙ).

    Uses n actual rewards then bootstraps from a value estimate.
    Interpolates between TD(0) (n=1) and Monte Carlo (n=∞).
    """
    g = bootstrap_value
    for r in reversed(rewards):
        g = scalar.add(r, scalar.multiply(gamma, g))
    return g


def lambda_return(rewards: Vector, values: Vector, gamma: float, lam: float) -> float:
    """TD(λ) return: weighted average of all n-step returns.

    λ=0 gives TD(0), λ=1 gives Monte Carlo.
    Computed efficiently with the forward view.
    """
    t = len(rewards)
    if t == 0:
        return 0.0
    g_lambda = 0.0
    g_n = values[t - 1] if t <= len(values) else 0.0
    for k in range(t - 1, -1, -1):
        v_next = values[k] if k < len(values) else 0.0
        g_1 = scalar.add(rewards[k], scalar.multiply(gamma, v_next))
        g_n = scalar.add(rewards[k], scalar.multiply(gamma, g_n))
        g_lambda = scalar.add(
            scalar.multiply(scalar.subtract(1.0, lam), g_1),
            scalar.multiply(lam, g_n),
        )
        g_n = g_lambda
    return g_lambda


def gae(rewards: Vector, values: Vector, next_value: float,
        gamma: float, lam: float) -> Vector:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Computes advantage estimates A_t for each timestep:
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
