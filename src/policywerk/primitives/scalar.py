"""Level 0: Scalar operations.

The most basic building blocks — named wrappers around arithmetic
so that every operation in the system has an explicit identity.
Everything above this level is built from these.
"""

import math


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def negate(a: float) -> float:
    """Flip the sign of a number."""
    return -a


def inverse(a: float) -> float:
    """Reciprocal: 1 divided by a."""
    return 1.0 / a


def exp(x: float) -> float:
    """Raise e (≈2.718) to the power x — grows very fast.

    Used throughout for sigmoid, softmax, and probability computations.
    Clamped to avoid overflow — exp(710) is already beyond float64.
    """
    x = max(-500.0, min(500.0, x))
    return math.exp(x)


def log(x: float) -> float:
    """Natural logarithm — the inverse of exp. log(exp(x)) = x.

    Compresses large numbers and expands small ones. Used for
    log-probabilities and loss functions. Guarded against log(0).
    """
    if x <= 0:
        x = 1e-15
    return math.log(x)


def power(x: float, n: float) -> float:
    """Raise x to the power n."""
    return x ** n


def clamp(x: float, lo: float, hi: float) -> float:
    """Restrict x to stay within [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def abs_val(x: float) -> float:
    """Absolute value — distance from zero, always non-negative."""
    return -x if x < 0 else x


def sign(x: float) -> float:
    """Sign of x: -1.0 if negative, +1.0 if positive, 0.0 if zero."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0
