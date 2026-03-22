"""Level 0: Scalar operations.

The most basic building blocks — named wrappers around arithmetic
so that every operation in the system has an explicit identity.
Everything above this level is built from these.
"""

import math


def multiply(a: float, b: float) -> float:
    return a * b


def add(a: float, b: float) -> float:
    return a + b


def subtract(a: float, b: float) -> float:
    return a - b


def negate(a: float) -> float:
    return -a


def inverse(a: float) -> float:
    return 1.0 / a


def exp(x: float) -> float:
    # Clamp input to avoid overflow — exp(710) is already beyond float64
    x = max(-500.0, min(500.0, x))
    return math.exp(x)


def log(x: float) -> float:
    # Guard against log(0) — use a tiny epsilon
    if x <= 0:
        x = 1e-15
    return math.log(x)


def power(x: float, n: float) -> float:
    return x ** n


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def abs_val(x: float) -> float:
    """Absolute value of x."""
    return -x if x < 0 else x


def sign(x: float) -> float:
    """Sign of x: -1.0, 0.0, or 1.0."""
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return 0.0
