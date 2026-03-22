"""Level 0: Random number generation.

Seeded RNG wrappers for reproducible training and exploration.
All randomness in the project flows through this module.
"""

import math
import random as _random

Vector = list[float]
Matrix = list[list[float]]


def create_rng(seed: int = 42) -> _random.Random:
    return _random.Random(seed)


def uniform(rng: _random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def random_vector(rng: _random.Random, n: int, lo: float = -1.0, hi: float = 1.0) -> Vector:
    return [rng.uniform(lo, hi) for _ in range(n)]


def random_matrix(rng: _random.Random, rows: int, cols: int, lo: float = -1.0, hi: float = 1.0) -> Matrix:
    return [random_vector(rng, cols, lo, hi) for _ in range(rows)]


def xavier_init(rng: _random.Random, fan_in: int, fan_out: int) -> Matrix:
    """Initialize weights scaled to the layer size (Xavier/Glorot uniform).

    Weights drawn from Uniform[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random_matrix(rng, fan_out, fan_in, -limit, limit)


def normal(rng: _random.Random, mean: float = 0.0, std: float = 1.0) -> float:
    """Sample from a normal (Gaussian) distribution."""
    return rng.gauss(mean, std)


def normal_vector(rng: _random.Random, n: int, mean: float = 0.0, std: float = 1.0) -> Vector:
    """Sample n values from a normal distribution."""
    return [rng.gauss(mean, std) for _ in range(n)]


def choice(rng: _random.Random, n: int) -> int:
    """Choose a random integer from 0 to n-1."""
    return rng.randint(0, n - 1)


def sample_categorical(rng: _random.Random, probs: Vector) -> int:
    """Sample an index from a categorical distribution defined by probs.

    Draws a uniform random number and walks through the cumulative
    distribution to find which category it falls in.
    """
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u <= cumulative:
            return i
    return len(probs) - 1
