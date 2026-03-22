"""Level 0: Random number generation.

Seeded RNG wrappers for reproducible training and exploration.
All randomness in the project flows through this module.

Using a seed means the same seed always produces the same sequence
of random numbers — making experiments reproducible. If something
works with seed 42, it will always work with seed 42.
"""

import math
import random as _random

Vector = list[float]
Matrix = list[list[float]]


def create_rng(seed: int = 42) -> _random.Random:
    """Create a new random number generator with a fixed seed."""
    return _random.Random(seed)


def uniform(rng: _random.Random, lo: float, hi: float) -> float:
    """Random number uniformly distributed between lo and hi."""
    return rng.uniform(lo, hi)


def random_vector(rng: _random.Random, n: int, lo: float = -1.0, hi: float = 1.0) -> Vector:
    """Create a vector of n random numbers between lo and hi."""
    return [rng.uniform(lo, hi) for _ in range(n)]


def random_matrix(rng: _random.Random, rows: int, cols: int, lo: float = -1.0, hi: float = 1.0) -> Matrix:
    """Create a matrix of random numbers between lo and hi."""
    return [random_vector(rng, cols, lo, hi) for _ in range(rows)]


def xavier_init(rng: _random.Random, fan_in: int, fan_out: int) -> Matrix:
    """Initialize neural network weights scaled to the layer size.

    fan_in: number of inputs to each neuron.
    fan_out: number of outputs from this layer.

    If initial weights are too large, signals explode as they pass
    through layers. Too small, and they vanish to zero. Xavier
    initialization keeps them in a useful range by scaling to
    sqrt(6 / (fan_in + fan_out)).
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return random_matrix(rng, fan_out, fan_in, -limit, limit)


def normal(rng: _random.Random, mean: float = 0.0, std: float = 1.0) -> float:
    """Sample from a bell-curve (Gaussian) distribution."""
    return rng.gauss(mean, std)


def normal_vector(rng: _random.Random, n: int, mean: float = 0.0, std: float = 1.0) -> Vector:
    """Sample n values from a bell-curve distribution."""
    return [rng.gauss(mean, std) for _ in range(n)]


def choice(rng: _random.Random, n: int) -> int:
    """Choose a random integer from 0 to n-1."""
    return rng.randint(0, n - 1)


def sample_categorical(rng: _random.Random, probs: Vector) -> int:
    """Pick a random index weighted by the given probabilities.

    Higher probabilities mean higher chance of being picked.
    Draws a uniform random number and walks through the cumulative
    probabilities to find which category it falls in.

    Example: probs=[0.1, 0.7, 0.2] → index 1 is chosen 70% of the time.
    """
    u = rng.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if u <= cumulative:
            return i
    return len(probs) - 1
