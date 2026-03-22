"""Level 1: Probability distributions.

Categorical and Gaussian distributions with sampling, log-probability,
and entropy. Used by policy gradient methods (PPO, Dreamer) to
represent stochastic policies.
"""

import math
import random as _random

from policywerk.primitives import scalar, vector, activations
from policywerk.primitives.random import sample_categorical

Vector = list[float]


class Categorical:
    """Categorical distribution parameterized by logits.

    Converts logits to probabilities via softmax, then supports
    sampling, log-probability computation, and entropy.
    """

    def __init__(self, logits: Vector):
        self.logits = logits
        self.probs = activations.softmax(logits)

    def sample(self, rng: _random.Random) -> int:
        """Draw a sample from the distribution."""
        return sample_categorical(rng, self.probs)

    def log_prob(self, action: int) -> float:
        """Log probability of a specific action."""
        return scalar.log(self.probs[action])

    def entropy(self) -> float:
        """Shannon entropy: -Σ p log p. Measures randomness."""
        total = 0.0
        for p in self.probs:
            if p > 1e-15:
                total = scalar.subtract(total, scalar.multiply(p, scalar.log(p)))
        return total


class Gaussian:
    """Diagonal Gaussian distribution parameterized by mean and std.

    Each dimension is independent. Used for continuous action spaces.
    """

    def __init__(self, mean: Vector, std: Vector):
        self.mean = mean
        self.std = std

    def sample(self, rng: _random.Random) -> Vector:
        """Draw a sample: mean + std * N(0, 1)."""
        return [
            scalar.add(m, scalar.multiply(s, rng.gauss(0.0, 1.0)))
            for m, s in zip(self.mean, self.std)
        ]

    def log_prob(self, x: Vector) -> float:
        """Log probability of x under this Gaussian.

        log p(x) = -0.5 * Σ [ ((x-μ)/σ)² + 2*log(σ) + log(2π) ]
        """
        total = 0.0
        for xi, m, s in zip(x, self.mean, self.std):
            diff = scalar.subtract(xi, m)
            z = scalar.multiply(diff, scalar.inverse(s))
            total = scalar.subtract(total,
                                    scalar.multiply(0.5,
                                                    scalar.add(scalar.multiply(z, z),
                                                               scalar.add(scalar.multiply(2.0, scalar.log(s)),
                                                                          math.log(2.0 * math.pi)))))
        return total

    def entropy(self) -> float:
        """Entropy of diagonal Gaussian: 0.5 * Σ [1 + log(2π) + 2*log(σ)]."""
        total = 0.0
        for s in self.std:
            total = scalar.add(total,
                               scalar.multiply(0.5,
                                               scalar.add(1.0,
                                                          scalar.add(math.log(2.0 * math.pi),
                                                                     scalar.multiply(2.0, scalar.log(s))))))
        return total
