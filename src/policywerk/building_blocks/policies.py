"""Level 1: Action selection policies.

Strategies for choosing actions given Q-values or preferences.
These bridge value functions and environment interaction.
"""

import random as _random

from policywerk.primitives import scalar, vector, activations
from policywerk.primitives.random import sample_categorical

Vector = list[float]


def greedy(q_values: Vector) -> int:
    """Select the action with the highest Q-value."""
    return vector.argmax(q_values)


def epsilon_greedy(rng: _random.Random, q_values: Vector, epsilon: float) -> int:
    """With probability epsilon explore randomly, otherwise exploit."""
    if rng.random() < epsilon:
        return rng.randint(0, len(q_values) - 1)
    return vector.argmax(q_values)


def softmax_policy(rng: _random.Random, preferences: Vector, temperature: float = 1.0) -> int:
    """Sample an action from a Boltzmann distribution over preferences.

    Higher temperature → more uniform (exploratory).
    Lower temperature → more greedy (exploitative).
    """
    scaled = vector.scale(scalar.inverse(temperature), preferences)
    probs = activations.softmax(scaled)
    return sample_categorical(rng, probs)


def gaussian_policy(
    rng: _random.Random,
    mean: float,
    std: float,
    lo: float = float("-inf"),
    hi: float = float("inf"),
) -> float:
    """Sample a continuous action from a Gaussian distribution, clamped to [lo, hi].

    Used by continuous-action algorithms (PPO with continuous actions, Dreamer).
    """
    action = rng.gauss(mean, std)
    return scalar.clamp(action, lo, hi)
