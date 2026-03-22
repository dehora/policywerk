"""Level 1: Action selection policies.

A policy is the agent's decision-making strategy — given what it
knows about the current state, which action should it take?

The fundamental dilemma is exploration vs exploitation. If the agent
always picks the action it currently thinks is best (exploitation),
it might miss better options it hasn't tried yet. But if it tries
random actions too often (exploration), it wastes time on things it
already knows are bad. Every RL algorithm must resolve this tension.

This problem doesn't exist in supervised learning, where correct
answers are given. In RL, the agent must discover good actions
through trial and error, which means it must sometimes deliberately
choose actions it believes are suboptimal — just to find out if
it's wrong.

The strategies here represent different points on the spectrum:

  greedy: always pick the best-known action. Pure exploitation.
    Fast once you know the right answer, but can't discover it.

  epsilon_greedy: usually pick the best action, but with probability
    epsilon pick a random one. The simplest solution — blunt but
    effective. Most tabular RL methods use this (L04 Q-learning).

  softmax_policy: convert action values to probabilities — higher
    values get higher probability, but every action has a chance.
    Temperature controls how peaked the distribution is: high
    temperature ≈ random, low temperature ≈ greedy. Smoother than
    epsilon-greedy.

  gaussian_policy: for continuous actions (not just pick-from-a-list),
    sample from a bell curve centered on the agent's preferred action.
    The width of the curve controls exploration. Used by PPO (L06)
    and Dreamer (L07).

In L06, the policy itself becomes a neural network, and the agent
learns the policy directly rather than deriving it from Q-values.
That's a fundamental shift — from "learn values, derive actions"
to "learn actions directly."
"""

import random as _random

from policywerk.primitives import scalar, vector, activations
from policywerk.primitives.random import sample_categorical

Vector = list[float]


def greedy(q_values: Vector) -> int:
    """Select the action with the highest Q-value."""
    return vector.argmax(q_values)


def epsilon_greedy(rng: _random.Random, q_values: Vector, epsilon: float) -> int:
    """With probability epsilon, try a random action (explore).
    Otherwise, pick the best-known action (exploit)."""
    if rng.random() < epsilon:
        return rng.randint(0, len(q_values) - 1)
    return vector.argmax(q_values)


def softmax_policy(rng: _random.Random, q_values: Vector, temperature: float = 1.0) -> int:
    """Convert action scores into probabilities — higher-scored actions are more
    likely to be chosen, but not guaranteed.

    Higher temperature → more uniform (exploratory).
    Lower temperature → more greedy (exploitative).
    """
    scaled = vector.scale(scalar.inverse(temperature), q_values)
    probs = activations.softmax(scaled)
    return sample_categorical(rng, probs)


def gaussian_policy(
    rng: _random.Random,
    mean: float,
    std: float,
    lo: float = float("-inf"),
    hi: float = float("inf"),
) -> float:
    """Sample a continuous action from a Gaussian (bell curve) distribution, clamped to [lo, hi].

    Used by continuous-action algorithms (PPO with continuous actions, Dreamer).
    """
    action = rng.gauss(mean, std)
    return scalar.clamp(action, lo, hi)
