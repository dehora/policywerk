"""Level 1: Experience replay buffer.

Circular buffer that stores transitions and serves random mini-batches.
Breaks temporal correlation in training data — the key insight of DQN.
"""

import random as _random
from dataclasses import dataclass

from policywerk.building_blocks.mdp import Transition

Vector = list[float]


class ReplayBuffer:
    """Fixed-size circular buffer of transitions."""

    def __init__(self, capacity: int):
        self._buffer: list[Transition] = []
        self._capacity = capacity
        self._position = 0

    def add(self, transition: Transition) -> None:
        """Store a transition, overwriting the oldest if full."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._position] = transition
        self._position = (self._position + 1) % self._capacity

    def sample(self, rng: _random.Random, batch_size: int) -> list[Transition]:
        """Sample a random mini-batch of transitions."""
        indices = [rng.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)
