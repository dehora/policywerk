"""Level 1: MDP framework.

The environment protocol that every world and actor depends on.
Defines the Markov Decision Process interface — states, actions,
transitions, and the contract between agent and environment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Vector = list[float]


@dataclass
class State:
    """A state in the environment.

    features: numeric representation the agent can observe.
    label: human-readable identifier (e.g. grid position, box index).
    """
    features: Vector
    label: str = ""


@dataclass
class Transition:
    """One step of experience: (s, a, r, s', done)."""
    state: State
    action: int
    reward: float
    next_state: State
    done: bool


@dataclass
class Episode:
    """A complete sequence of transitions from start to terminal."""
    transitions: list[Transition] = field(default_factory=list)

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    @property
    def total_reward(self) -> float:
        return sum(t.reward for t in self.transitions)

    def __len__(self) -> int:
        return len(self.transitions)


class Environment(ABC):
    """Base environment protocol.

    Every environment must implement reset() and step().
    The agent interacts through this interface only.
    """

    @abstractmethod
    def reset(self) -> State:
        """Reset to initial state and return it."""
        ...

    @abstractmethod
    def step(self, action: int) -> tuple[State, float, bool]:
        """Take an action, return (next_state, reward, done)."""
        ...

    @abstractmethod
    def num_actions(self) -> int:
        """Number of discrete actions available."""
        ...


class StochasticMDP(Environment):
    """Environment with known transition dynamics.

    Extends Environment with transition_probs() for dynamic programming
    methods that need the full model (e.g. value iteration).
    """

    @abstractmethod
    def states(self) -> list[State]:
        """Return all states in the MDP."""
        ...

    @abstractmethod
    def transition_probs(self, state: State, action: int) -> list[tuple[State, float, float]]:
        """Return all possible transitions from (state, action).

        Returns list of (next_state, probability, reward) tuples.
        Probabilities must sum to 1.
        """
        ...
