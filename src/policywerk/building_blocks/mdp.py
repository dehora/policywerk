"""Level 1: MDP framework.

The Markov Decision Process is the formal foundation of reinforcement
learning. Every RL algorithm in this project — from Bellman's 1957
value iteration to DreamerV3's 2023 world model — operates within
this framework.

The idea is simple: an agent lives in an environment. At each moment,
the agent sees the current state (where it is, what it observes),
chooses an action (move north, apply force, do nothing), and the
environment responds with a reward (a number saying how good or bad
that action was) and a new state. This cycle repeats until the
episode ends.

    state → agent chooses action → environment returns (reward, new state) → repeat

The "Markov" part means the future depends only on the current state,
not on the history of how the agent got there. The grid cell you're
standing on matters; the path you took to reach it doesn't. This
simplification is what makes RL tractable — the agent only needs to
learn a value for each state, not for every possible history.

The key design decision in this module is the separation between two
kinds of environments:

  Environment: the agent can only interact by calling step(). It
    observes states and rewards but has no access to the environment's
    internal rules. This is how most RL works — learning from experience.

  StochasticMDP: the agent can also query transition_probs() to ask
    "if I take action A from state S, what are all the possible outcomes
    and their probabilities?" This is only possible when the model is
    fully known, and it's what Bellman's value iteration (L01) requires.

Everything above this module — value functions, policies, actors,
lessons — depends on these interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

Vector = list[float]


@dataclass
class State:
    """A state in the environment.

    features: The numbers the agent sees — grid coordinates, sensor readings,
              or pixel values depending on the environment.
    label: human-readable identifier (e.g. grid position, box index).
    """
    features: Vector
    label: str = ""


@dataclass
class Transition:
    """One step of experience: state, action, reward, next_state, done."""
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
    The agent interacts through this interface only — it never accesses
    the environment's internal state directly.
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

    The name refers to having known transition probabilities, not random
    outcomes — even deterministic environments use this when planning
    algorithms need to query 'what would happen if?'
    """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """True if this state is terminal (episode ends here)."""
        ...

    @abstractmethod
    def states(self) -> list[State]:
        """Return all states in the MDP."""
        ...

    @abstractmethod
    def transition_probs(self, state: State, action: int) -> list[tuple[State, float, float]]:
        """Return all possible transitions from (state, action).

        Returns list of (next_state, probability, reward) tuples.
        Each tuple describes one possible outcome and how likely it is.
        Probabilities must sum to 1.
        """
        ...
