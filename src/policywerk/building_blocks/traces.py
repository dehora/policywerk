"""Level 1: Eligibility traces.

Credit assignment is the hard problem of reinforcement learning.
When the agent finally reaches the goal after 50 steps, which of
those steps actually mattered? Was it step 3 (turning right at
the fork) or step 47 (moving forward into the goal)? In supervised
learning, every input has a clear label — you always know what the
right answer was. In RL, rewards can arrive long after the decisions
that caused them.

Eligibility traces solve this by maintaining a fading memory of
recently visited states. Every time the agent visits a state, that
state's trace increases. Every time step, all traces decay by a
factor of gamma × lambda. When the agent receives a reward (or
learns something new), it updates every state in proportion to
its current trace — recent states get large updates, distant
states get small ones.

The two parameters control the reach of credit assignment:

  gamma (discount factor): how much to value future vs present.
    Also used in return computation — it appears throughout RL.

  lambda (trace decay): how far back to spread credit.
    lambda=0: only the most recent state gets credit (like TD(0)).
    lambda=1: all visited states share credit equally (like Monte Carlo).
    Values in between give a smooth tradeoff.

There are two trace update strategies:

  Accumulating traces (visit): the trace grows each time a state
    is revisited. A state visited 3 times has trace ≈ 3 (before
    decay). This rewards states that appear often.

  Replacing traces (replace): the trace resets to 1 on each visit.
    Revisiting doesn't stack. This is often better in practice
    because it avoids inflating traces for states in loops.

Eligibility traces are used by TD(λ) in L03 and the ACE/ASE
actor-critic architecture in L02.
"""

from policywerk.primitives import scalar


class EligibilityTrace:
    """Trace that remembers which states were recently visited.

    Used by TD(λ) and the Barto/Sutton ACE/ASE architecture.
    """

    def __init__(self, gamma: float, lam: float):
        """
        gamma: discount factor — how much to care about future versus present.
        lam: trace decay rate — how far back to spread credit.
        """
        self._traces: dict[str, float] = {}
        self._gamma = gamma
        self._lam = lam

    def get(self, state_label: str) -> float:
        return self._traces.get(state_label, 0.0)

    def visit(self, state_label: str) -> None:
        """Mark a state as visited — accumulating trace.

        The trace grows each revisit, stacking credit.
        """
        self._traces[state_label] = scalar.add(self.get(state_label), 1.0)

    def replace(self, state_label: str) -> None:
        """Mark a state as visited — replacing trace (resets to 1, doesn't stack)."""
        self._traces[state_label] = 1.0

    def decay(self) -> None:
        """Decay all traces by gamma * lambda."""
        factor = scalar.multiply(self._gamma, self._lam)
        for key in self._traces:
            self._traces[key] = scalar.multiply(self._traces[key], factor)

    def reset(self) -> None:
        """Clear all traces (start of new episode)."""
        self._traces.clear()

    def all_traces(self) -> dict[str, float]:
        return dict(self._traces)
