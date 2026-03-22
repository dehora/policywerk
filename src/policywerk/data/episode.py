"""Level 1: Episode collection.

Utilities for running agents in environments and collecting
trajectories of experience.
"""

from policywerk.building_blocks.mdp import Environment, State, Transition, Episode

Vector = list[float]


def collect_episode(
    env: Environment,
    policy_fn,
    max_steps: int = 1000,
) -> Episode:
    """Run a policy in an environment until done or max_steps.

    policy_fn: takes a State, returns an action (int).
    """
    episode = Episode()
    state = env.reset()
    for _ in range(max_steps):
        action = policy_fn(state)
        next_state, reward, done = env.step(action)
        episode.add(Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        ))
        if done:
            break
        state = next_state
    return episode


def collect_episodes(
    env: Environment,
    policy_fn,
    num_episodes: int,
    max_steps: int = 1000,
) -> list[Episode]:
    """Collect multiple episodes."""
    return [collect_episode(env, policy_fn, max_steps) for _ in range(num_episodes)]
