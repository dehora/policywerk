# policywerk

Reinforcement learning actors from scratch, built piece by piece from scalar operations up to complete architectures.

## Rules

- **Python standard library only** — no numpy, torch, tensorflow, or any ML/data framework
- **matplotlib is the sole exception** — allowed for visualization only (in `src/policywerk/viz/`)
- **Compositional layering** — each level imports only from levels below:
  - L0: primitives (scalar, vector, matrix, activations, losses, random)
  - L1: building_blocks (RL components + NN components)
  - L2: world (environments)
  - L3: actors (RL implementations)
  - data, viz are utilities available to lessons
- Types: `list[float]` for vectors, `list[list[float]]` for matrices, dataclasses for structured objects
- All randomness goes through `src/policywerk/primitives/random.py` with explicit seeds

## Running

- `uv run python lessons/01_bellman.py` — run a lesson
- `uv run pytest tests/` — run tests

## Structure

- `src/policywerk/primitives/` — scalar, vector, matrix ops, activations, losses
- `src/policywerk/building_blocks/` — RL components (MDP, value functions, policies, traces, replay buffer, returns, distributions) + NN components (dense, conv, network, backprop, optimizers)
- `src/policywerk/world/` — environments (gridworld, balance, random walk, cliffworld, breakout, catcher, pointmass, pixel_pointmass)
- `src/policywerk/actors/` — the seven RL implementations (bellman, barto_sutton, td_learner, q_learner, dqn, ppo, dreamer)
- `src/policywerk/data/` — episode collection and logging
- `src/policywerk/viz/` — matplotlib visualizations
- `lessons/` — runnable scripts, one per paper, with narrative explanations
- `examples/` — captured lesson outputs
- `tests/` — test cases
