# policywerk

Reinforcement learning from scratch, built piece by piece from scalar operations up to complete RL architectures. Pure Python, no frameworks—just `math` and lists.

This is the RL counterpart to [modelwerk](https://github.com/dehora/modelwerk). Same philosophy: understand the machinery by building it yourself. If you're looking for a Stable Baselines tutorial, this isn't it. If you want to know what those frameworks are doing under the hood, read on.

Blog post: [Policywerk: Building Reinforcement Learning from First Principles](https://dehora.net/journal/2026/policywerk-building-reinforcement-learning-from-first-principles)

The project follows seven landmark papers chronologically, each one building on the previous lesson's code and concepts:

| Lesson | Paper | Year | Algorithm | Environment | Status |
|--------|-------|------|-----------|-------------|--------|
| 01 | Bellman, "A Markovian Decision Process" | 1957 | Value iteration | Gridworld | Done |
| 02 | Barto, Sutton & Anderson, "Neuronlike Adaptive Elements" | 1983 | ACE/ASE actor-critic | Balance | Done |
| 03 | Sutton, "Learning to Predict by the Methods of Temporal Differences" | 1988 | TD(0) and TD(λ) | Random walk | Done |
| 04 | Watkins, "Learning from Delayed Rewards" | 1989 | Q-learning | Cliff walking | Done |
| 05 | Mnih et al., "Playing Atari with Deep Reinforcement Learning" | 2013 | DQN | Mini Breakout | Done |
| 06 | Schulman et al., "Proximal Policy Optimization Algorithms" | 2017 | PPO | Balance (continuous) | Done |
| 07 | Hafner et al., "Mastering Diverse Domains through World Models" | 2023 | DreamerV3 | Pixel point-mass | Done |

## Concepts

Seven ideas underpin everything in this project—the MDP framework, value functions, exploration vs exploitation, discounted returns, credit assignment, backpropagation, and function approximation. If any are unfamiliar, read [CONCEPTS.md](CONCEPTS.md) before diving into the code.

## Why

Most RL tutorials start with `import gymnasium`. This project starts with `1.0 + 1.0`.

Every operation is composed from scalar arithmetic up through vectors, matrices, activations, environments, value functions, policies, and finally complete agents. Nothing is hidden behind a library call. The goal is to understand what the frameworks do, not how to call them.

Each lesson introduces one new source of complexity. Lessons 01–04 use tables and scalar weights. Lessons 05–07 use neural networks built from the same primitives—the transition from tabular to neural RL happens at the L04/L05 boundary.

```
L01 (Bellman)        → Planning: exact solutions with a known model
    ↓
L02 (Barto/Sutton)   → Learning from interaction (no model)
    ↓
L03 (TD Learning)    → Bootstrapping: update from own predictions
    ↓
L04 (Q-learning)     → Off-policy control: learn the best policy while exploring
    ↓
L05 (DQN)            → Function approximation: neural nets replace tables
    ↓
L06 (PPO)            → Policy gradients: optimize the policy directly
    ↓
L07 (DreamerV3)      → World models: learn the dynamics, train in imagination
```

## Artifacts

Each lesson produces an animated visualization as its primary artifact—not a static plot, but a short animation that shows how learning unfolds. Every animation uses the same three-pane layout:

```
┌─────────────────┬─────────────────┐
│                 │                 │
│   Environment   │   Algorithm     │
│   / Trajectory  │   Internals     │
│                 │                 │
├─────────────────┴─────────────────┤
│   Training Trace (reward/loss)    │
└───────────────────────────────────┘
```

The animations answer *how learning unfolds*, not just *what was learned*. For RL, the process is often more revealing than the result.

| Lesson | What the animation shows |
|--------|--------------------------|
| 01 | Value heatmap rippling backward through the grid, sweep by sweep |
| 02 | Chaotic balance attempts becoming controlled over episodes |
| 03 | Value estimate bars shifting toward true values—TD vs Monte Carlo |
| 04 | Greedy policy arrows settling into a cliff-edge route |
| 05 | Random policy failing, reward curve climbing, trained agent clearing bricks |
| 06 | Gaussian policy distribution smoothing over updates |
| 07 | Real rollout vs imagined rollout, aligned then diverging |

## Running

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Run a lesson
uv run python lessons/01_bellman.py

# Run tests
uv run pytest tests/
```

Animations are saved to `output/` as GIFs. A poster frame (PNG) and training trace (PNG) are also exported.

## Project structure

```
src/policywerk/
  primitives/         L0: Scalar, vector, matrix ops, activations, losses
    scalar.py           Addition, multiplication, exp, log, abs, sign—the atoms
    vector.py           Dot product, element-wise ops, argmax, concat
    matrix.py           Matrix multiply, transpose, outer product, 3D tensors
    activations.py      Sigmoid, tanh, ReLU, ELU, softmax, layer norm + derivatives
    losses.py           MSE, cross-entropy, Huber, symlog, twohot + derivatives
    random.py           Seeded RNG, normal distribution, categorical sampling

  building_blocks/    L1: RL components + neural network components
    mdp.py              Environment ABC, State, Transition, Episode
    value_functions.py  Tabular V(s) and Q(s,a)
    policies.py         Epsilon-greedy, softmax, Gaussian
    traces.py           Eligibility traces
    returns.py          Discounted return, n-step, lambda-return, GAE
    replay_buffer.py    Circular experience buffer
    distributions.py    Categorical and Gaussian distributions
    neuron.py           Single neuron
    dense.py            Dense layer with forward/cache
    conv.py             Convolutional layer with forward/backward
    pool.py             Max and average pooling
    network.py          Sequential network container
    grad.py             Backpropagation + numerical gradient check
    optimizers.py       SGD, SGD with momentum, Adam
    recurrent.py        GRU layer with forward/backward

  world/              L2: Environments
    gridworld.py        5×5 deterministic grid with known dynamics (L01)
    balance.py          Simplified 1D inverted pendulum (L02)
    random_walk.py      5-state chain with known true values (L03)
    cliffworld.py       4×12 cliff walking grid (L04)
    breakout.py         8×10 mini Breakout with pixel+velocity obs (L05)
    catcher.py          16×16 pixel gridworld (general purpose)
    pointmass.py        2D continuous point-mass control (L06)
    pixel_pointmass.py  Pixel-observed point-mass wrapper (L07)

  actors/             L3: RL implementations
    bellman.py          Value iteration + policy iteration (L01)
    barto_sutton.py     ACE/ASE actor-critic (L02)
    td_learner.py       TD(0) and TD(λ) prediction (L03)
    q_learner.py        Tabular Q-learning (L04)
    dqn.py              Deep Q-network (L05)
    ppo.py              Proximal policy optimization (L06)
    dreamer.py          DreamerV3 world model (L07)

  data/               Episode collection and training metrics
  viz/                Animated visualizations (matplotlib)

lessons/              Runnable scripts—one per paper
examples/             Captured lesson outputs
papers/               Reference PDFs
tests/                Unit tests
```

## Rules

- **Python standard library only**—no numpy, torch, tensorflow, or any ML/data framework
- **matplotlib is the sole exception**—allowed for visualization only
- **Compositional layering**—each level imports only from levels below (primitives → building blocks → world → actors)
- All randomness goes through `random.py` with explicit seeds
