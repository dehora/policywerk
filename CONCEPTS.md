# Concepts

Seven ideas that underpin everything in this project. If you understand these, you can follow any of the seven lessons. If any of them are unfamiliar, read the relevant section before diving into the code.

Each concept is also explained in the source code where it's implemented—links are provided at the end of each section.

---

## The Markov Decision Process

The Markov Decision Process (MDP) is the formal foundation of reinforcement learning. Every RL algorithm in this project—from Bellman's 1957 value iteration to DreamerV3's 2023 world model—operates within this framework.

The idea is simple: an agent lives in an environment. At each moment, the agent sees the current state (where it is, what it observes), chooses an action (move north, apply force, do nothing), and the environment responds with a reward (a number saying how good or bad that action was) and a new state. This cycle repeats until the episode ends.

```
state → agent chooses action → environment returns (reward, new state) → repeat
```

The "Markov" part means the future depends only on the current state, not on the history of how the agent got there. The grid cell you're standing on matters; the path you took to reach it doesn't. This simplification is what makes RL tractable—the agent only needs to learn a value for each state, not for every possible history.

There are two kinds of environments in this project:

- **Environment**: the agent can only interact by calling `step()`. It observes states and rewards but has no access to the environment's internal rules. This is how most RL works—learning from experience.

- **StochasticMDP**: the agent can also query `transition_probs()` to ask "if I take action A from state S, what are all the possible outcomes and their probabilities?" This is only possible when the model is fully known, and it's what Bellman's value iteration (L01) requires.

Everything above this—value functions, policies, actors, lessons—depends on these interfaces.

> Code: [`src/policywerk/building_blocks/mdp.py`](src/policywerk/building_blocks/mdp.py)

---

## Value Functions

A value function is how the agent remembers what it has learned. After thousands of episodes of experience, the agent doesn't remember individual episodes—it distills everything into a single number per state (or per state-action pair) that answers: "how much total future reward can I expect from here?"

This is a powerful idea. Instead of planning ahead by simulating future actions (expensive), the agent just looks up the value of each neighboring state and picks the best one. The value function turns a sequential decision problem into a series of one-step lookups.

There are two kinds:

- **V(s)—State value**: "how good is this state, assuming I follow my current policy from here?" Used by TD learning and critics.

- **Q(s, a)—Action value**: "how good is taking action *a* in state *s*?" Used by Q-learning and DQN. Q is more directly useful than V because it tells you which action to pick: just take the one with the highest Q-value.

In the first four lessons, values are stored in Python dicts—one entry per state (or state-action pair). This works for small environments (grids, chains) where every state can be enumerated. For larger environments (pixel observations in L05+), a neural network replaces the dict—see Function Approximation below.

> Code: [`src/policywerk/building_blocks/value_functions.py`](src/policywerk/building_blocks/value_functions.py)

---

## Function Approximation

In Lessons 01–04, Q-values live in a table: one entry per state-action pair, stored in a Python dict. The agent visits state "2,3", takes action East, and looks up `Q[("2,3", East)]`. This works when there are dozens or hundreds of states. It fails when the state is an image.

An 8×10 pixel grid has 80 values that change every step as the ball moves. The agent will almost never see the exact same pixel pattern twice. A table that has never seen this exact frame has no Q-value for it and cannot generalize from similar frames.

Function approximation replaces the table with a parameterized model—in this project, a neural network. Instead of looking up Q-values in a dict, the agent runs a forward pass:

```
Table:   Q[("2,3", East)] -> float           (one entry, one state)
Network: forward([0.0, 0.0, ..., 0.7, ..., 1.0]) -> [Q_left, Q_stay, Q_right]  (all actions at once)
```

The network's weights are shared across all states. Two pixel patterns that are similar (ball shifted one pixel, paddle in the same place) produce similar Q-values because they activate similar patterns in the hidden layer. This is generalization—the ability to make reasonable predictions about states the agent has never visited.

Training the network uses the same TD error as tabular Q-learning. The difference is how the update is applied. In the table, you adjust one entry. In the network, backpropagation traces the error through every layer and nudges every weight. A single training step changes the Q-values for all states simultaneously.

This power comes with a cost: neural networks can be unstable when combined with RL's bootstrapping (learning from your own predictions). DQN (L05) stabilizes training with three ideas: experience replay (break temporal correlation by sampling random mini-batches from a buffer of past transitions), a target network (use a frozen copy of the network for TD targets, updated periodically), and epsilon decay (shift gradually from random exploration to greedy exploitation).

> Code: [`src/policywerk/actors/dqn.py`](src/policywerk/actors/dqn.py), [`src/policywerk/building_blocks/network.py`](src/policywerk/building_blocks/network.py), [`src/policywerk/building_blocks/replay_buffer.py`](src/policywerk/building_blocks/replay_buffer.py)

---

## Exploration vs Exploitation

A policy is the agent's decision-making strategy—given what it knows about the current state, which action should it take?

The fundamental dilemma is exploration vs exploitation. If the agent always picks the action it currently thinks is best (exploitation), it might miss better options it hasn't tried yet. But if it tries random actions too often (exploration), it wastes time on things it already knows are bad. Every RL algorithm must resolve this tension.

This problem doesn't exist in supervised learning, where correct answers are given. In RL, the agent must discover good actions through trial and error, which means it must sometimes deliberately choose actions it believes are suboptimal—just to find out if it's wrong.

The strategies in this project represent different points on the spectrum:

- **greedy**: always pick the best-known action. Pure exploitation. Fast once you know the right answer, but can't discover it.

- **epsilon-greedy**: usually pick the best action, but with probability epsilon pick a random one. The simplest solution—blunt but effective. Most tabular RL methods use this (L04 Q-learning).

- **softmax**: convert action values to probabilities—higher values get higher probability, but every action has a chance. Temperature controls how peaked the distribution is: high temperature = random, low temperature = greedy. Smoother than epsilon-greedy.

- **Gaussian**: for continuous actions (not just pick-from-a-list), sample from a bell curve centered on the agent's preferred action. The width of the curve controls exploration. Used by PPO (L06) and Dreamer (L07).

In L06, the policy itself becomes a neural network, and the agent learns the policy directly rather than deriving it from Q-values. That's a fundamental shift—from "learn values, derive actions" to "learn actions directly."

> Code: [`src/policywerk/building_blocks/policies.py`](src/policywerk/building_blocks/policies.py)

---

## Discounted Returns

The "return" in RL (nothing to do with Python's `return` statement) is the total reward an agent collects from the current moment until the episode ends. It's the number the agent is ultimately trying to maximize—not the immediate reward from one step, but the cumulative payoff of an entire sequence of decisions.

The complication is that future rewards are uncertain and distant. A reward right now is guaranteed, but a reward 10 steps from now depends on what happens in between. Discounting handles this: each step into the future multiplies the reward by gamma (γ), a number between 0 and 1. With gamma=0.9, a reward 10 steps away is worth 0.9^10 ≈ 0.35 of its face value. This captures the intuition that a bird in the hand is worth more than one in the bush.

The fundamental tradeoff in return estimation:

- **Monte Carlo**: wait until the episode ends, then add up all the actual rewards with discounting. This gives the true return—no guessing—but it's noisy. Two episodes from the same state can give very different returns due to randomness.

- **TD(0) / bootstrapping**: don't wait. After one step, use the agent's own value estimate for the next state as a stand-in for all future rewards. This is biased (the estimate might be wrong) but low-variance (it doesn't depend on the randomness of an entire episode).

- **TD(λ)**: a weighted blend of all n-step returns. λ=0 gives pure TD(0), λ=1 gives pure Monte Carlo, and values in between trade off bias against variance.

- **GAE**: a practical version of the same idea, computing per-step advantages (how much better was this action than average?) with the same λ-controlled tradeoff. Used by PPO (L06).

This spectrum—from "wait and see" to "guess and go"—is one of the central ideas in reinforcement learning. Every algorithm in this project sits somewhere on it.

> Code: [`src/policywerk/building_blocks/returns.py`](src/policywerk/building_blocks/returns.py)

---

## Credit Assignment

Credit assignment is the hard problem of reinforcement learning. When the agent finally reaches the goal after 50 steps, which of those steps actually mattered? Was it step 3 (turning right at the fork) or step 47 (moving forward into the goal)? In supervised learning, every input has a clear label—you always know what the right answer was. In RL, rewards can arrive long after the decisions that caused them.

Eligibility traces solve this by maintaining a fading memory of recently visited states. Every time the agent visits a state, that state's trace increases. Every time step, all traces decay by a factor of gamma × lambda. When the agent receives a reward (or learns something new), it updates every state in proportion to its current trace—recent states get large updates, distant states get small ones.

The two parameters control the reach of credit assignment:

- **gamma** (discount factor): how much to value future vs present. Also used in return computation—it appears throughout RL.

- **lambda** (trace decay): how far back to spread credit. λ=0 means only the most recent state gets credit (like TD(0)). λ=1 means all visited states share credit equally (like Monte Carlo). Values in between give a smooth tradeoff.

There are two trace update strategies:

- **Accumulating traces**: the trace grows each time a state is revisited. A state visited 3 times has trace ≈ 3 (before decay). This rewards states that appear often.

- **Replacing traces**: the trace resets to 1 on each visit. Revisiting doesn't stack. This is often better in practice because it avoids inflating traces for states in loops.

> Code: [`src/policywerk/building_blocks/traces.py`](src/policywerk/building_blocks/traces.py)

---

## Backpropagation

Backpropagation is the algorithm that makes neural network training possible. Before Rumelhart, Hinton, and Williams published it in 1986, there was no efficient way to train networks with hidden layers—you could build them, but you couldn't teach them.

The problem backpropagation solves is credit assignment in a network: when the output is wrong, which weights are responsible? A network might have thousands of weights across many layers, and the output error is a combined result of all of them. Adjusting weights randomly would take forever. Backpropagation finds the answer in a single backward pass.

The key insight is the chain rule from calculus. If the network computes y = f(g(h(x))), then the sensitivity of y to a small change in x flows backward through each function:

```
dy/dx = dy/df × df/dg × dg/dh × dh/dx
```

In a network, each layer is one of those functions. The backward pass starts at the loss (how wrong was the output?) and works backward through each layer, asking two questions:

1. **How much did each weight in this layer contribute to the error?** This gives the weight gradients, used by the optimizer to adjust weights.

2. **How much error should be passed to the previous layer?** This gives the input gradient, used by the next layer back.

The forward pass cached intermediate values at each layer (the inputs, pre-activation sums, and post-activation outputs). The backward pass uses those cached values to compute gradients without re-running the forward computation.

Concretely, at each layer the backward pass computes:

```
delta     = incoming_error × activation_derivative(pre_activation)
dW        = outer_product(delta, layer_inputs)
db        = delta
pass_back = transpose(weights) × delta
```

> Code: [`src/policywerk/building_blocks/grad.py`](src/policywerk/building_blocks/grad.py)

---

## Glossary

Quick reference for terms that appear throughout the code.

| Term | Meaning |
|------|---------|
| **action** | What the agent does at each step (move north, apply force, etc.) |
| **actor** | The component that chooses actions (the policy). In actor-critic methods, paired with a critic |
| **advantage** | How much better an action was than average. Positive = better than expected |
| **baseline** | A reference value subtracted from the return to reduce variance without adding bias |
| **batch** | A group of training examples processed together in one update |
| **bootstrap** | Using the agent's own estimate as a stand-in for unknown future rewards |
| **critic** | The component that evaluates states or actions (the value function). Guides the actor |
| **DQN** | Deep Q-Network—Q-learning with a neural network instead of a table. Uses experience replay, a target network, and epsilon decay to stabilize training (L05) |
| **discount factor (γ)** | How much to devalue future rewards. γ=0.9 means 10 steps away is worth ~0.35× |
| **done** | Whether the episode has ended (goal reached, failure, or time limit) |
| **episode** | One complete run from start to terminal state |
| **epsilon decay** | Gradually reducing the exploration rate over training. Start random (epsilon=1.0), end mostly greedy (epsilon=0.1). Lets the agent explore early and exploit later |
| **epoch** | One pass through the entire training dataset |
| **experience replay** | Storing transitions in a buffer and training on random samples rather than the most recent experience. Breaks the correlation between consecutive frames that would otherwise destabilize training |
| **exploit** | Choose the action the agent currently believes is best |
| **explore** | Try an action that might not be best, to discover new information |
| **features** | The numbers the agent observes—coordinates, sensor values, or pixels |
| **function approximation** | Using a parameterized model (e.g. a neural network) to approximate a function (e.g. Q-values) that is too large to store in a table |
| **gradient** | How much the loss changes when a weight changes slightly. Points uphill |
| **horizon** | How many steps into the future the agent considers |
| **learning rate (α)** | How big a step to take when updating weights. Too large = unstable, too small = slow |
| **loss** | A number measuring how wrong the predictions are. Training minimizes this |
| **off-policy** | Learning from data generated by a different policy than the current one (e.g. replay buffer) |
| **on-policy** | Learning only from data generated by the current policy (e.g. PPO) |
| **policy** | The agent's strategy for choosing actions given a state |
| **Q-learning** | Off-policy TD control—updates Q(s,a) toward the max Q-value of the next state, regardless of the action actually taken |
| **return** | Total discounted reward from now until the episode ends (not a Python return) |
| **reward** | A number the environment gives after each action—the training signal |
| **SARSA** | On-policy TD control—updates Q(s,a) using the Q-value of the action actually taken next. Name comes from (S,A,R,S',A') |
| **state** | What the agent observes at one moment (grid position, sensor readings, pixels) |
| **target network** | A frozen copy of the online network used to compute stable TD targets. Updated periodically (e.g. every 20 episodes) rather than after every gradient step |
| **TD error** | The prediction error in temporal-difference learning: reward + gamma * V(s') - V(s). Drives all TD updates |
| **terminal** | A state where the episode ends (goal, failure, or absorbing state) |
| **trajectory** | A sequence of (state, action, reward) tuples from one episode |
| **transition** | One step: (state, action, reward, next_state, done) |
