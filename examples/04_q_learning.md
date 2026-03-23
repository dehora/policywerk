# Lesson 4: Q-Learning (Watkins, 1989)

In 1989, Chris Watkins showed how an agent could learn the optimal policy while exploring—without needing a model and without being constrained to follow its current best guess. This extends TD learning from prediction (Lesson 03) to control.

```
uv run python lessons/04_q_learning.py
```

## The Cliff World

A 4x12 grid. Start at bottom-left, goal at bottom-right. The cliff runs along the bottom edge between start and goal.

```
. . . . . . . . . . . .
. . . . . . . . . . . .
. . . . . . . . . . . .
S C C C C C C C C C C G    S=start, C=cliff(-100), G=goal
```

Actions: N, E, S, W. Normal steps cost -1. Stepping on the cliff costs -100 and teleports back to start (the episode continues—the agent must try again). Reaching the goal ends the episode with reward 0.

The optimal path runs just above the cliff (row 1 in the grid, row 2 in cartesian coordinates): up from start, 11 steps east, down to goal. That is 13 steps, total reward -12 (12 steps at -1, goal step at 0).

## Q-Learning: Off-Policy Control

Lesson 03 learned V(s)—how good each state is. But V(s) does not say which action to take. Q-learning learns Q(s, a)—how good each action is in each state. Once you have Q-values for every action, pick the highest.

The update after each step (s, a, r, s'):

```
Q(s, a) += alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
```

The crucial word is "max." The agent asks "what is the best I could do from s'?" regardless of what it actually does next.

Concrete example. The agent is at (2, 1), just above the cliff. It goes East to (2, 2), getting reward -1.

```
Q((2,1), East) += 0.5 * [-1 + 1.0 * max_a Q((2,2), a) - Q((2,1), East)]

If max_a Q((2,2), a) = -10 and Q((2,1), East) = -12:
  TD error = -1 + (-10) - (-12) = 1
  Q((2,1), East) += 0.5 * 1 = +0.5 -> Q = -11.5
```

The value went up because -10 is better than the old estimate of -12. The "max" means Q-learning learns what the optimal policy would do, even while the agent explores with epsilon-greedy (with probability epsilon, take a random action instead of the best). This is off-policy learning.

## SARSA: On-Policy Alternative

SARSA uses the action actually taken next, not the best:

```
Q(s, a) += alpha * [r + gamma * Q(s', a_next) - Q(s, a)]
```

The name comes from the five values used in each update: State, Action, Reward, State', Action'.

On the cliff world, this matters. With epsilon=0.1, the agent sometimes takes a random action. Near the cliff edge, a random action might step south into the cliff (-100 penalty). SARSA accounts for this risk because it uses the action actually taken (which might be random). Q-learning ignores the risk because it uses the best action (which would never step into the cliff).

Q-learning learns the cliff-edge path—optimal if the agent follows the greedy policy perfectly. SARSA learns a safer path one row up—optimal given that the agent is still exploring. Both are correct for the question they answer.

## Training Results

```
Average reward per 50 episodes:
  Episodes   0- 49:  Q-learning  -117.2   SARSA  -108.2
  Episodes 100-149:  Q-learning   -56.1   SARSA   -24.5
  Episodes 450-499:  Q-learning   -43.6   SARSA   -27.7
```

Q-learning's average training reward stays worse than SARSA's even after 500 episodes. This is the on-policy/off-policy trade-off in a nutshell: the better the learned policy, the worse the training performance when exploration adds noise. Q-learning's cliff-edge path is optimal when executed perfectly, but maximally punished by random exploration. SARSA's safer path is suboptimal but more forgiving of mistakes.

The early episodes are volatile. Episode 1 gets reward < -1000 in 470 steps—the agent falls off the cliff repeatedly and gets teleported back. By episode 3 it is down to -33 in 34 steps. But occasional relapses (episode 6 at -428, episode 7 at -248) show that epsilon exploration near the cliff is genuinely dangerous even after the agent mostly knows what to do.

```
Greedy evaluation (no exploration):
  Q-learning: 13 steps, reward -12, avg row 2.1
  SARSA:      17 steps, reward -16, avg row 0.9
```

With exploration turned off, Q-learning's policy is strictly better: 13 steps (optimal) versus SARSA's 17. SARSA's lower average row confirms it learned the safer, longer route.

## Artifacts

### Q-Learning Trajectories

![Q-learning animation](img/04_q_learning_artifact.gif)

Policy arrows start chaotic and settle into a clear rightward route along row 2. The greedy replay at the end walks through Q-learning's learned path step by step, then SARSA's safer route, highlighting each cell as the agent lands on it.

### Q-Learning vs SARSA Policies

![Q-learning vs SARSA](img/04_q_learning_poster.png)

Q-learning's arrows point east along the row just above the cliff. SARSA's arrows take a wider path further from the edge.

### Reward Comparison

![Reward comparison](img/04_q_learning_trace.png)

SARSA achieves better average reward during training (fewer cliff falls) but Q-learning's greedy policy is optimal when executed without exploration.

## Next

Q-learning uses tables: one entry per state-action pair. But what if the state space is too large to tabulate—like pixel observations? In Lesson 05, DQN replaces the table with a neural network: Q-learning at scale.
