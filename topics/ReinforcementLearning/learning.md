# Learning 

## What is temporal-difference (TD) learning, and how is it different from Monte Carlo (MC) learning?
**T:** Basic
**A:** 
Both are model-free evaluation algorithms. TD learning updates value estimates based on other learned estimates (bootstrapping) after every step. MC learning requires a complete episode to finish before updating values based on actual accumulated returns. 

Consequently, TD learning has lower variance but higher bias, whereas MC learning has high variance but zero bias.

## What is the difference between value iteration and policy iteration?
**T:** Basic
**A:** Policy Iteration is a two-step process: it explicitly evaluates a fixed policy until convergence, then greedily improves the policy, repeating until optimal. 

Value Iteration collapses this process by directly applying the Bellman optimality equation to update state-value estimates in one sweep, extracting the optimal policy only at the end.

## What is Q-learning?
**T:** Basic
**A:** Q-learning is an off-policy TD control algorithm. It iteratively approximates the optimal action-value function $Q(s,a)$, representing the expected cumulative reward of taking action $a$ in state $s$. Unlike a static transition table, it updates these values dynamically using the Bellman equation, independently of the behavioral policy used for exploration.

## What is the difference between A2C and A3C?
**T:** Basic
**A:** Both are Actor-Critic architectures utilizing parallel workers. A3C (Asynchronous) allows independent workers to asynchronously update a global network, which can lead to parameter staleness. A2C (Advantage Actor-Critic) is the synchronous variant; a coordinator waits for all workers to complete their rollout segments before averaging the gradients to perform a unified update, making it more stable and GPU-efficient.

## What is SARSA, and what is its primary disadvantage?
**T:** Basic
**A:** SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm. It updates $Q(s,a)$ based on the action actually selected by the current behavioral policy for the next state.

*Disadvantage:* Because it is on-policy, its convergence to the optimal policy is slower than Q-learning; it safely incorporates exploration penalties (like the randomness in an $\epsilon$-greedy policy) into its value updates, which can drag down performance during training.

## What is a POMDP, and how is it solved?
**T:** Deep
**A:** A Partially Observable Markov Decision Process (POMDP) occurs when an agent cannot perfectly observe the underlying environment state, only receiving observation signals (effects). For instance, real-world mid-air aircraft collision avoidance systems (like ACAS X) are modeled this way.

It is typically solved by converting it into an MDP where the ``state'' is replaced by a belief state (a probability distribution over all possible true states) or by using recurrent neural networks (RNNs) to compress the entire history of past observations into a hidden state vector.

## What is TRPO, and how did it evolve into PPO?
**T:** Basic
**A:** Trust Region Policy Optimization (TRPO) ensures monotonic policy improvement by mathematically constraining the policy update step size. It uses a hard KL-divergence constraint to guarantee the new policy stays within a ``trust region'' of the old policy.

TRPO relies on computationally expensive second-order optimizations (Fisher Information Matrix). Proximal Policy Optimization (PPO) simplifies this by subtracting the constraint penalty directly or, more commonly, using a clipped first-order surrogate objective function that penalizes excessively large updates, achieving similar stability with far less compute.

## How does Soft Actor-Critic (SAC) differ from A2C, and why is it called ``soft''?
**T:** Deep
**A:** While A2C is typically an on-policy algorithm, SAC is an off-policy Actor-Critic method.

**Why "soft"?** It relies on maximum entropy RL. A temperature-scaled entropy term is added to the reward function. The policy is optimized to maximize both the expected return and the entropy (randomness) of its actions, acting as a ``soft'' maximum that inherently drives stable exploration.

## Why does SAC utilize two Q-value functions?
**T:** Deep
**A:** Approximating Q-values in continuous action spaces frequently suffers from overestimation bias. To mitigate this, SAC maintains two separate Q-networks and takes the minimum of their outputs when computing the temporal difference target. This pessimistic bounding strategy was adapted from Double DQN (DDQN).

## What is Hindsight Experience Replay (HER)?
**T:** Deep
**A:** HER is a buffer technique designed for sparse-reward environments. If an agent fails to reach its target goal in an episode, HER retrospectively relabels the actual state achieved at the end of the trajectory as a new pseudo-goal. By storing this modified trajectory, the agent receives a positive reward signal and learns from its failures by treating them as successes for alternative objectives.

---
