# Theory - Som


## What is the Bellman Equation?
**T:** Basic  
**A:** The Bellman Equation expresses the value of a state recursively — the value of being in a state equals the immediate reward you get from that state plus the discounted value of the next state you end up in. Formally, for a state $s$:

$$V(s) = \mathbb{E}\left[r + \gamma V(s')\right]$$

where $r$ is the immediate reward, $\gamma \in [0,1]$ is the discount factor controlling how much future rewards matter, and $s'$ is the next state. The key insight is that you don't need to sum over all future time steps explicitly — the recursive structure lets the value of $s'$ already encode everything beyond it.

## What is a Markov Chain?
**T:** Basic  
**A:** A Markov Chain is a stochastic process over a set of states where the probability of transitioning to the next state depends *only* on the current state — not on any history of how you got there. This is called the **Markov property**: $P(s_{t+1} \mid s_t, s_{t-1}, \ldots) = P(s_{t+1} \mid s_t)$. There are no actions and no rewards; it is purely a model of how a system evolves over time. This makes it the simplest building block for more complex frameworks like MDPs.

## What is a Markov Decision Process (MDP)?
**T:** Basic  
**A:** An MDP extends a Markov Chain by introducing an **agent** that takes **actions** and receives **rewards**. It is defined by the tuple $(S, A, P, R, \gamma)$ — states, actions, transition probabilities, a reward function, and a discount factor. The agent's goal is to find a **policy** $\pi(a \mid s)$ that maximizes its expected cumulative discounted reward. The Markov property is retained: the next state and reward depend only on the current state and action, not the full history. This is a deliberate simplification — in practice it's rarely perfectly true, but it makes the problem tractable while still capturing the essential structure of sequential decision-making.

## What is a Q-value vs. a State Value?
**T:** Basic  
**A:** Both are measures of long-term return, but they condition on different things:
 
- **State Value** $V^\pi(s)$: the expected cumulative reward starting from state $s$ and following policy $\pi$ thereafter. It tells you how good it is to *be* in a state.
- **Q-value (Action-Value)** $Q^\pi(s, a)$: the expected cumulative reward starting from state $s$, taking a *specific action* $a$ first, and then following policy $\pi$. It tells you how good it is to take a particular action in a state.
The relationship between them is: $V^\pi(s) = \sum_a \pi(a \mid s)\, Q^\pi(s, a)$ — the state value is simply the Q-values averaged over actions according to the policy. Q-values are more directly useful for control, since comparing $Q(s, a)$ across actions tells you which action to pick.


## Where can we NOT use RL?
**T:** Basic  
**A:** RL breaks down in **non-stationary environments** — settings where the underlying dynamics or reward structure change over time such that the Markov property no longer holds. If the rules of the environment keep shifting, the value estimates the agent has learned become stale and unreliable, since they were computed under assumptions that no longer apply. More broadly, RL is also impractical when environment interaction is extremely expensive or dangerous, when rewards are completely unobservable, or when the state space is so poorly defined that meaningful generalization is impossible.

## What is the difference between Model-Based and Model-Free RL?
**T:** Basic  
**A:** The distinction is whether the agent has access to — or explicitly learns — a model of the environment's transition dynamics $P(s' \mid s, a)$:
 
- **Model-Based:** The agent uses a known or learned transition model to plan ahead (e.g., by simulating trajectories) before acting. This can be sample-efficient but is vulnerable to model errors — if the learned model is wrong, the policy optimized against it can fail badly in the real environment. Examples include MCTS/AlphaZero, MPC, LQR, and PILCO.
- **Model-Free:** The agent learns purely from real experience, updating its value function or policy directly from observed transitions without ever explicitly modeling $P(s' \mid s, a)$. This avoids model bias and scales better to high-dimensional, complex environments, but typically requires far more environment interactions.

## What is the difference between Optimal Control and RL?
**T:** Basic  
**A:** Both aim to find a policy that maximizes some notion of cumulative reward, but they differ in what they assume is known upfront. **Optimal control** typically takes the mathematical model of the environment (dynamics, cost function) as given and solves for the optimal policy analytically or numerically — no learning from data is required. **RL**, by contrast, assumes the agent does *not* have access to the true model and must discover an effective policy through trial-and-error interaction with the environment. In practice, the boundary has blurred — model-based RL learns a dynamics model and then applies optimal control on top of it — but the conceptual distinction is known vs. unknown environment dynamics.

## What is the difference between On-Policy and Off-Policy learning?
**T:** Basic  
**A:** The distinction is about *whose* data is being used to update the policy:
 
- **On-Policy:** The agent can only learn from transitions generated by the policy it is currently trying to improve. Every time the policy updates, old data becomes invalid and must be discarded. SARSA and PPO are canonical examples.
- **Off-Policy:** The agent can learn from data generated by a *different* (possibly older) policy, or from a replay buffer of past experience. This makes off-policy methods more sample-efficient since data can be reused, but introduces additional complexity (e.g., importance sampling corrections). Q-learning and DQN are canonical examples


## What is the difference between Online and Offline RL?
**T:** Basic  
**A:** The distinction is about whether the agent can interact with the environment during training:
 
- **Online RL:** The agent actively collects new experience by interacting with the environment throughout training. It can explore, correct mistakes, and continuously refine its behavior.
- **Offline RL:** The agent trains entirely on a fixed, pre-collected dataset — there is no live environment interaction. This is critical in domains where data collection is expensive or dangerous (e.g., healthcare, robotics), but introduces the challenge of **distributional shift**: the agent may encounter state-action pairs during training that are poorly represented in the dataset, leading to unreliable value estimates.

## What are the main Types of RL Algorithms?
**T:** Basic  
**A:** RL algorithms are broadly categorized into three families based on what they learn:
 
- **Value-Based:** The agent learns a value function ($V$ or $Q$) and derives its policy implicitly by always selecting the highest-valued action. The policy itself is never parameterized directly. Examples: DQN, Q-learning.
- **Policy Gradient:** The agent directly parameterizes and optimizes the policy $\pi_\theta$ using gradient ascent on expected return, without necessarily maintaining an explicit value function. This handles continuous action spaces naturally. Example: PPO, REINFORCE.
- **Actor-Critic:** A hybrid that combines both. The **Actor** is the policy that selects actions; the **Critic** is a learned value function that evaluates those actions and provides a lower-variance training signal for the actor. Most modern deep RL algorithms (PPO with a value head, SAC, A3C) fall into this category.


## What is the Exploration vs. Exploitation Trade-off?
**T:** Basic  
**A:** At every step, the agent faces a fundamental dilemma:
 
- **Exploitation:** Acting greedily with respect to current knowledge — choosing the action believed to yield the highest reward. This maximizes short-term performance but risks getting stuck in a suboptimal policy if the agent's estimates are wrong.
- **Exploration:** Deliberately trying actions that may not seem optimal in order to gather new information about the environment. This sacrifices immediate reward for the possibility of discovering better long-term strategies.
Balancing the two is non-trivial. Common strategies include $\epsilon$-greedy (explore randomly with probability $\epsilon$), softmax action selection, and principled approaches like UCB (Upper Confidence Bound) or entropy bonuses in policy gradient methods.

 
## What is the Credit Assignment Problem?
**T:** Deep  
**A:** When an agent receives a reward, it must determine which of the many actions it took in the past were actually responsible for that outcome. This is the **credit assignment problem**. It is hard for two reasons: (1) actions have delayed consequences, so the reward signal arrives long after the relevant decision was made; and (2) rewards are often **sparse** — the agent may take hundreds of steps before receiving any feedback at all. Temporal Difference methods partially address this by propagating value estimates backward through time, and techniques like **reward shaping** and **hindsight experience replay** are explicitly designed to make credit assignment more tractable.

## What is Generalization in RL?
**T:** Basic  
**A:** Generalization refers to the agent's ability to apply learned behavior successfully to states it has never encountered during training. This is harder in RL than in supervised learning because the agent's training distribution is determined by its own policy — if the agent never visits certain states, it has no data to learn from them. Poor generalization manifests as agents that perform well on their training environment but fail when the environment is even slightly perturbed (different visual textures, layout variations, etc.). Approaches to improve generalization include domain randomization, data augmentation, and learning compact state representations that abstract away irrelevant details.

---
