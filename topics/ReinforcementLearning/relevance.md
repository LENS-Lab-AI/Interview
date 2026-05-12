# RL Related

## What is thomspon sampling?

**T:** Basic


**A:** one with best mean (bandit algorithm that samples a reward model from the posterior and picks the action that looks best under that sample.)
## What is UCB, how is it different from thompson sampling

**T:** Basic

**A:** UCB, or Upper Confidence Bound, is another bandit algorithm that chooses actions according to an optimistic estimate of their value. Instead of using only the empirical mean reward, it adds an uncertainty bonus, so the action score is roughly “mean plus uncertainty.” This encourages the algorithm to try actions that are either known to be good or still uncertain. The main difference from Thompson Sampling is that UCB is deterministic once the estimates are fixed and explicitly builds in optimism through confidence intervals, while Thompson Sampling is randomized and explores by sampling from a posterior distribution over rewards.


## Why is UCB and Thompson used commonly?

**T:** Hands-on
**A:** They are widely used because they are simple, principled, and effective ways to trade off exploration and exploitation. More importantly, both come with strong theoretical regret guarantees, which makes them attractive in settings where we want both practical performance and mathematical justification.

## What theoretical guarantees you can provide with UCB and Thompson samplimg?

**T:** Deep

**A:** The main theoretical guarantee for both UCB and Thompson Sampling is that their regret grows sublinearly with the number of rounds. Regret measures how much reward is lost compared with always choosing the optimal action. If regret is sublinear, then the average regret per round goes to zero as the number of rounds increases, which means the algorithm asymptotically learns to act optimally.


## What is Bayesian Optimization?

**T:** Basic

**A:** Bayesian Optimization is a framework for optimizing expensive black-box functions when evaluations are costly and gradients may be unavailable. It works by fitting a surrogate model, often a Gaussian process, to approximate the unknown objective function, and then using an acquisition function to decide where to evaluate next. The surrogate captures both predicted performance and uncertainty, while the acquisition function balances exploring uncertain regions and exploiting areas that already look promising. This makes Bayesian Optimization especially useful in settings like hyperparameter tuning, robotics experiments, and scientific optimization where each evaluation may be expensive or time-consuming.

## What are the limitations of BO, what are the alternatives?

**T:** Deep

**A:** Problems - gaussian is expensive  use RL when n is high

## What is Krigging?

**T:** Basic

**A:** Kriging is a statistical interpolation method originally developed in geostatistics, and it is closely related to Gaussian process regression. The basic idea is to predict the value of an unknown function at a new point by using nearby observed data points together with an explicit model of spatial or functional correlation. In machine learning and optimization, Kriging is often used as a surrogate model because it provides not only a mean prediction but also an uncertainty estimate, which is very useful for deciding where to sample next. In fact, classical Bayesian Optimization with Gaussian processes can be viewed as using a Kriging-style predictive model.

## What is MCTS?

**T:** Basic

**A:** Monte Carlo Tree Search is a planning algorithm that builds a search tree by simulating many rollouts and using them to estimate good actions.

## What is the weakness of MCTS?

**T:** Hands-on

**A:** The main weakness of MCTS is that it can be computationally expensive, especially when the branching factor is large or when accurate rollouts require a strong simulator. Its performance also depends heavily on the quality of the simulation model and rollout policy; if the simulator is poor or the horizon is very long, the estimates can become noisy and misleading. In very large, continuous, or partially observed environments, naive MCTS may struggle because the tree grows too quickly and many states are rarely revisited. As a result, MCTS is often most effective when combined with good heuristics, learned value functions, or domain-specific structure.

## What is Model predictive control

**T:** Basic

**A:** Model Predictive Control (MPC) is a control strategy where, at every time step, you use a model of the system to predict how the system will evolve over a finite future horizon, solve an optimization problem to find the best sequence of control actions, apply only the first action, and then repeat the process at the next step using new state feedback.

## Compare MPC with RL

**T:** Hands-on

**A:** MPC plans online using an explicit dynamics model, while RL learns a policy from experience that can act quickly at test time.

## What is MPPI?

**T:** Basic

**A:** Model Predictive Path Integral control, is a sampling-based MPC method that updates controls by weighting noisy trajectories by their costs.


## How can we use LLM as policy, how does it learn implicitly

**T:** Basic

**A:** An LLM can act as a policy by mapping observations or text state descriptions to actions, and it learns implicitly by absorbing behavioral patterns from pretraining or fine-tuning data.

---
