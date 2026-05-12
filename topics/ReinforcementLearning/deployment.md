# Deployment/Applications - Eren


## What is Curriculum Learning?
**T:** Basic

**A:** For the complex reinforcement learning tasks, teaching an agent a behavior in a single step is hard. In that case instead of training in a single step, teaching an agent with multiple stages are required. That's why Curriculum learning is used. In Curriculum learning, the given RL task is divided into multiple stages and trained with different reward functions at each stage. For example, if we are aiming to teach a robot how to walk, we will start with teaching it to stand up, then walk slowly, and then walk at the required speed.

## How do you decide the curriculum?
**T:** Hands-on

**A:** To define a curriculum, dividing the task into correct subtasks is required. Those tasks are required to be sequential with each other. Besides that, knowing the exact points at which the policies are sufficiently trained is required. Humans can define the curriculum, or a model can understand how to define the curriculum. LLMs are one way; the other way is based on putting a threshold.

## You train a policy in simulation, how would you deploy it in the real world?
**T:** Hands-on

**A:** Simulation and real-world physics can differ due to the complexity and non-linearity of the real world. Factors like motor backlash, friction, latency, sensor noise, and actuator dynamics are hard to model accurately in simulation. For simple cases, the policy can be deployed directly, but for complex systems like multi-joint robots, adaptation is required. Common techniques include domain randomization (training across varied simulated conditions), system identification (tuning simulation parameters to match the real robot), and domain adaptation (aligning simulated and real observation distributions). Fine-tuning on a small real-world dataset or using BC bootstrapping to initialize the policy from demonstrations can also help bridge the gap.

## What is domain randomization, and what are its effects?
**T:** Basic

**A:** If we always use the same environment in simulations, our RL model can memorize the setup and fail when a simple change occurs, such as a change in light color, brightness, or the objects in the environment. To solve this problem, random variations, such as lighting and slight angle differences, can be used. Other methodologies, such as friction randomization, in such a diverse scenario, alter the visualization of objects and distractors.

## What is hierarchical RL?
**T:** Basic

**A:** For some problems, multiple controllers in different control levels are required. For example, an autonomous car needs to navigate to the correct location while also completing the current street without hitting any objects. In a hierarchy, the top one cares about the high-level problem(like navigation), and the bottom one cares about the low-level problems(controlling the car and object avoidance).

## What is residual RL?
**T:** Basic

**A:** Classical controllers(PID, MPC, etc.) are strong and fast for solving predefined problems by solving the issues deterministically. However, they are weak at handling the unseen situations. For RL, it solves the problems slower but it has a high adaptibility. To combine those advantages, Residual RL has emerged. In here, a traditional controller controls the agent, while there is additional RL that is doing the learning and adapting.

## For Safe RL, how control barrier keep it safe?
**T:** Hands-on

**A:** Safety in learning-based algorithms is crucial, especially for real-world agents. For example, if an industrial-grade, high-torque 200 kg robotic arm is controlled by RL, we have to ensure it does not take actions that harm humans or damage the environment. And especially for neural network-based controllers, there is always a possibility that they may end up taking an awkward action, since neural network policies lack formal safety guarantees. To solve this problem, multiple techniques can be used:
- Control Barrier Function: Defines a safe set in state space and mathematically constrains the policy so that actions always keep the system inside this set, guaranteeing safety by construction.
- Action Shielding: Wraps the RL policy with an external safety filter that checks every proposed action and overrides it with a safe alternative whenever it would violate safety constraints.
- Penalty Shaping: Gives large penalties to wrong actions while training RL, so the agent learns to avoid unsafe behaviors through the reward signal rather than hard constraints.


## What is Offline RL?
**T:** Basic

**A:** In some cases, it is hard to do random actions in the environment, which makes it hard to train RL models. To overcome it, we will first move the agent in the environment. Then we will collect the data and train the RL model based on that. In here, RL models learn by not interacting with the environment in real time (not online), but learn from the collected data.

## If we are able to collect data, why are we doing online RL?
**T:** Deep

**A:** Offline data only contains a fixed distribution of states/actions; the agent never sees what happens in situations the data didn't cover, leading to poor generalization and overestimation of unseen actions. Online RL lets the agent actively explore, generate new trajectories, and receive feedback on novel states, so it can correct its own mistakes and discover better policies beyond the dataset's support. For example, a self-driving car trained only on safe human driving never learns to recover from rare skids; a robot arm trained on offline demos fails on object poses absent from the data; a game agent stuck imitating logged play can't discover superior strategies its demonstrators never tried.



## What is the difference between supervised learning(e.g., behavior cloning) and offline RL
**T:** Deep

**A:** Supervised learning, like behavior cloning, just copies the actions in the dataset without caring about rewards or long-term outcomes, so it treats good and bad demonstrations equally and cannot improve beyond the demonstrator's skill level. Offline RL, on the other hand, uses the reward signal and value estimation to figure out which actions actually lead to high returns, so it can stitch together good parts of different trajectories and learn a policy better than any single behavior in the dataset. For example, behavior cloning on mixed expert and beginner driving data will imitate both, while offline RL can pick out only the expert-like actions. In robot manipulation, BC fails when the demos contain suboptimal grasps, but offline RL can learn which grasps gave higher success rewards. In healthcare, BC copies doctor decisions blindly, while offline RL evaluates which treatment choices actually led to better patient outcomes.

## What is multi-agent RL?
**T:** Basic

**A:** Some of the RL tasks require multiple agents to interact with each other. For example, if we want humanoid robots to play football, it requires multiple agents with different or the same tasks interacting with each other. These interactions can be mutual (same-team humanoid robots) or competitive (humanoid robots on different teams). In Multi-Agent RL, multiple agents learn by interacting with the environment and with each other.

## What is action shielding?
**T:** Deep

**A:** In safe RL, harmful actions must be avoided. To do that, action shielding overrides unsafe actions with safe alternatives at execution time, so no harmful action ever reaches the environment. This way, the agent can explore freely while an external safety layer guarantees that no unsafe action ever gets executed, though it is often combined with a penalty signal to discourage the policy from proposing those actions in the first place.

## What are the problems with too much domain Randomization?
**T:** Hands-on

**A:** Too much domain randomization makes the task unnecessarily hard by forcing the policy to be robust to irrelevant variations, leading to slower learning, worse asymptotic performance, and potential failure to learn at all. For example, randomizing object colors when the task only depends on shape wastes learning capacity, varying lighting conditions too wildly in a pick-and-place task makes it harder to learn grasping, and extreme physics randomization can make the simulated dynamics so different from reality that the learned policy doesn't transfer. To solve this, you can use curriculum learning to gradually increase randomization difficulty, apply automatic domain randomization that adapts the distribution based on task performance, or leverage domain knowledge to randomize only the parameters that actually affect sim-to-real transfer.


.

