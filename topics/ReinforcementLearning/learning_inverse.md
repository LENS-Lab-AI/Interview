# Learning (Inverse/Preference) - Xin

## What is inverse RL?
**T:** Basic

**A:** 
Inverse reinforcement learning (IRL) asks the reverse question of RL: instead of being given a reward function and solving for an optimal policy, you observe expert behavior and try to infer a reward function under which that behavior is optimal or near-optimal. The classic formulation is due to Ng and Russell (2000), and Abbeel and Ng (2004) turned this into apprenticeship learning, where the goal is to recover behavior that matches an expert without requiring the expert to hand-design the reward.

The deep intuition is that demonstrations tell you what the expert prefers, not just what action they took at one state. IRL is useful when writing the reward explicitly is hard, but showing good behavior is easy—for example driving, robotics, or human preference learning.

## What's the challenge of inverse RL?
**T:** Basic

**A:** 
The central difficulty is that IRL is ill-posed: many different reward functions can rationalize the same observed trajectories. This is not a bug of one algorithm; it is a structural identifiability problem. If all you observe is behavior, then several rewards may induce the same optimal policy, especially up to shaping-like transformations or feature equivalences.

A second challenge is that IRL is effectively a bi-level problem: for each candidate reward, you must solve an RL problem to see what policy it induces, then compare that policy to the expert. So learning the reward and solving control are coupled, which is why IRL can be unstable or expensive.

A third challenge is distribution mismatch. The demonstrations come from the expert’s state-action distribution, but once the learner acts on its own, it visits different states, and small reward misspecification can compound. That is one reason apprenticeship learning focuses on matching feature expectations or occupancy behavior rather than only copying actions locally.

## How to solve the multiple-reward problem?
**T:** Basic

**A:** 
A standard answer is: add an inductive principle that selects one reward among all rewards consistent with the demonstrations. Different IRL methods differ mainly in that principle.

In Apprenticeship Learning via IRL, Abbeel and Ng assume the reward is linear in known features and then search for a policy whose feature expectations match the expert’s. This avoids needing to recover the one “true” reward exactly; it is enough to recover a reward-equivalent policy.

In Maximum Entropy IRL, Ziebart resolves reward ambiguity by choosing, among all trajectory distributions that match the demonstrations, the one with maximum entropy. Intuitively: do not assume extra structure that the data did not justify. This gives a probabilistic model over trajectories and is usually the cleanest interview answer to “why does MaxEnt help?”: it turns an underdetermined inverse problem into a well-posed one by preferring the least-committed explanation.

**In short**

- IRL: infer reward from expert behavior
- classic papers: Ng–Russell 2000; Abbeel–Ng 2004
- hard because many rewards explain same demos
- also hard because each reward guess requires solving RL
- fix ambiguity with extra principle: feature matching, max margin, maximum entropy

## What is InstructGPT and what did they use?
**T:** Basic

**A:** 
InstructGPT is not a new base architecture; it is a GPT-3-family model fine-tuned to follow instructions better using human feedback. The key point of the paper is that next-token pretraining alone does not make a model reliably aligned with user intent, so OpenAI added a post-training alignment pipeline.

The pipeline in the paper has three stages. First, collect demonstrations and do supervised fine-tuning (SFT). Second, collect rankings of multiple model outputs and train a reward model. Third, optimize the policy with RLHF using PPO, with a KL-style penalty to keep the model from drifting too far from the reference policy. The prompts came from both labeler-written tasks and real prompts submitted through the OpenAI API.

The punchline result is interview-worthy: a 1.3B InstructGPT model was preferred by human raters over the much larger 175B GPT-3 on their prompt distribution. That made the paper influential because it showed that better alignment can dominate brute-force scale on user-perceived usefulness.

**In short**

- InstructGPT = GPT-3 aligned to follow instructions
- used SFT on demonstrations
- then reward model on ranked outputs
- then PPO-based RLHF with KL constraint
- prompts came from labelers and real API users

## What is GRPO? will it be unstable from the average? will it take forever to converge?
**T:** Basic

**A:** 
GRPO stands for Group Relative Policy Optimization. It was introduced in DeepSeekMath as a PPO variant that removes the separate critic/value model and instead estimates advantage from group-relative rewards—that is, compare multiple sampled outputs for the same prompt and normalize them relative to the group. The practical motivation in the paper is memory and training-efficiency savings over standard PPO.

So the answer to “is it unstable because of the average?” is: it can be. The group-average baseline is simple and effective when rewards are clean and mostly relative, especially in verifiable domains like math. But this same simplicity can create optimization bias. Later analyses found that GRPO can introduce length bias, and with richer ordinal rewards its group-average baseline can even assign positive advantage to failed trajectories, reinforcing wrong behavior.

Does it “take forever to converge”? Not inherently. In practice it can converge well enough to be useful, and DeepSeekMath introduced it precisely because it trains efficiently. But convergence speed depends heavily on verifier quality, reward noise, group size, exploration, and the base model. With noisy verifiers, RLVR-style training can slow down dramatically or even go in the wrong direction if false positives/false negatives are bad enough.

A good oral answer is: GRPO trades the stability of an explicit critic for the simplicity of a relative group baseline. That is often a good trade in LLM reasoning with verifiable rewards, but it is not a free lunch.

**In short**

- GRPO = PPO variant with no separate critic
- uses within-group relative rewards as baseline
- attractive because cheaper in memory/implementation
- can be unstable with noisy or ordinal rewards
- convergence depends on reward quality and group design, not just on “the average”

## What is DPO?
**T:** Basic

**A:** 
**DPO** stands for **Direct Preference Optimization**. Its key idea is that, under the standard KL-constrained RLHF setup, you can derive the optimal policy in closed form and train it **directly from preference pairs** with a simple classification-style loss, instead of first fitting an explicit reward model and then running RL. The DPO paper’s slogan is exactly this: the language model is “secretly a reward model.” 

So conceptually, DPO says: if you have pairwise preferences like “response A is better than response B,” you can optimize the policy **offline and directly** rather than doing the usual reward-model-plus-PPO pipeline. That is why DPO is seen as simpler and more stable in many practical settings. 

**If DPO is easier, why didn’t people just do that first?**

Because it came **later**, after the community had already built the RLHF pipeline around reward models and PPO, and because DPO depends on a particular theoretical reduction of the RLHF objective. Historically, PPO-style RLHF was the standard practical recipe first; DPO was proposed in 2023 as a simplification once the field better understood the objective. 

Also, DPO is not strictly “better in all cases.” It is especially attractive for **static offline preference datasets**, but RL-style methods remain more general when you want **online exploration**, richer sequential credit assignment, or repeated data collection from the current policy as the distribution shifts. That distribution-shift issue is exactly why on-policy reward-learning variants continued to be studied after classic RLHF and after DPO. 

**In short**

- DPO = direct optimization from preference pairs
- avoids explicit reward model + PPO loop
- simpler and often more stable
- arrived later than PPO-based RLHF
- great for offline preferences, less general than full online RL in some settings

## What is PPO, and what is the difference between PPO and GRPO?
**T:** Basic

**A:** 
**PPO** stands for **Proximal Policy Optimization**. It is a policy-gradient method that updates a policy using a **clipped surrogate objective**, which limits how far the new policy can move from the old one in a single step. Its appeal is that it keeps some of the practical stability of trust-region ideas while remaining much simpler and first-order. 

In standard PPO, advantage estimation usually depends on a **learned value function / critic**. So PPO is typically an **actor-critic** method: the actor changes the policy, and the critic estimates expected return to reduce variance. 

**GRPO** keeps the clipped-policy style but **removes the separate critic**. Instead of estimating value with a learned network, it uses **group-relative normalized rewards** from several samples of the same prompt. So the main difference is:

- **PPO:** advantage from critic/value estimates
- **GRPO:** advantage from relative performance within a sampled group 

That means PPO is often more classical and general-purpose, while GRPO is more specialized to settings like LLM reasoning where you can cheaply score multiple candidate completions for one prompt. The trade-off is less memory and engineering for GRPO, but more dependence on reward design and group statistics. 

**In short**

- PPO: clipped policy-gradient actor-critic
- GRPO: PPO-like update without a learned critic
- PPO uses value estimates
- GRPO uses relative group rewards
- GRPO is cheaper, but can be more sensitive to reward/group effects

## What is Jensen–Shannon divergence? Why use it instead of KL divergence?
**T:** Deep

**A:** 
For two distributions $P$ and $Q$, the **Jensen–Shannon divergence** is

$$
\mathrm{JSD}(P,Q)
=\frac12 \mathrm{KL}(P\|M)+\frac12 \mathrm{KL}(Q\|M),\qquad{}M=\frac12(P+Q).
$$

So it measures how far each distribution is from their average mixture. It was introduced by Lin as a symmetric Shannon-entropy-based divergence. 

The main intuitive reason to use JSD instead of KL is that **KL is asymmetric and can blow up to infinity** when supports do not overlap, while **JSD is symmetric and bounded**. So JSD is often a more stable notion of “distributional difference” when you do not want one direction to dominate or when support mismatch is common. 

Another useful intuition: JSD compares both distributions to a **shared midpoint** $M$. That makes it a kind of “consensus discrepancy,” whereas KL is directional and answers “how inefficient is it to code $P$ using $Q$?” not “how different are they in a balanced sense?” 

Why do we need it in ML? One famous example is the original GAN theory, where the ideal discriminator game is connected to minimizing a Jensen–Shannon-type divergence between the data distribution and the generator distribution. More broadly, JSD is useful when you want **symmetry, boundedness, or a metric-like comparison**. 

If they ask about **Rényi divergence**: JSD is not the same thing, but it is related historically to the **information radius** viewpoint, and that connects to more general divergence families like Rényi-based radii. 

**In short**

- $\mathrm{JSD}(P,Q)=\frac12\mathrm{KL}(P\|M)+\frac12\mathrm{KL}(Q\|M)$
- symmetric, unlike KL
- bounded/finite, unlike KL under support mismatch
- compares both distributions to their midpoint
- useful when you want a balanced notion of distance

## what are the pros and cons of RLVR?
**T:** Basic

**A:** 
**RLVR** means **Reinforcement Learning with Verifiable Rewards**: instead of asking humans to rank outputs, you use an automated verifier—unit tests, exact-answer checking, symbolic equivalence, theorem checking, etc.—to produce the reward. It has become especially important in math and code. 

**Pros:**

The biggest advantage is **cheap, scalable supervision**. If correctness can be checked automatically, you can generate many trajectories and train online without the bottleneck of human preference labeling. That is one reason RLVR has looked attractive for mathematical and coding reasoning. 

A second advantage is that the reward is often **more objective** than human preference scores. For tasks with exact correctness, RLVR can directly reinforce answers that satisfy the external criterion instead of relying on human taste or noisy judging. 

**Cons:**

The limitation is obvious but deep: RLVR only works well where you have a **reliable verifier**. That makes it much easier in math and code than in open-ended dialogue, taste, policy, or creativity. Extending RLVR beyond tightly verifiable domains is an active research problem. 

Another problem is **verifier noise and reward hacking**. Real verifiers produce false positives and false negatives; rule-based checkers can be brittle, and learned judges can be gamed. Recent work shows that these errors can slow convergence, distort learning, or even reverse the learning signal if verifier quality is poor enough. 

A final caveat is that RLVR may improve **which trajectories are preferred** without necessarily increasing true reasoning diversity or exploration. There has been debate about whether it learns genuinely new reasoning versus mostly reweighting existing chains; recent work argues evaluation itself can be misleading if it ignores correctness of the full chain of thought. 

**In short**

- RLVR = RL using automated correctness checks
- great for math/code because supervision scales cheaply
- more objective than preference labels when correctness is verifiable
- limited by verifier availability and quality
- vulnerable to false positives, false negatives, and reward hacking

## What is the difference between PlaNet and Dreamer?
**T:** Hands-on

**A:** 
Both **PlaNet** and **Dreamer** are **world-model RL** methods from Danijar Hafner and collaborators. Both learn a latent dynamics model from pixels, so both belong to the same family. 

The core difference is in **how they choose actions**. **PlaNet** learns the world model and then does **online planning in latent space**—typically model predictive control over imagined trajectories—at decision time. So it is explicitly a planning-based model-based RL agent. 

**Dreamer** keeps the latent world model but replaces online planning with a learned **actor-critic** trained on imagined trajectories. So Dreamer “dreams” future latent rollouts and directly improves a policy and value function in imagination, then executes the learned actor at test time. 

That gives a clean interview contrast:

- **PlaNet:** learn model, then **plan online**
- **Dreamer:** learn model, then **learn a policy in imagination** 

Practically, PlaNet can be more explicitly planning-driven, while Dreamer usually has cheaper action selection at inference because it does not solve a planner from scratch every step. Dreamer’s actor-critic formulation also made it easier to scale to longer-horizon behavior and later versions like DreamerV3. 

**In short**

- both are latent world-model RL
- PlaNet = online planning in latent space
- Dreamer = actor-critic trained in imagined latent rollouts
- PlaNet plans every step
- Dreamer learns a policy, so inference is usually cheaper

---
