# Learning - Xinyuan

---

### Q1. What is AdamW? How is it differ from Adam?
**T:** Basic

**A:** **AdamW** is a variant of the **Adam optimizer** that fixes how **weight decay** is applied. 

In the original **Adam**, weight decay was implemented by adding an L2 penalty to the loss function. However, because Adam rescales gradients adaptively, this L2 regularization does **not behave like true weight decay**, it gets entangled with the gradient updates, often leading to suboptimal generalization.

**AdamW** (proposed by Loshchilov & Hutter, 2017) decouples weight decay from the gradient-based update. Instead of adding L2 loss to the objective, it directly **decays the weights after each update step**, keeping the adaptive gradient behavior intact.

As a result, AdamW offers **better generalization** and **more stable training**, especially for large-scale models like transformers.

**In short:**

- **Adam:** mixes L2 regularization into the gradient, which interacts with adaptive updates.
- **AdamW:** applies *true weight decay* separately from gradient updates.

This decoupling makes AdamW the **default optimizer** for most modern deep learning frameworks and transformer-based models.

---

### Q2. What tricks improve training stability and convergence?
**T:** Deep

**A:**

- **Learning-rate Warmup & Annealing:**
    - Start with a small LR, ramp up linearly (Warmup), and then slowly decay it, often using cosine or linear schedules (Annealing).
    - **Intuition:** At the very beginning, model parameters are random, so the gradients are massive and noisy. If taking a full step immediately, the model might be destabilized and have possible early divergence. Warmup lets the model find a reasonable starting direction safely. Later, annealing reduces the step size so the model can settle into a minima instead of bouncing around the edges.
- **Batch Size Scaling:**
    - Gradually increasing the batch size over time to maintain stability in SGD in LLM training.
    - **Intuition:** Increasing batch size reduces the noise in the gradient estimation (lowering the variance). This mimics the effect of learning rate decay (increasing the “signal-to-noise” ratio of the updates) without slowing down the effective training speed, often helping the model escape sharp minima in favor of flatter, more robust ones.
- **Gradient Clipping:**
    - Cap the norm of the gradient vector (e.g., to 1.0) before the update to prevent exploding gradients.
    - **Intuition:** Loss landscapes of deep networks (especially RNNs/Transformers) can have steep cliffs. Without clipping, a single massive gradient step can shoot parameters into a bad region (exploding gradients), undoing days of training.
- **Weight Decay & Dropout**
    - Weight Decay pushes weights towards zero (L2 penalty). Dropout randomly switches off neurons during training.
    - **Intuition:** Against overfitting. Weight Decay keeps the weights small, preventing the model from relying too heavily on any single feature (which usually means it's memorizing noise). Dropout forces the network to be robust. By randomly breaking connections, the model can't rely on one specific path to get the answer; it has to learn redundant, distributed representations that generalize better.
- **Layer normalization & Residual connections:**
    - Add skip connections ($x + F(x)$) and normalize inputs across features per layer to stabilize deep architectures.
    - **Intuition:** Residuals create a “gradient highway” for gradients to flow backward without vanishing. Layer Norm ensures the input to the next layer has a stable mean/variance, preventing covariate shift where layers have to constantly re-learn how to handle shifting scales from previous layers.
- **Mixed precision (FP16/BF16):**
    - Use lower precision for calculations and full precision for weight accumulation for speed and memory efficiency.
    - **Intuition:** Reduces memory bandwidth pressure and allows larger batch sizes, which indirectly stabilizes training by improving gradient estimation quality.
    
**Note:** In LLMs, these stability tricks are mostly applied during the Pre-training and Supervised Fine-Tuning (SFT) stages. Later stages like RLHF usually require less aggressive regularization.

**In short:**
- **Warmup & Annealing** manage the **speed**: start cautious, end precise.
- **Gradient Clipping & Batch Scaling** manage the **volatility**: prevent crashes and reduce noise.
- **Residuals & Norms** fix the **signal flow**: allow gradients to survive deep networks.
- **Decay & Dropout** improve **generalization**: force the model to learn robust patterns, not memorization.

---

### Q3. Why is a cosine decay schedule (with warmup) typically preferred over linear or step-based schedules in LLM training??
**T:** Deep

**A:**
The choice of scheduler defines the optimization trajectory. In LLM training, Cosine Decay with Warmup is the standard because it offers the best balance between stability, exploration, and ease of tuning.
1. **Warmup Phase:** We initially ramp up the learning rate linearly from zero. At initialization, parameters are random and gradient variances are high. A full-strength step immediately can destabilize the model or cause early divergence. Warmup allows the optimizer to gather statistics and find a stable descent direction before accelerating.
2. **Decay Phase (Why Cosine?):** After warmup, the learning rate must decrease to allow the model to converge into a local minimum. Cosine decay follows a cosine curve, slowly decreasing initially, accelerating in the middle, and slowing again at the end:

$$
\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\pi t / T))
$$

- **Vs. Step Decay:** Step schedules drop the LR abruptly. These shocks can trap the model in a suboptimal basin before it has finished exploring the current loss landscape. Cosine is smooth and continuous, allowing the model to settle naturally without optimization shocks.
- **Vs. Linear Decay:** A linear schedule decays too constantly. Cosine spends a longer time at a relatively high learning rate before dropping off. This extended period of high-energy updates encourages broader exploration, helping the model find flatter, more robust minima.

3. **Modern Context: The Rise of WSD** While Cosine is the default, recent large-scale runs (like MiniCPM or Llama 3) often use WSD (Warmup-Stable-Decay).
- Instead of constantly decaying, WSD keeps the learning rate constant (stable) for 80-90\% of training, then decays rapidly at the end.
- **Why?** It decouples training from a fixed end-step. Because the LR stays high, you can pause training, add more data, and continue easily—something that is mathematically difficult with Cosine once the LR has already decayed to near-zero.

**In short:**
- Warmup ensures stability during the volatile early phase.
- Cosine is preferred for its smooth convergence and “set-and-forget” simplicity for fixed-budget runs.

---

### Q4. What is *gradient clipping*, and why is it used?
**T:** Basic

**A:** **Gradient clipping** limits the magnitude of gradients during backpropagation.

- **Purpose:** Prevent *gradient explosion*, which can destabilize or crash training.
- **Common method:** Clip gradients by norm, e.g.

$$
g \leftarrow c \cdot \frac{g}{\|g\|} \quad \text{if } \|g\| > \text{c}
$$

where $c$ is a hyperparameter, $g$ is the gradient, and $\|g\|$ is the norm of $g$. Since $g/\|g\|$ is a unit vector, after rescaling the new $g$ will have norm $c$. Note that if $\|g\| < c$, then we don’t need to do anything.

- **Effect:** Stabilizes updates, especially in RNNs, transformers, and reinforcement learning models.

---

### Q5. What is *model parallelism* vs *pipeline parallelism*?
**T:** Basic

**A:** Both are strategies to train large models across multiple GPUs, but they differ in *how* the model is split.

**Model parallelism** divides the **model’s parameters** across devices — for example, placing different layers or parts of a layer on different GPUs. Each GPU computes its part of the forward and backward pass. This approach is mainly used when the model is too large to fit in one GPU’s memory.

**Pipeline parallelism** divides the **training process** into sequential stages across GPUs. The input batch is broken into *micro-batches* that move through the pipeline — while one GPU works on the next batch, another continues the previous one. This improves utilization and throughput.


**In short:**

- *Model parallelism* → splits **the model** to fit memory.
- *Pipeline parallelism* → splits **the computation** to improve efficiency.
  Large-scale systems (e.g., DeepSpeed, Megatron-LM) often combine both for optimal performance.

They are often **combined** for large-scale model training (e.g., Megatron-LM, DeepSpeed).

---

### Q6. What is the role of the FFN (Feed-Forward Network) in a Transformer?
**T:** Basic

**A:** While the **Self-Attention** mechanism mixes information *across* time steps (tokens), the **FFN** processes information *per token* individually.

* **Rank Restoration:** Attention matrices are often low-rank (collapsing information). The FFN involves a projection to a higher dimension ($d_{model} \to 4d_{model}$) and back, adding non-linearity (ReLU/GELU/SwiGLU). This restores the rank of the representations and increases model capacity.
* **Key-Value Memory:** Research (e.g., Geva et al.) suggests FFNs act as **key-value memories**, where the first layer detects specific patterns (keys) in the input, and the second layer outputs the corresponding distribution over the vocabulary or features (values).

**In short:**

* **Attention:** Mixes information spatially (token-to-token).
* **FFN:** Processes information deeply (state-to-state), acting as the model's static memory.

---

### Q7. How does Mixture of Experts (MoE) work, and what are its advantages?
**T:** Basic

**A:** **MoE** replaces the dense FFN layers in a Transformer with a **sparse** layer containing multiple experts (independent FFNs) and a **router** (gate).

For every incoming token $x$, the router selects only the top-$k$ experts (usually $k=1$ or $2$) to process that token. The output is the weighted sum of the selected experts.

**Advantages:**

1. **Decouples Parameter Count from Compute:** You can increase the model size (parameters) massively (e.g., 100x) without increasing the inference cost (FLOPs), because only a fraction of parameters are active per token.
2. **Huge Capacity:** Allows the model to memorize significantly more information (due to the memory nature of FFNs described above) compared to a dense model of the same compute budget.

**In short:**
MoE enables **sparse activation**. It scales the model's *knowledge* (parameters) without scaling the *compute cost* linearly.

---

### Q8. How is the MoE Router trained?
**T:** Deep

**A:** The router is typically a learnable linear layer followed by a Softmax:
$$G(x) = \text{Softmax}(x \cdot W_g)$$
The routing decision (selecting indices) is discrete and non-differentiable. To train this, we usually use the **weighted average** of the selected experts to allow gradients to flow back into the router weights.

If the router selects indices $i$ and $j$ for token $x$, the output is:$$y = P_i \cdot E_i(x) + P_j \cdot E_j(x)$$
Where $P$ represents the probability assigned by the gate.
During backpropagation, gradients flow through $E(x)$ to train the experts, and through $P$ to train the router weights $W_g$ (teaching it which expert to pick).

*Note: Some implementations use Gumbel-Softmax or noisy top-k gating to encourage exploration.*

---

### Q9. What are *load balancing losses* in LLMs?
**T:** Deep

**A:** Without intervention, an MoE router often converges to a degenerate solution where it sends **all tokens to a single expert** (Expert Collapse). This maximizes early rewards but wastes the capacity of other experts and causes memory OOM on the favored expert's GPU.

**Load Balancing Loss ($L_{aux}$)** is an auxiliary loss added to the objective to force uniform usage of experts. It usually minimizes the coefficient of variation of the dispatching.

A common formulation (e.g., in Switch Transformer) minimizes the dot product of:

1. $f_i$: The fraction of tokens dispatched to expert $i$.
2. $P_i$: The average probability assigned to expert $i$ across the batch.
   $$L_{aux} = N \sum_{i=1}^{N} f_i \cdot P_i$$

**In short:**
This loss penalizes the model if it routes too many tokens to specific experts, ensuring all experts are trained equally and computational load is balanced across devices.

### Q10. What’s the difference between *LoRA* and *QLoRA*?
**T:** Basic

**A:** Both are Parameter-Efficient Fine-Tuning (PEFT) methods, but they solve slightly different bottlenecks.
**1. LoRA (Low-Rank Adaptation)** Instead of re-training the massive weight matrices of the model, LoRA freezes the pre-trained weights and injects trainable **rank decomposition matrices** into each layer.

- If the weight update is $\Delta W$, LoRA approximates it as $\Delta W = B \times A$, where $B$ and $A$ are tiny matrices. You only train $A$ and $B$.
- **Performance:**
    - **Accuracy:** Matches full fine-tuning results almost exactly.
    - **Speed:** Training is faster (fewer gradients to calculate), and serving is efficient because you can merge the adapter weights back into the base model (zero inference latency).

**2. QLoRA (Quantized LoRA)** QLoRA is essentially LoRA applied to a **4-bit quantized base model**. It uses a special data type (NormalFloat4 or NF4) and Double Quantization to squeeze the model size down as much as possible.
- **Performance:**
    - **Memory (The Big Win):** It drastically reduces VRAM usage. You can fine-tune a 65B/70B parameter model on a single 48GB GPU, which is impossible with standard LoRA.
    - **Speed (The Cost):** QLoRA is actually slower to train than standard LoRA (roughly 30\% slower). Because the weights are stored in 4-bit, but computations must happen in 16-bit (BF16). The system has to constantly de-quantize weights on the fly during the forward and backward passes, adding computational overhead.
    - **Accuracy:** Surprisingly, it incurs negligible accuracy loss compared to 16-bit LoRA.

**In short:**

- LoRA makes fine-tuning efficient by training small adapter layers.
- QLoRA makes it even more efficient by fine-tuning **quantized** models, enabling large-scale fine-tuning on modest hardware.

---

### Q11. What are other adapter-based fine-tuning methods (e.g., DoRA)?
**T:** Deep

**A:**
While LoRA is the default, newer methods try to fix its specific weaknesses.
- **DoRA (Weight-Decomposed LoRA):** LoRA updates weights in a way that forces the direction and magnitude of the change to be coupled. In full fine-tuning, these often change independently. DoRA splits the weight matrix into two parts: **Magnitude** (how strong the weight is) and **Direction** (where it points).
  $$W = m \cdot \frac{V}{\|V\|}$$
  It applies LoRA only to the Direction ($V$) and trains the Magnitude ($m$) directly. It mimics full fine-tuning behavior much closer than LoRA, often yielding higher accuracy and better robustnes, with no extra inference cost.
- **AdaLoRA (Adaptive LoRA):** Not all layers are equally important. Standard LoRA uses the same rank for every single layer. AdaLoRA figures out which layers need more capacity and which don't, dynamically allocating the rank budget to the most critical layers during training.
- **Text-to-LoRA (Cutting Edge):** Instead of training an adapter on a dataset, can you just ask for one? This method uses a **Hypernetwork**—a small model trained to generate the LoRA weights ($A$ and $B$ matrices) directly from a text description of a task (e.g., “Make the model speak like a pirate”). It allows for instant, zero-shot adaptation without standard training.
- **Llama-Adapter / Prefix Tuning:** Instead of adding weights inside the layers, these methods add “soft prompts” (learnable vectors) to the input or specific attention layers. They are generally less stable and perform worse than LoRA for complex reasoning, but are highly parameter-efficient.

**In short:**
- **DoRA** is the “better LoRA” (more accuracy, same inference speed).
- **AdaLoRA** is the “efficient LoRA” (allocates parameters smartly).
- **Text-to-LoRA** is the “instant LoRA” (generates adapters from prompts).


---

### Q12. What’s the difference between *SFT* and *RLHF*?
**T:** Basic

**A:** **SFT**, or *Supervised Fine-Tuning*, is the stage where a model learns to produce good answers by **imitating human-written examples**.
 It’s a straightforward supervised learning process: the model is trained on prompt–response pairs, minimizing cross-entropy loss to match the human responses. This step teaches the model to follow instructions and generate coherent, helpful outputs.

**RLHF**, or *Reinforcement Learning from Human Feedback*, builds on top of SFT. Instead of directly imitating humans, the model now learns from **human preferences** — for example, which of two responses a human finds better.
 A separate *reward model* is trained using this preference data, and the main model is then optimized to maximize this learned reward signal using reinforcement learning (usually with PPO).

**In short:**

- SFT trains the model to **imitate good answers** using labeled data.
- RLHF trains the model to **align with human preferences**, optimizing for what people *prefer*, not just what they *wrote*.

Together, SFT gives the model basic instruction-following ability, and RLHF refines that behavior to make it more aligned, safe, and human-like.


---

### Q13. How do Direct Preference Optimization (DPO) and RLHF differ?
**T:** Deep

**A:**
DPO simplifies RLHF by removing the reinforcement learning loop.
Instead of training a separate reward model and using PPO, DPO directly optimizes the model parameters to prefer responses that humans liked more.

It minimizes a loss derived from the same Bradley–Terry preference model, without the instability or extra complexity of RL.

**In short:**

- RLHF → reward model + PPO optimization
- DPO → direct, simpler preference learning (no reward model training)

DPO achieves alignment comparable to that of that of RLHF wicomputation computation and fewer moving parts.

---

### Q14. What is the Bradley–Terry model, and why is it the standard for Reward Modeling?
**T:** Basic

**A:** The **Bradley–Terry model** is the mathematical link between abstract scores and human choices.

In RLHF, humans don't give absolute scores. They give **pairwise comparisons** We need a way to translate those A-vs-B wins into a continuous reward score for the model to optimize.

**The Math:** Bradley-Terry formalizes this. It assumes every item $i$ has a latent score $r_i$ (reward). The probability that $i$ beats $j$ is a sigmoid function of the difference in their scores:$$P(i \succ j) = \sigma(r_i - r_j) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}}$$

**In LLM Training:** When we train a Reward Model, we feed it pairs of (Good Response, Bad Response). The model tries to predict a scalar score $r$ for each. We train it by minimizing the error so that $r_{\text{good}} > r_{\text{bad}}$ according to the probability formula above.

**In short:** The Bradley-Terry model allows us to derive absolute reward scores from relative human rankings.

---

### Q15. Beyond pairwise comparison, what alternative methods exist for evaluating model responses, and what are their trade-offs?
**T:** Deep

**A:** 

**1. Pointwise Scoring (Absolute Grading)**
  Assign a numerical score to each response (e.g., 1–5 or 0–100). Useful for tasks where absolute quality can be graded.
  - **Pros:** **Simplicity and Direct Magnitude.** Capture absolute quality rather than the relative ordering.
  - **Cons:** **Poor Calibration and drift.** Subject to heavy bias; one annotator's “7” is another's “9”, making data noisy.

**2. Listwise Ranking** 
  Rank a set of $K$ responses (e.g., 4 or 5) from best to worst. Helps capture *relative* quality across several answers at once.
  - **Pros: High Efficiency.** A single ranking induces $K(K-1)/2$ equivalent pairwise comparisons, maximizing signal per dollar.
  - **Cons: High Cognitive Load.** Ranking multiple long responses causes annotator fatigue and errors.

**3. Probabilistic modeling** 
  Estimate a probability distribution over responses (e.g., the likelihood of each being the best) rather than a fixed score.
  - **Pros: Uncertainty Quantification.** It models ambiguity in human preferences rather than forcing a deterministic choice.
  - **Cons: Complexity.** Requires more sophisticated inference methods than standard regression or classification.

**4. Direct preference prediction** 
  Train a model to directly predict preference scores or rankings without explicit pairwise data.
  - **Pros: Simplicity.** Bypasses the combinatorial explosion of generating pairs.
  - **Cons: Data Scarcity.** Requires datasets that are already scored/ranked, which are harder to obtain than simple pairwise clicks.

**5. Reward modeling (RLHF style)** 
  Use a trained Reward Model to evaluate responses continuously during generation or testing.
  - **Pros: Automation.** Allows for scalable evaluation without humans in the loop.
  - **Cons: Proxy Bias.** The model optimizes for the reward score, not necessarily true quality (Goodhart’s Law), leading to reward hacking.

**In short:**
- **Score-based** measures **absolute** quality (but is noisy).
- **Ranking** measures **relative** quality (and is efficient).
- **Probabilistic/Direct** methods are emerging research areas to handle **uncertainty** and **efficiency**.
  
---

### Q16. What are emerging trends in alignment beyond RLHF? Where are they used?
**T:** Deep
**A:** 
The field is moving away from the complex PPO pipeline toward simpler optimization and scalable supervision.

**1. DPO (Direct Preference Optimization)**
- Removes the separate Reward Model and PPO loop. It optimizes the policy directly on preference data using a simple classification loss.
- **Use Case:** The current open-source standard (Llama 3, Mistral) due to stability and memory efficiency.

**2. IPO (Implicit Preference Optimization)**
- Adds regularization to DPO to prevent overfitting on preference data.
- **Use Case: Robust Alignment.** Essential when preference labels are noisy or theoretically grounded constraints are needed (e.g., scientific code generation).

**3. Reasoning Alignment (Process Supervision / “System 2”)**
- Instead of rewarding just the final answer (Outcome Supervision), this method rewards valid steps of reasoning (Process Supervision). It trains models to think (Chain-of-Thought) before answering.
- **Use Case: Complex Logic.** Essential for Math/Code models (e.g., OpenAI o1, DeepSeek-R1).

**4. RLAIF (Reinforcement Learning from AI Feedback)**
- Uses a strong model (e.g., GPT-4) to rank outputs for a smaller model.
- **Use Case: Scaling.** Used by Google/Anthropic to bypass human bottlenecks.

**5. KTO (Kahneman-Tversky Optimization)**
- Aligns using simple “Good/Bad” (binary) labels, removing the need for paired “A vs B” data.
- **Use Case: Enterprise Data.** Useful when you have logs of successful tasks but no direct comparisons.

**6. Constitutional AI**
- Aligns model to principles (rules or constitutions) rather than human ratings.
- **Use Case: Safety & Compliance.** Pioneered by Anthropic (Claude) to ensure models remain harmless without extensive human intervention on every edge case.

**7. Self-rewarding models**
- The model acts as both generator and judge to train itself iteratively.
- **Use Case: Superhuman Performance.** Attempting to break the “human ceiling”.

**In short:**
- **DPO / IPO / KTO** fix the **optimization**: They make training stable, robust, and possible with simpler (or unpaired) data.
- **RLAIF / Constitutional / Self-Reward** fix the **supervision**: They remove the bottleneck of human labeling by using AI judges or rule sets.
- **Process Supervision** fixes the **reasoning**: It forces models to "show their work" and verify logic step-by-step, reducing hallucinations in math and code.
