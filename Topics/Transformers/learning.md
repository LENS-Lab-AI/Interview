# Learning 

---

### Q1. What is AdamW, how it differ from Adam?

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

**A:**

- **Learning-rate warmup** and **cosine decay** for smoother optimization.
- **Gradient clipping** to prevent exploding gradients.
- **Layer normalization** and **residual connections** to stabilize deep architectures.
- **Weight decay** or **dropout** for regularization.
- **Mixed precision (FP16/BF16)** for speed and memory efficiency.
why they helps intuitively
---

### Q3. Why use *cosine decay* after warmup? Why cosine (not others)?
frame the question better
**A:**

- **Warmup:** Gradually increases the learning rate to stabilize initial training.
- **Cosine decay:** Smoothly decreases the learning rate following a cosine curve:

$$
\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\pi t / T))
$$

**Why cosine:**

- Smooth and non-abrupt decay.
- Avoids sudden jumps seen in step or exponential schedules.
- Empirically stable and yields good generalization.

**Alternatives:** linear decay, exponential decay, polynomial decay — but cosine offers a good trade-off between stability and convergence.

---

### Q4. What is *gradient clipping*, and why is it used?
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

**A:** Both are strategies to train large models across multiple GPUs, but they differ in *how* the model is split.

**Model parallelism** divides the **model’s parameters** across devices — for example, placing different layers or parts of a layer on different GPUs. Each GPU computes its part of the forward and backward pass. This approach is mainly used when the model is too large to fit in one GPU’s memory.

**Pipeline parallelism** divides the **training process** into sequential stages across GPUs. The input batch is broken into *micro-batches* that move through the pipeline — while one GPU works on the next batch, another continues the previous one. This improves utilization and throughput.


**In short:**

- *Model parallelism* → splits **the model** to fit memory.
- *Pipeline parallelism* → splits **the computation** to improve efficiency.
  Large-scale systems (e.g., DeepSpeed, Megatron-LM) often combine both for optimal performance.

They are often **combined** for large-scale model training (e.g., Megatron-LM, DeepSpeed).

---

### Q6. What’s the difference between *LoRA* and *QLoRA*?

**A:** **LoRA**, short for *Low-Rank Adaptation*, is a fine-tuning technique that makes large language model training more efficient.
 Instead of updating all model parameters, LoRA inserts small, low-rank adapter matrices into the model’s weight layers. During fine-tuning, only these adapters are trained while the base model remains frozen. This drastically reduces the number of trainable parameters and speeds up training.

**QLoRA**, or *Quantized LoRA*, takes this a step further. It allows fine-tuning on **quantized models**, typically using 4-bit quantization (like NF4). By representing the model weights in lower precision, QLoRA cuts down on memory usage even more — without significantly hurting performance. This makes it possible to fine-tune very large models (like 65B parameters) on a single high-end GPU or small cluster. add peerformance affect; accuracy/speed etc.

**In short:**

- LoRA makes fine-tuning efficient by training small adapter layers.
- QLoRA makes it even more efficient by fine-tuning **quantized** models, enabling large-scale fine-tuning on modest hardware.

---

### Q7. What are other adapter-based fine-tuning methods (e.g., DoRA)?

**A:**
 Beyond LoRA and QLoRA:

- **DoRA (Weight-Decomposed LoRA)** — decomposes pretrained weights into direction and magnitude, updating only the direction for better stability.
- **AdapterFusion / Prefix-Tuning / BitFit** — other parameter-efficient fine-tuning (PEFT) techniques targeting specific model components or embeddings.

These methods balance performance with memory efficiency in downstream fine-tuning.
text to lora
---

### Q8. What’s the difference between *SFT* and *RLHF*?

**A:** **SFT**, or *Supervised Fine-Tuning*, is the stage where a model learns to produce good answers by **imitating human-written examples**.
 It’s a straightforward supervised learning process: the model is trained on prompt–response pairs, minimizing cross-entropy loss to match the human responses. This step teaches the model to follow instructions and generate coherent, helpful outputs.

**RLHF**, or *Reinforcement Learning from Human Feedback*, builds on top of SFT. Instead of directly imitating humans, the model now learns from **human preferences** — for example, which of two responses a human finds better.
 A separate *reward model* is trained using this preference data, and the main model is then optimized to maximize this learned reward signal using reinforcement learning (usually with PPO).

**In short:**

- SFT trains the model to **imitate good answers** using labeled data.
- RLHF trains the model to **align with human preferences**, optimizing for what people *prefer*, not just what they *wrote*.

Together, SFT gives the model basic instruction-following ability, and RLHF refines that behavior to make it more aligned, safe, and human-like.


---

### Q9. How do Direct Preference Optimization (DPO) and RLHF differ?

**A:**
DPO simplifies RLHF by removing the reinforcement learning loop.
Instead of training a separate reward model and using PPO, DPO directly optimizes the model parameters to prefer responses that humans liked more.

It minimizes a loss derived from the same Bradley–Terry preference model, without the instability or extra complexity of RL.

**In short:**

- RLHF → reward model + PPO optimization
- DPO → direct, simpler preference learning (no reward model training)

DPO achieves alignment comparable to that of that of RLHF wicomputation computation and fewer moving parts.

---

### Q10. What is the Bradley–Terry model originally?

**A:** The **Bradley–Terry model** is a probabilistic model used to represent *pairwise comparisons* between items. 
It assumes that each item $i$ has a latent "ability" parameter $\beta_i$, and the probability that item $i$ is preferred over item $j$ is:
$$
P(i \succ j) = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}}
$$

This model is widely used for preference learning, ranking, and competition outcomes (e.g., sports or survey comparisons).

---

### Q11. Are there other ways besides pairwise preference to evaluate answers?

**A:** 

- **Score-based evaluation**
  Assign a numerical score to each response (e.g., 1–5 or 0–100). 
  Useful for tasks where absolute quality can be graded.

- **Ranking (listwise) evaluation** 
  Rank multiple responses simultaneously rather than pairwise. 
  Helps capture *relative* quality across several answers at once.

- **Probabilistic modeling** 
  Estimate a probability distribution over responses — for example, the likelihood of each being the "best" answer. 
  See: [Biyik et al., *Learning from Comparisons and Choices*, 2022](https://iliad.stanford.edu/pdfs/publications/biyik2022learning.pdf)

- **Direct preference prediction** 
  Train a model to directly predict preference scores or rankings without explicit pairwise data.

- **Reward modeling (RLHF style)** 
  Learn a reward signal from human feedback to evaluate responses in a continuous way rather than discrete comparisons.

---

### Q12. What are emerging trends in alignment beyond RLHF?

**A:** 
New research explores alignment methods that are simpler, more sample-efficient, or less human-dependent than RLHF:

- **DPO (Direct Preference Optimization)**: simplifies RLHF by removing the reinforcement learning loop; direct loss from preference pairs.
- **IPO (Implicit Preference Optimization)**: unifies RLHF and DPO in a single framework.
- **RLAIF (Reinforcement Learning from AI Feedback)**: use strong models (like GPT-4) to generate preference data.
- **Constitutional AI**: align models to principles (rules or constitutions) rather than human ratings.
- **Self-rewarding models**: train models to critique or rank their own generations, reducing dependence on human labels.

These approaches aim for scalable alignment. Keeping model behavior consistent with human values while minimizing human supervision costs.