# Learning - Aditya
**Q. What is DDIM and what is difference from DDPM?**

**A:** Denoising Diffusion Probabilistic Models (DDPM) rely on a Markov chain to iteratively add and remove noise, which requires hundreds of sequential steps for generation. 

Denoising Diffusion Implicit Models (DDIM) generalize this by formulating a non-Markovian forward process that yields the same objective function. This allows DDIM to skip steps during inference, resulting in a deterministic sampling process that dramatically accelerates generation without retraining the model.

**Q: How are few-step diffusion models trained while maintaining consistency?**

**A:** Few-step diffusion models try to reduce the number of denoising steps without destroying sample quality. This can be achieved in different ways:

* Consistency model:
The core idea is to learn a function $f_{\theta}$ that maps any point on a probability flow ODE trajectory directly to its origin (the clean sample). This enforces a self-consistency property: $f_{\theta}(x_t, t) = f_{\theta}(x_{t'}, t')$ for any two points on the same trajectory. Training can be done via consistency distillation (using a pre-trained diffusion model to generate ODE pairs) or consistency training (directly from data, no teacher needed). At inference, you can generate in as few as 1–2 steps.
* Logit-based Distillation: : A student model is trained to match the output distributions (scores/logits) of a multi-step teacher model but is constrained to do so over much larger timestep intervals, effectively compressing the teacher's schedule. 

**Q: Difference between distillation and sampler (or scheduler)?**

**A:** A sampler or scheduler changes how you run inference on the same trained diffusion model. It decides the timestep sequence and numerical update rule used to go from noise to sample. For example: DDIM sampler, Euler sampler, DPM-Solver, etc.

Distillation, in contrast, trains a new model so that fewer denoising steps are needed. It changes the model itself, not just the inference procedure.

**Q: How is self-attention and cross-attention utilized in diffusion architectures?**

**A:** In diffusion architectures, Self-Attention processes the image latents to capture global spatial dependencies, typically applied after convolution blocks to relate distant patches of the image. 

Cross-Attention is the mechanism for conditioning; it maps external modalities (like CLIP text embeddings) into the spatial features, dictating how the prompt guides the image generation.

**Note:** In DiT-style architectures, conditioning is done via adaptive layer norm (adaLN) instead of cross-attention, where the text/timestep embedding modulates the scale and shift of normalization layers. This is cheaper and has been shown to work comparably well.

**Q: What is the start-up problem in multi-step solvers, and how is it addressed?**

**A:** Multi-step numerical solvers (like Adams-Bashforth) require a history of previous timesteps ($x_t, x_{t+1}, \dots$) to predict the next step $x_{t-1}$. At $t=T$ (the beginning of generation), there is no history. 

To solve this, single-step solvers like Runge-Kutta or Euler are used to bootstrap the first $k$ steps. Once a sufficient history buffer is built, the solver switches to the more efficient multi-step method.

**Q: Can we do KV caching in diffusion?**

**A:** In standard autoregressive LLMs, KV caching works well because previous tokens do not change, so the cached keys and values remain valid. In diffusion, hiowever, this is harder because the entire latent/image representation changes at every denoising step. That means cached activations or attention states from one step may become stale at the next step.

That said, while not standard in continuous spatial diffusion, it is used in Diffusion LLMs. Methods like dKV-Cache exploit the high representational similarity across adjacent diffusion noise levels. By caching intermediate attention keys and values, the model avoids redundant computations across timesteps.

**Q: How can forward passes be reused to accelerate diffusion?**

**A:** The core observation is that adjacent denoising steps often produce very similar intermediate features, making full recomputation redundant. Key approaches include:

* Block/Layer Caching: Cache the outputs of certain U-Net or DiT blocks from timestep t and reuse them at timestep t-1 instead of recomputing. Typically, the earlier (encoder) blocks change more slowly across steps than the decoder blocks, so caching is most effective there. DeepCache and Block Caching follow this pattern.
* Attention Reuse: Reuse the self-attention maps from a previous step, since spatial attention patterns tend to be stable across nearby timesteps. This avoids the expensive $QK^T$ computation.

**Q: How can diffusion be interpreted in an RL setting?**

**A:** The reverse diffusion process can be formulated as a sequential decision-making problem where the denoiser acts as a stochastic policy. The state is the noisy data $x_t$, the action is the denoised prediction, and the environment dictates the transition to $x_{t-1}$. This allows RL algorithms (like PPO) to fine-tune diffusion models by treating aesthetic scores, prompt alignment, or task success as reward signals to maximize.

---