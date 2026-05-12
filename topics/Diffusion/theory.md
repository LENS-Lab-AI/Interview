# Theory - Som

## Why use score functions instead of directly learning the data distribution?
**T: Deep**

**A:** Learning a probability distribution directly requires knowing how probable every possible input is, which means you need everything to sum to 1. Computing that normalization constant is intractable in high dimensions. The score function the gradient of the log density tells you which direction to move to find more likely samples, and it never requires that normalization. You get the useful information (where to go) without the intractable part (how much everything weighs).

he score function is defined as $\nabla_x \log p(x)$ — the gradient of the log density with respect to the input.
 
Any explicit density model has the form:
 
$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}$$
 
where $Z_\theta = \int e^{-E_\theta(x)} dx$ is the normalization constant. In high dimensions this integral is intractable.
 
Taking the log and then the gradient with respect to $x$:
 
$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x) - \nabla_x \log Z_\theta$$
 
Since $Z_\theta$ does not depend on $x$, its gradient is zero. So:
 
$$\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$$
 
$Z_\theta$ disappears entirely. You can compute and train on the score without ever evaluating the normalization constant.
 
Sampling then works via Langevin dynamics — follow the score, add a little noise, repeat:
 
$$x_{t+1} = x_t + \frac{\eta}{2} \nabla_x \log p_\theta(x_t) + \sqrt{\eta}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
 
In diffusion models, the denoiser is exactly a score estimator. At each noise level $t$, it predicts the direction back toward clean data. This is equivalent to predicting the added noise $\epsilon$, up to a scaling factor — which is why diffusion training reduces to simple noise prediction.



## What is latent diffusion, and why was it introduced?
**T: Basic**

**A:** Pixel-space diffusion is expensive, each denoising step runs on the full resolution image, hundreds of times. Latent diffusion compresses the image into a small latent representation first, runs the entire diffusion process there, then decodes back to pixels at the end. You get the same generative power at a fraction of the compute.

Latent diffusion (Rombach et al., 2022) uses a pretrained autoencoder to split the problem into two stages.
 
First, a VAE encoder compresses the image $x$ into a latent $z$:
 
$$z = E(x), \quad z \in \mathbb{R}^{h \times w \times c}$$
 
where $h \ll H$ and $w \ll W$. Stable Diffusion uses a factor of 8 spatial downsampling, so a $512 \times 512$ image becomes a $64 \times 64 \times 4$ latent - 64 times fewer elements.
 
The diffusion model trains entirely on $z$, not on $x$. At inference, the generated latent is decoded back:
 
$$\hat{x} = D(\hat{z})$$
 
The two stages train independently. The autoencoder is trained first with a perceptual loss and a KL or VQ regularizer on the latent space. The diffusion model then trains on the frozen latent representations.
 


## Why do we compress first and then apply diffusion in latent diffusion models?
**T: Deep**

**A:** The core reason is that the data manifold for natural images is much lower-dimensional than pixel space. A $512 \times 512 \times 3$ image has 786,432 dimensions, but the actual degrees of freedom that determine semantic content are far fewer. The autoencoder finds and keeps those degrees of freedom; everything else is discarded.
 
This matters for diffusion in two ways.
 
First, the distribution diffusion needs to learn becomes simpler. In pixel space, the model must account for local correlations, texture statistics, and perceptually irrelevant noise. In latent space, those are already handled by the decoder. The diffusion model only needs to learn structure at the level of objects, layouts, and semantics.
 
Second, the computation per denoising step drops dramatically. Each step in pixel-space diffusion processes the full resolution. In latent space with $8\times$ downsampling, each step processes $64\times$ fewer elements. Since hundreds of steps are required, this compounds into a large total saving.
 
The order cannot be reversed. Compressing a maximally noisy image does not produce a useful latent — the VAE was trained on real images and its encoder has no meaningful behavior on pure Gaussian noise. Diffusion must happen in a space where the encoder's representation is valid, which means starting from a compressed real image and diffusing there.

## Why are diffusion transformers used instead of U-Net–based models?
**T: Deep**

**A:** Peebles and Xie (2023) showed that replacing the U-Net backbone with a transformer, treating image patches as tokens — achieves better FID at the same parameter count and follows a clean power-law relationship between compute and sample quality. U-Nets do not exhibit this scaling behavior.
 
Three structural reasons explain why transformers are preferred at scale.
 
*Global attention.* Convolutions in U-Nets have a limited receptive field — they aggregate local information and only see the full image through many stacked layers. Transformers attend across all patches at every layer. For tasks where long-range spatial relationships matter (e.g., consistent object layout, coherent lighting), this is a direct advantage.
 
*Cleaner conditioning.* In U-Nets, timestep and text conditioning are injected through a combination of scale-and-shift in residual blocks and cross-attention layers inserted between conv blocks. In DiTs, both are handled through adaptive layer norm (adaLN), which modulates the normalization scale and shift using a projection of the conditioning signal. It is simpler, cheaper, and in ablations performs comparably or better.
 
*Unified architecture.* Transformers are the dominant architecture across vision and language. Using them for diffusion means training recipes, scaling intuitions, and infrastructure transfer directly from the broader ecosystem.
 
U-Nets do have one structural advantage: their skip connections between encoder and decoder levels are explicitly multi-scale. Spatial detail flows directly from early encoder layers to late decoder layers without going through the bottleneck. DiTs recover some of this through patch size choices and depth, but the multi-scale routing is less explicit.

## Why are U-Nets commonly used in diffusion models?
**T: Basic**

**A:** A U-Net has three parts: a contracting encoder that progressively downsamples the input, an expanding decoder that upsamples back to the original resolution, and skip connections that directly link each encoder level to the corresponding decoder level.
 
For diffusion, this is a natural fit. The denoiser must predict either the added noise $\epsilon$ or the clean image $x_0$ at full resolution. Doing that well requires two things simultaneously — understanding global structure (what object is this, what is the overall layout) and preserving local spatial detail (edges, textures, exact positions). The encoder handles the first by aggregating context over a large receptive field through successive downsampling. The decoder handles the second by restoring resolution step by step. The skip connections are what tie them together: they route high-resolution feature maps from the encoder directly into the decoder, bypassing the bottleneck, so spatial detail is never fully discarded.
 
Conditioning fits naturally into this structure too. Timestep embeddings are injected into the residual blocks via scale-and-shift. Text embeddings enter through cross-attention layers inserted between conv blocks in the decoder. Both can be added at multiple scales, which gives the model fine-grained control over how conditioning affects different levels of spatial detail.

## What is classifier-free guidance?
**T: Deep**

**A:** A single model is trained to handle both conditional and unconditional generation. During training, the conditioning signal $c$ — a class label, text embedding, or any other context — is randomly dropped with some probability and replaced by a null token $\emptyset$. The model therefore learns both $\epsilon_\theta(x_t, c)$ and $\epsilon_\theta(x_t, \emptyset)$ from the same set of weights.
 
At inference, the two predictions are combined:
 
$$\hat{\epsilon} = \epsilon_\theta(x_t, \emptyset) + w \cdot \left( \epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset) \right)$$
 
The term in brackets is the difference between the conditional and unconditional predictions, the direction in noise space that points toward the condition. The guidance scale $w$ amplifies that direction. At $w = 1$ you recover the standard conditional prediction. At $w > 1$ you push further along that direction, increasing prompt adherence at the cost of diversity. At $w = 0$ you get purely unconditional generation.
 
This requires two forward passes per denoising step — one with $c$, one with $\emptyset$, which doubles inference cost. The trade-off is worth it in practice: guidance at $w$ between 7 and 15 is what makes text-to-image models produce sharp, prompt-consistent results.
 
The alternative, classifier guidance, trains a separate classifier on noisy images and uses its gradient to steer generation. Classifier-free guidance folds everything into one network, which is simpler and avoids training a classifier that must handle heavily corrupted inputs.

## What is score matching?
**T: Deep**

**A:** The ideal objective is Fisher divergence — the expected squared difference between the model score $s_\theta(x)$ and the true score $\nabla_x \log p_{data}(x)$:
 
$$L = E \left[ \| s_\theta(x) - \nabla_x \log p_{data}(x) \|^2 \right]$$
 
The problem is that $\nabla_x \log p_{data}(x)$ is unknown. Hyvärinen (2005) showed via integration by parts that this is equivalent to an objective involving only the model score and its Jacobian trace — no true score needed. But computing the Jacobian trace is expensive in high dimensions.
 
Denoising score matching (Vincent, 2011) solves this cleanly. Instead of learning the score of the data distribution, learn the score of a noisy version of the data. Corrupt $x_0$ with Gaussian noise to get $x_t$, then train the network to predict the noise that was added:
 
$$L_{DSM} = E_{x_0, \epsilon, t} \left[ \| s_\theta(x_t, t) + \frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}} \|^2 \right]$$
 
The score of the noisy distribution $\nabla_{x_t} \log p(x_t | x_0)$ is analytically available because $p(x_t | x_0)$ is Gaussian. This is exactly the $\epsilon$-prediction objective in DDPM, diffusion training is denoising score matching at multiple noise levels simultaneously.

## What assumptions are made in diffusion models?
**T: Deep**

**A:** The standard DDPM framework rests on four assumptions.
 
**Markov forward process.** Each noisy step depends only on the previous one, not the full history:
 
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t;\ \sqrt{1 - \beta_t}\, x_{t-1},\ \beta_t I)$$
 
**Gaussian noise at each step.** Because each transition is Gaussian, the marginal at any timestep $t$ has a closed form:
 
$$q(x_t | x_0) = \mathcal{N}(x_t;\ \sqrt{\bar{\alpha}_t}\, x_0,\ (1 - \bar{\alpha}_t) I)$$
 
where $\bar{\alpha}_t$ is the cumulative product of $(1 - \beta_s)$ from $s = 1$ to $t$. This is what enables efficient training — you can sample a noisy $x_t$ from any timestep directly, without simulating the full chain step by step.
 
**Gaussian reverse transitions.** For sufficiently small $\beta_t$, the reverse conditional $p_\theta(x_{t-1} | x_t)$ is approximately Gaussian. This is a classical result from stochastic processes. It justifies parameterizing the reverse process as a Gaussian and learning its mean and variance with a neural network.
 
**Schedule converges to isotropic Gaussian.** The noise schedule is designed so that $\bar{\alpha}_T \approx 0$, meaning $x_T \approx \mathcal{N}(0, I)$. This gives the model a simple known prior to start reverse sampling from.
 
**What breaks when assumptions fail.** If the data is discrete — text, graphs, categorical labels — Gaussian noise is not natural. You cannot smoothly interpolate between tokens the way you can between pixel values. This is why text diffusion requires different forward processes, such as uniform noise or absorbing-state masking.



## What is the reparameterization trick, and why is it needed?
**T: Basic**

**A:** In a VAE, the encoder outputs a mean $\mu_\phi(x)$ and variance $\sigma_\phi(x)$. To train with backpropagation, you need gradients with respect to $\phi$. But if you sample $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ directly, the sampling operation has no gradient.
 
The fix is to write:
 
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
 
Now $\epsilon$ is sampled from a fixed distribution that does not depend on $\phi$. The stochasticity is isolated in $\epsilon$, which is treated as a constant during backpropagation. Gradients flow through $\mu_\phi$ and $\sigma_\phi$ normally.
 
The alternative is REINFORCE — a gradient estimator that does not require reparameterization but has very high variance. In practice it is too noisy to train VAEs reliably.
 
In diffusion, the same trick is used to sample $x_t$ from the forward process:
 
$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$
 
The training target is just $\epsilon$ — the fixed noise sample — which is why the loss reduces to a simple mean squared error between predicted and actual noise.

## How is conditional diffusion handled in diffusion transformers
**T: Deep**

**A:** DiT uses adaptive layer normalization (adaLN) as the primary conditioning mechanism. The timestep embedding and class or text embedding are summed and projected to produce per-layer scale and shift parameters:
 
$$(\gamma, \beta) = \text{Linear}(c + t_{emb}), \quad \hat{x} = \gamma \cdot \text{LayerNorm}(x) + \beta$$
 
These are computed per sample and applied at every transformer block. The conditioning signal directly controls the scale and offset of every normalized activation in the network.
 
The adaLN-zero variant initializes the final linear projection in each block to zero. Every block starts as an identity function and learns to deviate from it gradually. This stabilizes early training when the conditioning signal could otherwise cause large activation shifts before the model has learned anything useful.
 
For dense text conditioning, where the condition is a sequence of token embeddings rather than a single vector — cross-attention is used instead. Image patch tokens serve as queries; text token embeddings serve as keys and values. This is more expressive than adaLN for long, variable-length conditions, but adds cost proportional to the number of text tokens at every block.
 
Token concatenation is a simpler alternative: prepend the conditioning tokens to the patch sequence and let standard self-attention handle everything. It works but forces spatial and conditioning interactions to compete within the same attention heads.
## Why do we induce time and class information in diffusion transformer denoising?
**T: Basic**

**A:** A diffusion model is a single network used at every step of a hundreds-step reverse process. The noise level changes dramatically across those steps, and so does the correct behavior. At $t = T$, the input is nearly pure noise and the model should output a rough prediction of the overall structure. At $t = 1$, the input is almost clean and the model should make tiny, precise corrections. These are fundamentally different operations. Timestep conditioning gives the network the information it needs to switch between them.
 
Class or text conditioning serves a different purpose. Without it, the model learns $p(x)$ — the marginal over all data. With it, the model learns $p(x | c)$ — the distribution conditioned on a specific label or prompt. At each denoising step, the conditioning steers the prediction toward the target class or description rather than toward some average.
 
In DiTs, both are embedded and injected via adaLN. Timestep $t$ is encoded with sinusoidal positional embeddings and passed through a small MLP. Class labels use a learned embedding table. The two are summed and projected to the scale and shift parameters for each layer.
## How do we control generation (e.g., with a pose)?
**T: Deep**

**A:** There are three main families of approaches.
 
**ControlNet.** Zhang et al. (2023) add a parallel encoder branch that takes the structural condition as input. This branch is a copy of the U-Net encoder with its own trainable weights; the original U-Net is frozen. The control branch outputs are added into the U-Net decoder through zero-initialized convolutions. Zero initialization ensures the model starts as identity, at the beginning of training, the control branch contributes nothing, and the frozen backbone generates normally. The branch learns to inject control progressively. For DiT-based models, the same idea applies with parallel transformer blocks.
 
**Cross-attention or token conditioning.** The control signal is encoded into a sequence of embeddings and provided as additional cross-attention context alongside the text embedding. This is architecturally simpler and works well for semantic conditions like style or identity. It is less precise for structural conditions like exact joint positions, because cross-attention does not have an explicit spatial correspondence mechanism.

**Classifier guidance.** A differentiable detector, for example, a pose estimator applied to the predicted $x_0$ at each step — produces a gradient that directly updates $x_t$ during sampling. No additional training is needed. The quality of control depends entirely on how differentiable and accurate the detector is, and it adds a backward pass through the detector at every step.
 

---
