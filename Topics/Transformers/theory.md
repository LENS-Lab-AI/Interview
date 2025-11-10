# Theory

---
- 
### Q1. What is Attention?
**A:** Attention is a mechanism that enables a model to identify and emphasize the most relevant components of an input sequence when generating each output. Instead of compressing the entire sequence into a single fixed-length vector (as earlier recurrent models did), attention dynamically computes a **weighted combination** of all input tokens, assigning higher weights to those that are contextually more important for the current output.

Formally, the attention operation takes as input a set of **queries** \( Q \), **keys** \( K \), and **values** \( V \), and computes:


$$
\text{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

In simple terms, each token in a sequence ``looks at'' other tokens and decides **how much attention to pay to each one**.  
This mechanism enables transformers to model **long-range dependencies** directly, without recurrence or convolution, making them highly effective for understanding complex patterns in language, vision, and other domains.


---

### Q2. What are Queries, Keys, and Values (Q, K, V)?
**A:** In the attention mechanism, each input token is transformed into three distinct representations, a **query**, a **key**, and a **value**.  
These are obtained through learned linear projections applied to the same input embeddings or hidden states:

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

where:
- $X \in \mathbb{R}^{n \times d_{\text{model}}}$ is the input matrix containing the token representations,  
- $W_Q, W_K, W_V$ are trainable parameter matrices that project the input into the query, key, and value subspaces.

Each of these serves a different role during attention computation:

- **Query (Q):** Represents the current token’s request for contextual information - ``what am I looking for?''  
- **Key (K):** Represents the available information for each token - ``what do I contain?''  
- **Value (V):** Contains the actual content or representation to be retrieved - ``what should be returned if I’m relevant?''

During attention computation, each query interacts with all keys to determine **relevance scores** through dot products $QK^{\top}$.  
The resulting attention weights are then used to take a weighted sum over the value vectors \( V \), producing an output that integrates relevant context from other tokens.


---

### Q3. Why divide by $\sqrt{d_k}$ in the attention formula?
**A:** The scaling factor $\sqrt{d_k}$ stabilizes the magnitude of the dot products between Queries and Keys.  
Without this normalization, the variance of $QK^{\top}$ increases with the dimensionality $d_k$, which can cause the softmax function to enter regions of very small gradient sensitivity, leading to unstable training.  
Dividing by $\sqrt{d_k}$ ensures that the distribution of attention scores remains well-conditioned and gradients propagate effectively.

---

### Q4. Why is the Softmax function used in Attention?
**A:** The softmax function transforms raw similarity scores into a normalized probability distribution, enabling the model to interpret these scores as relative importances, by exponentiating the similarity scores, softmax amplifies larger scores and suppresses smaller ones. This produces sharp, interpretable attention distributions that focus on the most relevant tokens. Softmax is also smooth and differentiable, which allows efficient gradient-based optimization during training.


---

-why we need this Q5?

### Q5. What is Multi-Head Attention?
**A:** Multi-head attention extends the standard attention mechanism by using multiple parallel attention layers, referred to as ``heads.''  
Each head operates in a distinct learned subspace, allowing the model to capture diverse relational patterns (such as syntactic structure, semantic meaning, or positional context). The outputs of all heads are concatenated and projected back into the model dimension, resulting in richer and more expressive contextual representations.

Formally, given an input matrix  $X \in \mathbb{R}^{n \times d_{\text{model}}}$, multi-head attention computes:

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(H_1, H_2, \ldots, H_h) W_O
$$

where each head $H_i$ is defined as:

$$
H_i = \operatorname{Attention}(X W_Q^{(i)},\, X W_K^{(i)},\, X W_V^{(i)})
$$

Here:
-  $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)} \in \mathbb{R}^{d_{\text{model}} \times d_k}$ are learned projection matrices specific to head $i$.  
- $W_O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ projects the concatenated result back into the model dimension.  
- $h$ denotes the number of attention heads.

---

### Q6. What is Positional Encoding, and why is it necessary?
**A:** Transformers differ from recurrent (RNN) and convolutional (CNN) architectures in that they process all tokens in a sequence **in parallel** rather than sequentially. While this parallelism improves efficiency, it also means that the Transformer lacks any inherent notion of **token order**, every position is treated as identical unless we explicitly encode sequence order information.

To address this, **positional encodings** are added to the input embeddings to inject information about the position of each token in the sequence.  
Formally, for an input sequence of token embeddings $X = [x_1, x_2, \dots, x_n]$, each token representation is modified as:

$$
z_i = x_i + p_i
$$

where $p_i \in \mathbb{R}^{d_{\text{model}}}$ is a positional encoding vector corresponding to position $i$.

This allows it to distinguish between sequences such as dog bites man and man bites dog, although they contain the same tokens.



---

### Q7. What are Sinusoidal Positional Encodings?
**A:** Sinusoidal positional encodings represent each position using deterministic sine and cosine functions of varying frequencies:

$$
\mathrm{PE}_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), 
$$

$$
\mathrm{PE}_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$


This continuous representation allows the model to infer both absolute and relative positional relationships, even for sequences longer than those encountered during training.

---
### Q8. What are Learned and Rotary (RoPE) Positional Encodings?
**A:** Positional information can also be introduced through learned or rotary positional encodings:

- **Learned Positional Encodings:** Each position in the input sequence is associated with a learnable vector that is optimized during training.  
  These embeddings adapt to the data but may not extrapolate effectively to sequences longer than those seen during training.

- **Rotary Positional Encodings (RoPE):** Instead of adding positional vectors, RoPE rotates the query and key representations in a complex plane by position-dependent angles.  
  This operation encodes relative position information directly within the dot-product attention, enhancing the model’s ability to generalize to unseen sequence lengths.

---
- why for all of them, give examples for each.
### Q9. What is the difference between Self-Attention, Cross-Attention, and Encoder–Decoder Attention?
**A:**  
- **Self-Attention:** The queries, keys, and values all originate from the same sequence, enabling tokens to attend to one another within that sequence.  
  This mechanism is used in both the encoder and decoder blocks.

- **Cross-Attention:** Queries come from the decoder, while keys and values are sourced from the encoder outputs.  
  This allows the decoder to condition its generation on encoded input information.

- **Encoder–Decoder Attention:** A broader term describing how the decoder interacts with the encoder representations, crucial for tasks such as translation and summarization.

---

### Q10. Why do Transformers use Layer Normalization instead of Batch Normalization?
**A:** Batch Normalization normalizes across batch dimensions and relies on batch-level statistics, which can vary with batch size or sequence length.  
This dependence makes it unstable for autoregressive generation where batch sizes may be small (even one).  
Layer Normalization, by contrast, normalizes across features within each individual token representation:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \odot \gamma + \beta
$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the feature dimension, and $\gamma, \beta$ are learnable parameters.  
LayerNorm provides stable training dynamics independent of batch size.

---


### Q11. What are Residual Connections and why are they important?
**A:**  
Residual connections add the input of a layer directly to its output, allowing information to bypass nonlinear transformations:

$$
x_{\text{out}} = x_{\text{in}} + f(x_{\text{in}})
$$

They help mitigate the vanishing-gradient problem, improve optimization, and preserve low-level features across deep layers.  
In transformers, residuals are essential for stable convergence and effective gradient flow in very deep architectures.

---
- why important
### Q12. What is Masking in Attention?
**A:** Masking restricts which tokens are visible to a given position during attention computation:

- **Causal Masking:** Ensures that when predicting token $t$, the model cannot attend to future tokens $t{+}1, t{+}2, \dots$ — enforcing the autoregressive constraint.  
- **Padding Masking:** Excludes padded tokens (used to equalize sequence lengths in batches) from the attention computation.  

Masks are implemented by assigning large negative values (e.g., $-\infty$) to the masked logits before applying softmax, effectively zeroing their influence.

---

### Q13. What is the Computational Complexity of Attention?
**A:**  
For a sequence of length $n$ and hidden dimension $d$, standard self-attention has:

$$
\text{Compute Complexity: } O(n^2 d), \qquad
\text{Memory Complexity: } O(n^2)
$$

The quadratic term arises from the pairwise similarity computation between all token pairs.  
This becomes a bottleneck for long sequences, motivating research into efficient variants such as **sparse**, **local**, or **linear** attention mechanisms.

---

### Q14. How can we reduce Attention cost during inference?
**A:** During autoregressive inference, previously computed key and value matrices can be cached.  
When generating a new token, only the query for the current position is computed and used with the stored keys and values:

$$
\text{Attention}(Q_t, K_{\leq t}, V_{\leq t}) = \mathrm{softmax}\left(\frac{Q_t K_{\leq t}^{\top}}{\sqrt{d_k}}\right) V_{\leq t}
$$

This **Key–Value caching** reduces computational cost per step from $O(n^2)$ to $O(n)$ and dramatically accelerates decoding speed.

---

### Q15. What are the Kaplan and Chinchilla Scaling Laws?
**A:** Empirical studies on large language models reveal predictable relationships between model performance, parameter count, data volume, and compute:

- **Kaplan et al. (2020):** Performance improves smoothly as model size, dataset size, and compute budget increase, following approximate power-law relationships.  
- **Chinchilla (2022):** Demonstrated that many models were under-trained for their size; optimal training occurs when the number of training tokens scales proportionally with model parameters.  

These scaling laws guide efficient resource allocation when training large transformer models.

---

### Q16. What is a Learning Rate Scheduler and why is it used?
**A:** A learning rate scheduler controls how the learning rate evolves during training.  
Transformers often employ a **warm-up phase**, where the learning rate increases linearly for the first few thousand steps, followed by a **decay phase** (linear or cosine) as training progresses.  

This scheduling prevents early training instability and encourages smooth convergence to optimal minima.  
A common formulation from the original Transformer paper is:

$$
\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5},\; t \cdot \text{warmup}^{-1.5}\right)
$$

where $t$ is the current step and $\text{warmup}$ is the number of warm-up steps.

---

### Q17. What is a Tokenizer and why is it needed?
**A:** Transformers operate on discrete tokens rather than raw text.  
Tokenization converts character sequences into integer indices suitable for model input. Common methods include:

- **Byte Pair Encoding (BPE):** Iteratively merges the most frequent character pairs into subword units.  
- **Unigram Language Model:** Represents text using a probabilistic model of subwords.  
- **SentencePiece / Byte-Level Encoding:** Works directly on raw text without relying on whitespace.  
- **Token-Free Models:** Operate directly on bytes or characters, eliminating explicit token boundaries.


---
- give examples
### Q18. What is Encoding and Decoding in Transformers?
**A:** The **encoder** maps input sequences into high-dimensional continuous representations capturing semantic and syntactic information.  
The **decoder** generates target sequences autoregressively, attending to both the previously generated tokens (via self-attention) and the encoder outputs (via cross-attention).  
Together, they form the encoder–decoder architecture used in sequence-to-sequence tasks such as translation and summarization.

---
- add with 14
### Q19. What is the difference between Training and Inference Attention?
**A:** During **training**, full sequences are processed simultaneously, allowing self-attention to consider all token pairs within each batch.  
During **inference**, tokens are generated sequentially — each step reuses cached keys and values, avoiding redundant computation.  

---
