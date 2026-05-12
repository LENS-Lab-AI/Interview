# Theory — Som

---
## What is Attention?
**T:** Basic

**A:** Attention is a mechanism that enables a model to identify and emphasize the most relevant components of an input sequence when generating each output. Instead of compressing the entire sequence into a single fixed-length vector (as earlier recurrent models did), attention dynamically computes a **weighted combination** of all input tokens, assigning higher weights to those that are contextually more important for the current output.

Formally, the attention operation takes as input a set of **queries** \( Q \), **keys** \( K \), and **values** \( V \), and computes:


$$
\text{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V
$$

In simple terms, each token in a sequence ``looks at'' other tokens and decides **how much attention to pay to each one**.  
This mechanism enables transformers to model **long-range dependencies** directly, without recurrence or convolution, making them highly effective for understanding complex patterns in language, vision, and other domains.


---

## What are Queries, Keys, and Values (Q, K, V)?
**T:** Basic 

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

## Why divide by $\sqrt{d_k}$ in the attention formula?
**T:** Deep

**A:** The scaling factor $\sqrt{d_k}$ stabilizes the magnitude of the dot products between Queries and Keys.  
Without this normalization, the variance of $QK^{\top}$ increases with the dimensionality $d_k$, which can cause the softmax function to enter regions of very small gradient sensitivity, leading to unstable training.  
Dividing by $\sqrt{d_k}$ ensures that the distribution of attention scores remains well-conditioned and gradients propagate effectively.

---

## Why is the Softmax function used in Attention?
**T:** Deep

**A:** The softmax function transforms raw similarity scores into a normalized probability distribution, enabling the model to interpret these scores as relative importances, by exponentiating the similarity scores, softmax amplifies larger scores and suppresses smaller ones. This produces sharp, interpretable attention distributions that focus on the most relevant tokens. Softmax is also smooth and differentiable, which allows efficient gradient-based optimization during training.


---

## What is Multi-Head Attention?
**T:** Basic

**A:** 
Multi-head attention extends the standard attention mechanism by using multiple parallel attention layers, referred to as ``heads.''  
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
### Q5.2. Why do we need Multi-Head Attention?
**T:** Basic

**:A** Single-head attention computes one similarity structure over the entire representation space. This limits the model to focusing on only one type of relationship at a time.

Multi-head attention solves this by allowing the model to:
- Attend to different aspects of the sequence in parallel
- Learn multiple relational subspaces simultaneously

Each head can specialize in capturing different patterns, such as:
- Syntactic relations (e.g., subject–verb agreement)
- Semantic similarity (e.g., coreference)
- Positional or locality-based dependencies

By splitting the model dimension across multiple heads, attention becomes:
- More expressive
- More robust
- Better at modeling complex structures
Without multi-head attention, Transformers show reduced performance and poorer generalization, even with larger hidden dimensions.

---

## What is Positional Encoding, and why is it necessary?
**T:** Basic

**A:** Transformers differ from recurrent (RNN) and convolutional (CNN) architectures in that they process all tokens in a sequence **in parallel** rather than sequentially. While this parallelism improves efficiency, it also means that the Transformer lacks any inherent notion of **token order**, every position is treated as identical unless we explicitly encode sequence order information.

To address this, **positional encodings** are added to the input embeddings to inject information about the position of each token in the sequence.  
Formally, for an input sequence of token embeddings $X = [x_1, x_2, \dots, x_n]$, each token representation is modified as:

$$
z_i = x_i + p_i
$$

where $p_i \in \mathbb{R}^{d_{\text{model}}}$ is a positional encoding vector corresponding to position $i$.

This allows it to distinguish between sequences such as dog bites man and man bites dog, although they contain the same tokens.



---

## What are Sinusoidal Positional Encodings?
**T:** Basic

**A:** Sinusoidal positional encodings represent each position using deterministic sine and cosine functions of varying frequencies:

$$
\mathrm{PE}_{(pos,\,2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), 
$$

$$
\mathrm{PE}_{(pos,\,2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
$$


This continuous representation allows the model to infer both absolute and relative positional relationships, even for sequences longer than those encountered during training.

---
## What are Learned and Rotary (RoPE) Positional Encodings?
**T:** Basic

**A:** Positional information can also be introduced through learned or rotary positional encodings:

- **Learned Positional Encodings:** Each position in the input sequence is associated with a learnable vector that is optimized during training.
These embeddings adapt to the data but may not extrapolate effectively to sequences longer than those seen during training.
Example: In BERT, the sentence “The lawyer questioned the witness because he was nervous” relies on attention between “he” and “lawyer”. If this sentence appears at positions 50-60 during training, the model learns to resolve this coreference using the learned positional embeddings for those positions. However, if the same sentence appears near position 1500 in a long document, the model has never learned embeddings for those positions, and the attention linking “he” to “lawyer” becomes unreliable.

- **Rotary Positional Encodings (RoPE):** Instead of adding positional vectors, RoPE rotates the query and key representations by position-dependent angles before computing attention.
This operation encodes relative position information directly within the dot-product attention.
Example: In models such as LLaMA, if a pronoun and its referent are separated by hundreds of tokens, the attention score between them depends on their relative distance rather than their absolute positions. As a result, the same attention behavior holds whether the sentence appears early or deep within a long document, enabling reliable long-context reasoning.

---
## What is the difference between Self-Attention, Cross-Attention, and Encoder–Decoder Attention?
**T:** Basic

**A:**  
- **Self-Attention:** The queries, keys, and values all originate from the same sequence, enabling tokens to attend to one another within that sequence.  
  This mechanism is used in both the encoder and decoder blocks.

- **Cross-Attention:** Queries come from the decoder, while keys and values are sourced from the encoder outputs.  
  This allows the decoder to condition its generation on encoded input information.

- **Encoder–Decoder Attention:** A broader term describing how the decoder interacts with the encoder representations, crucial for tasks such as translation and summarization.

---

## Why do Transformers use Layer Normalization instead of Batch Normalization?
**T:** Deep

**A:** Batch Normalization normalizes across batch dimensions and relies on batch-level statistics, which can vary with batch size or sequence length.  
This dependence makes it unstable for autoregressive generation where batch sizes may be small (even one).  
Layer Normalization, by contrast, normalizes across features within each individual token representation:

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sigma + \epsilon} \odot \gamma + \beta
$$

where $\mu$ and $\sigma$ are the mean and standard deviation computed over the feature dimension, and $\gamma, \beta$ are learnable parameters.  
LayerNorm provides stable training dynamics independent of batch size.

---


## What are Residual Connections and why are they important?
**T:** Basic

**A:**  
Residual connections add the input of a layer directly to its output, allowing information to bypass nonlinear transformations:

$$
x_{\text{out}} = x_{\text{in}} + f(x_{\text{in}})
$$

They are important because they allow gradients to flow directly through the network, mitigating the vanishing-gradient problem and making very deep transformer models trainable. Residual connections also preserve lower-level representations, enabling each layer to learn incremental refinements rather than entirely new transformations. In Transformers, residual connections stabilize optimization, accelerate convergence, and are essential for maintaining performance as depth increases.

---
## What is Masking in Attention?
**T:** Basic

**A:** Masking restricts which tokens are visible to a given position during attention computation:

- **Causal Masking:** Ensures that when predicting token $t$, the model cannot attend to future tokens $t{+}1, t{+}2, \dots$ — enforcing the autoregressive constraint.  
- **Padding Masking:** Excludes padded tokens (used to equalize sequence lengths in batches) from the attention computation.  

Masks are implemented by assigning large negative values (e.g., $-\infty$) to the masked logits before applying softmax, effectively zeroing their influence.

---

## What is the Computational Complexity of Attention?
**T:** Deep

**A:**  
For a sequence of length $n$ and hidden dimension $d$, standard self-attention has:

$$
\text{Compute Complexity: } O(n^2 d), \qquad
\text{Memory Complexity: } O(n^2)
$$

The quadratic term arises from the pairwise similarity computation between all token pairs.  
This becomes a bottleneck for long sequences, motivating research into efficient variants such as **sparse**, **local**, or **linear** attention mechanisms.

---

## How can we reduce Attention cost during inference?
**T:** Deep

**A:** During autoregressive inference, previously computed key and value matrices can be cached.  
When generating a new token, only the query for the current position is computed and used with the stored keys and values:

$$
\text{Attention}(Q_t, K_{\leq t}, V_{\leq t}) = \mathrm{softmax}\left(\frac{Q_t K_{\leq t}^{\top}}{\sqrt{d_k}}\right) V_{\leq t}
$$

This **Key–Value caching** reduces computational cost per step from $O(n^2)$ to $O(n)$ and dramatically accelerates decoding speed.

---

## What are the Kaplan and Chinchilla Scaling Laws?
**T:** Basic

**A:** Empirical studies on large language models reveal predictable relationships between model performance, parameter count, data volume, and compute:

- **Kaplan et al. (2020):** Performance improves smoothly as model size, dataset size, and compute budget increase, following approximate power-law relationships.  
- **Chinchilla (2022):** Demonstrated that many models were under-trained for their size; optimal training occurs when the number of training tokens scales proportionally with model parameters.  

These scaling laws guide efficient resource allocation when training large transformer models.

---

## What is a Learning Rate Scheduler and why is it used?
**T:** Basic

**A:** A learning rate scheduler controls how the learning rate evolves during training.  
Transformers often employ a **warm-up phase**, where the learning rate increases linearly for the first few thousand steps, followed by a **decay phase** (linear or cosine) as training progresses.  

This scheduling prevents early training instability and encourages smooth convergence to optimal minima.  
A common formulation from the original Transformer paper is:

$$
\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\left(t^{-0.5},\; t \cdot \text{warmup}^{-1.5}\right)
$$

where $t$ is the current step and $\text{warmup}$ is the number of warm-up steps.

---

## What is a Tokenizer and why is it needed?
**T:** Basic

**A:** Transformers operate on discrete tokens rather than raw text.  
Tokenization converts character sequences into integer indices suitable for model input. Common methods include:

- **Byte Pair Encoding (BPE):** Iteratively merges the most frequent character pairs into subword units. Example: “playing” → “play” + “ing”  
- **Unigram Language Model:** Represents text using a probabilistic model of subwords. Example: “unhappiness” may be split into multiple subwords such as “un”, “happi”, and “ness”, depending on corpus statistics.
- **SentencePiece / Byte-Level Encoding:** Works directly on raw text without relying on whitespace. Example: “New York” may be tokenized as a single unit or multiple subwords even without spaces. 
- **Token-Free Models:** Operate directly on bytes or characters, eliminating explicit token boundaries. Example: The word “hello” is represented as individual bytes rather than a word token.


---
## What is Encoding and Decoding in Transformers?
**T:** Basic

**A:** The **encoder** maps input sequences into high-dimensional continuous representations capturing semantic and syntactic information.  
The **decoder** generates target sequences autoregressively, attending to both the previously generated tokens (via self-attention) and the encoder outputs (via cross-attention).  
Together, they form the encoder–decoder architecture used in sequence-to-sequence tasks such as translation and summarization.

---
## What is the difference between Training and Inference Attention?
**T:** Deep

**A:** During **training**, full sequences are processed simultaneously, allowing self-attention to consider all token pairs within each batch.  
During **inference**, tokens are generated sequentially — each step reuses cached keys and values, avoiding redundant computation.  

---


## wha'ts pre-mid-post- training in LLMs? Why do we need these stages?
**T:** Deep

**A:** Large language models are trained in multiple stages to progressively acquire general language understanding, specialized capabilities, and safe, useful behavior.

- **Pre-training:** The model is trained on large-scale, diverse text corpora using a self-supervised objective (next-token prediction). This stage teaches general linguistic knowledge, grammar, factual information, and broad world understanding.

- **Mid-training:** The pre-trained model is further trained on high-quality, curated datasets with a lower learning rate to inject specific capabilities without destroying general knowledge.
This stage is used to improve abilities such as reasoning, long-context understanding, code proficiency, tool use, and agentic behavior. Examples include training on reasoning traces, long-context documents, or task-specific corpora.

- **Post-training:** After pre-training and mid-training, the model is further refined to align its outputs with human expectations of correctness, safety, and usefulness. This stage typically involves Supervised Fine-Tuning (SFT), where the model is trained on high quality human-written instruction–response pairs to improve instruction following and task formatting, and Reinforcement Learning from Human Feedback (RLHF), where human preferences are used to train a reward model that scores model outputs. Post-training does not primarily add new knowledge or reasoning ability; instead, it shapes how existing capabilities are expressed, improving controllability, adherence to instructions, refusal behavior, and overall interaction quality.

----

## What's group query attention (GQA)? 
**T:** Deep

**A:** Grouped Query Attention (GQA) is a self-attention variant designed to reduce inference-time memory and computation by **sharing key and value projections across groups of query heads**, while keeping **separate query projections for each head**. In standard multi-head attention, each of the \(h\) heads has its own query, key, and value projections. In GQA, the model still uses \(h\) distinct query projections, but only \(g < h\) distinct key–value projections. The \(h\) query heads are partitioned into \(g\) groups, and all heads within the same group attend to the same keys and values.

By sharing keys and values across head groups, GQA significantly reduces the size of the key–value cache during autoregressive decoding. This lowers memory usage and improves decoding speed for long-context generation, while retaining more modeling expressiveness than Multi-Query Attention. As a result, GQA provides a practical trade-off between efficiency and performance and is widely used in modern decoder-only large language models such as LLaMA and Mistral.



