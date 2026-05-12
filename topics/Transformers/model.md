# Models - Aditya

---

## Word Embeddings in NLP

### Q1. Why do we need word embeddings/encoding?

**T:** Basic

**A:** Neural models cannot directly operate on text. We need numerical representations that (1) are computationally feasible and (2) preserve semantic structure. The evolution from one-hot → dense embeddings reflects a shift from symbolic identity to distributed meaning.

### Q2. What is one-hot encoding and what are its limitations?

**T:** Basic

**A:** One-hot encoding represents each word as a binary vector where only one dimension is 1 and all others are 0. While simple, it leads to very large sparse vectors and does not capture any semantic similarity between words (e.g., “king” and “queen” are as different as “king” and “apple”). Additionally, it’s computationally expensive for large vocabularies.

### Q3. What is the Bag-of-Words (BoW) model?

**T:** Basic

**A:** BoW represents a document as a vector of word counts. It is good at capturing:
* Word presence and frequency, and
* Simple topical information

But, it ignores word order (syntax), context and polysemy, and semantics. Thus, BoW improves over one-hot at the document level, but still treats words independently.

### Q4. What does TF-IDF improve over BoW?

**T:** Basic

**A:** TF-IDF (Term Frequency–Inverse Document Frequency) improves over BoW by downweights common words and highlights distinctive ones by assigning higher weights to rarer but informative words. It captures some notion of word importance across documents. 

Limitation of TF-IDF is, it is still sparse, context-free, and no semantic similarity is considered.

### Q5. How does Word2Vec learn word representations?

**T:** Deep

**A:** Word2Vec learns dense, low-dimensional embeddings by exploiting the distributional hypothesis:
> Words that appear in similar contexts have similar meanings.

It uses neural networks (CBOW or Skip-gram) to learn embeddings that capture semantic relationships by predicting words from their context. It has two training objectives:
* CBOW: predict a word from its context.
* Skip-gram: predict context words from a target word.

This yields embeddings with linear semantic structure and it can encode semantic regularities. E.g., ```king - man + woman = queen```.

### Q6. How is GloVe different from Word2Vec?

**T:** Basic

**A:**
* GloVe (Global Vectors) uses global co-occurrence statistics of words rather than local context windows.
* It provides embeddings that better capture global structure in the corpus.

## Pre-Transformer Models

<!-- **Motivation:**
Bag-based models ignore order. Sequence models aim to capture temporal structure. -->

### Q7: What new idea did RNNs introduce?

**T:** Basic

**A:** Recurrent Neural Networks introduced recurrence, allowing models to maintain a hidden state across time steps — enabling sequence modeling and use of past information.

### Q8: How do LSTMs improve over RNNs? And how are GRUs different?

**T:** Basic

**A:** 
* LSTMs introduced gates (input, forget, output) to selectively store or discard information, mitigating the vanishing gradient problem and improving long-term dependency learning.
* GRUs simplify LSTMs by merging gates, offering similar performance with fewer parameters.

### Q9: What is ByteNet?

**T:** Deep

**A:** ByteNet is a NN which uses convolutional layers for sequence modeling, allowing parallelization and long receptive fields. It served as a bridge between CNNs and transformers.

## Language Transformers

### Q10. What are some key language transformers?

<!-- - Maybe make it like three types of transformer architectures- encode, decoder, and encoder-decoder. -->

**T:** Basic

**A:** Major architectures include BERT, RoBERTa, T5, and GPT.

* BERT: Bidirectional encoder, good for understanding tasks (classification, QA).
* RoBERTa: Robustly optimized BERT with better training strategies (no NSP, larger batch sizes).
* T5: Unifies all NLP tasks as text-to-text problems.
* GPT: Decoder-only autoregressive model optimized for generation.

### Q11: How does T5 unify multiple NLP tasks?

**T:** Deep

**A:** T5 reframes every task (translation, summarization, classification) as text-to-text by adding a task prefix token (e.g., “summarize:” or “translate English to German:”). This unified approach made multitask learning and fine-tuning more generalizable.

## Multimodal and Action Transformers

### Q12: Why do we need Vision Transformers (ViT)?

**T:** Deep

**A:** Traditional CNNs rely on local receptive fields, limiting long-range dependencies. ViTs introduce self-attention for global context modeling across image patches, improving performance on large datasets.

### Q13: What changes were introduced in ViT compared to traditional transformers?

**T:** Deep

**A:**
* ViTs take images as input by splitting them into fixed-size patches.
* A [CLS] token is added to represent the entire image for classification.
* Positional embeddings encode patch order since spatial structure is lost after flattening.

### Q14: How is Swin Transformer different from ViT?

**T:** Basic

**A:** Swin uses Shifted Window Attention, restricting attention to local windows for better efficiency while still enabling global interactions via window shifting.

### Q15: What is CLIP? How is SigLIP different from CLIP?

**T:** Basic

**A:**
* CLIP (Contrastive Language-Image Pretraining) aligns text and image embeddings using a contrastive loss to learn cross-modal representations.
* SigLIP modifies the loss to use sigmoid activation instead of softmax on similarity scores, allowing independent scoring.
    <!-- * Using SigLIP may introduce small inconsistencies due to non-normalized similarities. -->

### Q16: What is RT-2 and why is it significant?

**T:** Basic

**A:** RT-2 (Robotic Transformer 2) integrates language, vision, and action data, allowing robots to generalize from web-scale multimodal data to physical tasks. It builds on VLM pretraining principles.

### Q17: What is OpenVLA?

**T:** Deep

**A:** OpenVLA is an open-source Vision-Language-Action model inspired by RT-2, designed for robotic control tasks. It uses curriculum learning to first fine-tune VLMs on physical property understanding before action learning.

### Q18: Why does OpenVLA use Huber loss instead of MSE for fine-tuning?

**T:** Hands-On

**A:** Huber loss is quadratic for small errors and linear for large errors, offering robustness to outliers and improved training stability. This is crucial for robotics where noisy sensor readings are common.

### Q19: How could one build their own OpenVLA-style model?

**T:** Hands-On

**A:** Start with a pretrained Vision-Language Model (e.g., CLIP or SigLIP) and fine-tune it using a curriculum learning pipeline that progressively introduces physical reasoning and action tasks. Ensure stability via Huber loss and action discretization.


## Efficient and Long-Context Transformer Models

### Q20: Why are standard transformers inefficient for long sequences?

**T:** Basic

**A:** The self-attention mechanism scales as O(n²) with sequence length, making it memory and compute heavy.

### Q21: What are the main innovations for long-context handling?

**T:** Deep

<!-- - Add references so that we can go to the main paper as well. -->

**A:**
* Longformer: Uses sparse attention patterns (local + global) to scale linearly with sequence length.

* BigBird: Combines global, random, and local attention, improving efficiency and theoretical expressivity.

* Reformer: Uses LSH attention to reduce complexity from O(n²) to O(n log n).

* Linformer: Projects key and value matrices to lower dimensions.

* Hyena and Mamba: Explore state-space models as transformer alternatives for long-context retention.

### Q22: What are Adaptive Base Frequency (ABF) methods in LLM for handling long context?

**T:** Deep

**A:** ABF methods are techniques used to extend the context window of models that use RoPE (Rotary Positional Embeddings).

* **The Problem:** RoPE encodes position by rotating the embedding vector. If a model encounters a sequence longer than what it was trained on (extrapolation), the rotation angles go "out of distribution," confusing the model.The
* **Solution (ABF):** ABF modifies the base frequency ($\theta$, typically 10,000) of the rotation. By increasing this base frequency (e.g., to 500k or 1M), the method "stretches" the wavelengths of the positional encodings.
* **Result:** This maps the new, longer positions back into the range of values the model saw during training. It allows models (like CodeLlama or Llama 2 Long) to handle significantly longer contexts (e.g., 100k tokens) with minimal fine-tuning.

### Q23: How is a model trained or adapted using ABF?

**T:** Deep

**A:** ABF is rarely used zero-shot; it typically requires a Fine-Tuning (FT) phase to adapt the model to the new ``resolution'' of positions. This step is also referred to as mid-training. It is as follows:

1. **Initialize:** You start with the pre-trained weights/model.
2. **Scale Config:** Change the RoPE configuration (increase base $\theta$ or set a scale factor $s$).
3. **Fine-Tune:** Train on a small dataset of long sequences (e.g., 10k–100k tokens).
    * This works because the model does not need to relearn language semantics or grammar. It only needs to adapt its attention mechanism to the new positional scale.
    * Convergence happens very quickly (often just a few hundred steps) compared to training from scratch.

**Data Mix:** We follow 30/70 rule.
* ~30\% Long Data: High-quality long documents (books, repositories) to teach the model to use the new long positions.
* ~70\% Short Data: The original pre-training data. This is crucial to anchor the model's semantics and prevent catastrophic forgetting of short-term reasoning.

### Q24: How do BigBird and Longformer differ in attention and efficiency?

**T:** Basic

**A:**
* Longformer relies on fixed local + global attention windows.
* BigBird adds random attention connections, which help achieve theoretical universality (can simulate full attention). Also, BigBird typically handles longer contexts with O(n) complexity.

### Q25: What is FlashAttention and why is it faster?

**T:** Deep

**A:** FlashAttention computes attention in a tiled and memory-efficient way, avoiding explicit storage of the large attention matrix. It fuses softmax computation and reduces I/O bottlenecks, achieving significant speedups on GPUs.

### Q26: What’s new in FlashAttention 2 and 3?

**T:** Basic

**A:**
* FlashAttention 2: Improved kernel parallelism and supports multi-query attention.
* FlashAttention 3: Adds support for FP8 precision and further pipeline optimizations for large-scale models.

---


### Q27: How is the token budget calculate (theoretically and empirically) in an LLM?

**T:** 

**A:** 