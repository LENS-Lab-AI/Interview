# Models

---

## Word Embeddings in NLP

- Why do we need these encoding? Things are kind of disconnected.
- Add some details for each method.

### Q1. What is one-hot encoding and what are its limitations?
**A:** One-hot encoding represents each word as a binary vector where only one dimension is 1 and all others are 0. While simple, it leads to very large sparse vectors and does not capture any semantic similarity between words (e.g., “king” and “queen” are as different as “king” and “apple”). Additionally, it’s computationally expensive for large vocabularies.

---

### Q2. What is the Bag-of-Words (BoW) model?
**A:** BoW represents text by counting word occurrences without considering order or context. It’s useful for simple models but ignores syntax and semantics.

---

### Q3. What does TF-IDF improve over BoW?
**A:** TF-IDF (Term Frequency–Inverse Document Frequency) downweights common words and highlights distinctive ones by assigning higher weights to rarer but informative words. It captures some notion of word importance across documents.

---

### Q4. How does Word2Vec learn word representations?
**A:** Word2Vec uses neural networks (CBOW or Skip-gram) to learn embeddings that capture semantic relationships by predicting words from their context. It encodes semantic regularities. E.g., ```king - man + woman = queen```.

### Q5. How is GloVe different from Word2Vec?
**A:**
* GloVe (Global Vectors) uses global co-occurrence statistics of words rather than local context windows.
* It provides embeddings that better capture global structure in the corpus.


### Q6. What new idea did RNNs introduce?
**A:** Recurrent Neural Networks introduced recurrence, allowing models to maintain a hidden state across time steps — enabling sequence modeling and use of past information.

### Q7. How do LSTMs improve over RNNs? And how are GRUs different?
**A:** 
* LSTMs introduced gates (input, forget, output) to selectively store or discard information, mitigating the vanishing gradient problem and improving long-term dependency learning.
* GRUs simplify LSTMs by merging gates, offering similar performance with fewer parameters.

### Q8. What is ByteNet?
**A:** ByteNet is a NN which uses convolutional layers for sequence modeling, allowing parallelization and long receptive fields. It served as a bridge between CNNs and transformers.

### Q9. What are some key language transformers?

- Maybe make it like three types of transformer architectures- encode, decoder, and encoder-decoder.

**A:** Major architectures include BERT, RoBERTa, T5, and GPT.

* BERT: Bidirectional encoder, good for understanding tasks (classification, QA).
* RoBERTa: Robustly optimized BERT with better training strategies (no NSP, larger batch sizes).
* T5: Unifies all NLP tasks as text-to-text problems.
* GPT: Decoder-only autoregressive model optimized for generation.

### Q10. How does T5 unify multiple NLP tasks?
**A:** T5 reframes every task (translation, summarization, classification) as text-to-text by adding a task prefix token (e.g., “summarize:” or “translate English to German:”). This unified approach made multitask learning and fine-tuning more generalizable.

### Q11. Why do we need Vision Transformers (ViT)?
**A:** Traditional CNNs rely on local receptive fields, limiting long-range dependencies. ViTs introduce self-attention for global context modeling across image patches, improving performance on large datasets.

### Q12. What changes were introduced in ViT compared to traditional transformers?
**A:**
* ViTs take images as input by splitting them into fixed-size patches.
* A [CLS] token is added to represent the entire image for classification.
* Positional embeddings encode patch order since spatial structure is lost after flattening.

### Q13. How is Swin Transformer different from ViT?
**A:** Swin uses Shifted Window Attention, restricting attention to local windows for better efficiency while still enabling global interactions via window shifting.

### Q14. What is CLIP? How is SigLIP different from CLIP?
**A:**
* CLIP (Contrastive Language-Image Pretraining) aligns text and image embeddings using a contrastive loss to learn cross-modal representations.
* SigLIP modifies the loss to use sigmoid activation instead of softmax on similarity scores, allowing independent scoring.


### Q15. What is RT-2 and why is it significant?
**A:** RT-2 (Robotic Transformer 2) integrates language, vision, and action data, allowing robots to generalize from web-scale multimodal data to physical tasks. It builds on VLM pretraining principles.

### Q16. What is OpenVLA?
**A:** OpenVLA is an open-source Vision-Language-Action model inspired by RT-2, designed for robotic control tasks. It uses curriculum learning to first fine-tune VLMs on physical property understanding before action learning.

### Q17. Why does OpenVLA use Huber loss instead of MSE for fine-tuning?
**A:** Huber loss is quadratic for small errors and linear for large errors, offering robustness to outliers and improved training stability. This is crucial for robotics where noisy sensor readings are common.

### Q18. How could one build their own OpenVLA-style model?
**A:** Start with a pretrained Vision-Language Model (e.g., CLIP or SigLIP) and fine-tune it using a curriculum learning pipeline that progressively introduces physical reasoning and action tasks. Ensure stability via Huber loss and action discretization.


### Q19. Why are standard transformers inefficient for long sequences?
**A:** The self-attention mechanism scales as O(n²) with sequence length, making it memory and compute heavy.

### Q20. What are the main innovations for long-context handling?

- Add references so that we can go to the main paper as well.

**A:**
* Longformer: Uses sparse attention patterns (local + global) to scale linearly with sequence length.

* BigBird: Combines global, random, and local attention, improving efficiency and theoretical expressivity.

* Reformer: Uses LSH attention to reduce complexity from O(n²) to O(n log n).

* Linformer: Projects key and value matrices to lower dimensions.

* Hyena and Mamba: Explore state-space models as transformer alternatives for long-context retention.

### Q21. How do BigBird and Longformer differ in attention and efficiency?
**A:**
* Longformer relies on fixed local + global attention windows.
* BigBird adds random attention connections, which help achieve theoretical universality (can simulate full attention). Also, BigBird typically handles longer contexts with O(n) complexity.

### Q22. What is FlashAttention and why is it faster?
**A:** FlashAttention computes attention in a tiled and memory-efficient way, avoiding explicit storage of the large attention matrix. It fuses softmax computation and reduces I/O bottlenecks, achieving significant speedups on GPUs.

### Q23. What’s new in FlashAttention 2 and 3?
**A:**
* FlashAttention 2: Improved kernel parallelism and supports multi-query attention.
* FlashAttention 3: Adds support for FP8 precision and further pipeline optimizations for large-scale models.

---