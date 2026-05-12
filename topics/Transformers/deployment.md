# Deployment - Eren

---

## How to deploy transformer models faster?
**T:** Deep

**A:**  
- **Quantization:** Reduce weight precision (FP32 → FP16/INT8/INT4) to lower memory bandwidth and accelerate matrix multiplications.  
- **Pruning:** Remove redundant neurons or attention heads to shrink the model while maintaining core behavior.  
- **Knowledge Distillation:** Train a smaller “student” model to mimic a large “teacher,” improving efficiency with minimal performance loss.  
- **Model & Tensor Parallelism:** Distribute model layers or matrix operations across multiple GPUs to handle very large models.  
- **Efficient Transformer Variants:** Use architectures like FlashAttention, Reformer, or MPT-style models to reduce compute bottlenecks.  
- **Batching:** Process multiple requests simultaneously to increase throughput and improve GPU utilization.  
- **Specialized Inference Runtimes:** Use engines like TensorRT, ONNX Runtime, and vLLM for kernel fusion and optimized decoding.

---

## What is batching in model deployment, and how does it improve inference efficiency?

**T:** Deep

**A:**  
Batching processes multiple input sequences together in a single forward pass. Since GPUs excel at parallel computation, batching increases throughput and reduces kernel-launch overhead.  
For example, encoding 32 prompts together may be only slightly slower than processing one, yielding ~20–30× better throughput.  
However, large batches can increase **tail latency**, so production systems use **dynamic or adaptive batching** to maximize efficiency while keeping requests responsive.

---

## What is speculative decoding in LLM deployment?
**T:** Deep

**A:**  
Speculative decoding uses two models:  
- A small, fast **draft model** proposes multiple future tokens.  
- A large **target model** verifies or rejects these tokens in parallel.  
Accepted tokens allow the model to “jump ahead,” producing 2–4× faster inference with minimal accuracy loss.  
This method greatly accelerates autoregressive decoding without retraining the main model.

---

## Given an LLM, how can we ensure that it can scale to handle large context?
**T:** Basic

**A:**  
Scalability requires separating retrieval from computation. Retrieval-Augmented Generation (RAG) stores knowledge in vector DBs, enabling models to handle large domains without expanding their context windows.  
Chunking keeps retrieval efficient at scale.  
Horizontal model replicas, sharded vector indexes, KV-cache reuse, and continuous batching allow the system to serve high concurrency with predictable latency.

---

## What are the main challenges in deploying LLMs?
**T:** Basic

**A:**  
1. **Latency:** Autoregressive generation is sequential and slow.  
2. **GPU Memory Limitations:** Large models require multi-GPU inference or quantization.  
3. **Operational Cost:** Running GPU clusters is expensive.  
4. **Concurrency:** High traffic requires batching, caching, and schedulers.  
5. **Safety & Alignment:** Preventing harmful outputs requires guardrails.  
6. **Continuous Updating:** Updates to weights and policies must happen without service downtime.

---

## Why is RAG better than giving the model all the information at once? (Example: pasting an entire book into the prompt)
**T:** Basic

**A:**  
Transformer attention scales quadratically (O(n²)), so extremely long inputs cause slow, memory-heavy inference and costly prompts.  
Example: pasting a 200-page PDF forces the model to read thousands of irrelevant tokens.

RAG instead retrieves only the **most relevant chunks**, which:  
- Reduces latency  
- Lowers token cost  
- Prevents GPU memory exhaustion  
- Improves answer quality  
- Allows unlimited knowledge scaling without needing larger context windows  

RAG fetches exactly what is needed and lets the LLM reason over concise, focused information.

---

## How to handle when longer outputs are required in LLMs?
**T:** Deep

**A:**  
- **Sliding-window attention:** Restrict attention to recent tokens.  
- **Chunked generation:** Produce text in segments, summarize, and continue.  
- **Memory-augmented transformers:** Store long-range info outside the context window.  
- **Streaming:** Send partial outputs to reduce perceived delay.  
These techniques support long transcripts or documents efficiently.

---

## How to make an LLM safer?
**T:** Deep

**A:**  
Imagine safety as a multi-layer security gate:

1. **Prompt guardrails** stop dangerous requests before they reach the model.  
   *Example:* A user asks:  
   *“How can I bypass airport security?”*  
   The guardrail rewrites/blocks it.

2. **Output guardrails** inspect the model’s reply for toxicity, bias, or unsafe content.

3. **Policy models** act like expert reviewers scoring outputs for safety violations or hallucinations.

Additional layers include red-teaming pipelines, human-in-the-loop checks, RLHF tuning, audit logs, and domain-specific policies.  
Together these ensure responsible model behavior.

---

## What is model quantization and why is it important?
**T:** Basic

**A:**  
Quantization converts FP32/FP16 weights into INT8/INT4/NF4 formats, reducing memory and compute cost. An 8-bit model often uses 4× less RAM and runs nearly 2× faster.

**Story:** Deploying a full-precision LLM on a small device is like carrying a giant suitcase on a tiny scooter—technically possible but slow and unstable. Quantization shrinks it into a backpack: lighter, more efficient, and easy to use anywhere.

Modern PTQ/QAT techniques preserve accuracy, making quantization essential for edge devices and economical large-scale inference.

---

## What is pruning in transformer models?
**T:** Deep

**A:**  

Pruning removes weights, neurons, or attention heads that contribute little to model quality.  
- **Structured pruning:** Removes whole heads/channels and accelerates real hardware.  
- **Unstructured pruning:** Zeros individual weights (sparse), requiring special kernels.  
Pruning reduces FLOPs, memory, and latency, especially when paired with fine-tuning or distillation.

---

### Q11. What is tensor parallelism?
**T:** Deep

**A:**  
Tensor parallelism splits large matrix multiplications (QKV projections, FFN layers) across multiple GPUs.  
Each GPU computes a shard of the operation, then synchronizes results.  
This enables extremely large models (30B–400B) to run efficiently without exceeding a single GPU’s memory.

---

### Q12. What is pipeline parallelism?
**T:** Deep

**A:**  
Pipeline parallelism assigns different transformer layers to different GPUs.  
Micro-batches are passed through the pipeline so each GPU stays busy.  
The 1F1B schedule reduces idle “bubbles,” improving throughput for deep models.

---

### Q13. What is KV-cache optimization?
**T:** Deep

**A:**  
KV-cache stores past key/value tensors so the model doesn’t recompute them every token.  
This reduces per-token compute from O(n) → O(1).  
Optimizations include cache quantization, paged layouts, and flash attention integration—crucial for high-concurrency inference.

---

### Q14. What is FlashAttention?
**T:** Basic

**A:**  
FlashAttention minimizes GPU memory traffic by keeping Q/K/V on-chip during computation.  
This fused-kernel design yields 2–3× faster attention and supports longer sequences without running out of memory.  
It is the default attention kernel in modern LLMs.

---

### Q15. How do VLMs optimize image processing during inference?
**T:** Deep

**A:**  
- **Image embedding caching**  
- **Lightweight preview encoders**  
- **Downsampling / patch compression**  
- **Sharing embeddings across queries**  
These techniques accelerate the vision front-end.

---

### Q16. What is batching in VLMs?
**T:** Basic

**A:**  
VLMs batch multiple images or image-text pairs to fully utilize GPU parallelism.  
This reduces per-image latency and increases throughput, especially under heavy traffic.

---

### Q17. How to reduce latency for very long context windows?
**T:** Deep

**A:**  
- **Sliding-window attention**  
- **Sparse attention (Longformer, BigBird, Mistral)**  
- **RoPE/NTK scaling**  
- **Hierarchical compression of old tokens**  
These techniques make 100k–1M token windows feasible.

---

### Q18. What is distillation for multimodal models?
**T:** Basic

**A:**  
Distillation trains a smaller VLM to mimic a larger one by aligning:  
- Logits  
- Image embeddings from the vision tower  
- Cross-modal attention maps  
- Hidden states  
This reduces memory, speeds up inference, and preserves multimodal reasoning.

---

### Q19. What is a serving engine (vLLM, TGI, TensorRT-LLM)?
**T:** Deep

**A:**  
A serving engine is an optimized runtime that manages batching, KV-cache, scheduling, and GPU kernels for fast inference.

**Advantages:**  
- **vLLM:** Paged attention, continuous batching, very high throughput  
- **TGI:** Production features, multi-model serving, stable APIs  
- **TensorRT-LLM:** NVIDIA-optimized kernels, CUDA graphs, fastest GPU performance

---

### Q20. What is paged attention in vLLM?
**T:** Deep

**A:**  
Paged Attention stores KV-cache in fixed-size pages, like virtual memory.  
Benefits include:  
- Efficient massive batching  
- Zero fragmentation  
- Low-latency scheduling  
This is a core reason vLLM scales so well.

---

### Q21. How do we deploy LLMs on edge devices?
**T:** Basic

**A:**  
- INT8/INT4/INT3 quantization  
- Structured pruning  
- Distillation  
- ONNX/CoreML/TFLite compilation  
- Operator fusion  
These allow LLMs to run on phones, edge GPUs, and embedded boards.

---

### Q22. How do we make an LLM deployment resilient so that the system keeps working even when parts of it fail?
**T:** Deep

**A:**  
- **Replication** of model servers  
- **Cross-zone deployment**  
- **Checkpointing** for fast recovery  
- **Retries & smart routing**  
- **Graceful degradation** (fallback to smaller models)  
These ensure uptime during hardware or network failures.

---

### Q23. How to monitor deployed LLMs in production?
**T:** Basic

**A:**  
Track:  
- Latency (p50/p90/p99)  
- Throughput, queue depth  
- GPU utilization  
- Token generation rate  
- Safety violations & hallucinations  
- RAG metrics (recall, hit-rate)  
- User feedback signals  
Monitoring ensures stable, safe, high-performance deployments.

---

### Q24. How to scale LLM APIs to millions of users?
**T:** Deep

**A:**  
- Load balancing  
- Autoscaling  
- Continuous batching & KV-cache sharing  
- Multi-tenant scheduling  
- Model sharding  
- Geo-distributed inference nodes  
This architecture supports global-scale traffic.

---

### Q25. What is the main bottleneck in VLM deployment? How to mitigate?
**T:** Basic

**A:**  
High-resolution image encoding is usually the slowest step.  
Mitigations:  
- Pre-encoding  
- Downsampling  
- Preview encoders  
- Sharing embeddings  
Optimizing the vision tower significantly improves speed.

---

### Q29. What personalization methods exist during deployment (online personalization)?
**T:** Basic

**A:**  
- RAG-based user memory  
- Soft prompting / prefix tuning per user  
- Routing to user-specific adapters  
- Contextual learning with stored preferences  
- Online RLHF-lite  
- Session-level embeddings  
These enable immediate personalization without retraining.

---

### Q31. How do we evaluate personalization quality in LLMs?
**T:** Basic

**A:**  
- User-profile accuracy  
- Relevance score  
- Consistency over long sessions  
- Wrong-personalization error rate  
- Engagement metrics  
- Retrieval metrics (recall@K, hit-rate, similarity)  
Evaluation combines metrics and user feedback loops.

---
