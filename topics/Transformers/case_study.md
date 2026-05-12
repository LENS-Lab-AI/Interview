# Case Studies - Sree

---

### Q1. You are building a student assistant that recommends future classes using past courses, likes/dislikes, and goals, with a focus on privacy. What architecture would you use and why?
**T:** Hands-on

**A:** Id use a local-first RAG setup: keep the student’s data encrypted on their device, cache the public course catalog locally, then retrieve candidates with a mix of keyword search and embedding search, apply hard filters (prereqs, schedule, credits), rerank the best matches, and use a small on-device LLM to generate short “why this course” notes. By default nothing leaves the device; if the student opts in, any cloud LLM call sends only public course text plus abstracted preferences (no Personal Information). Optional end-to-end-encrypted sync keeps multiple devices in step. 

---

### Q2. A photographer with Terra Bytes of images needs a private, fast system to browse everything for inspiration. What system will you build?
**T:** Hands-on

**A:** Use on-device image embeddings to turn each photo into a compact “meaning” vector, then a local search system that combines simple filters (date, camera, location, ratings) with semantic similarity so text like “moody blue hour portraits” or “like this photo” finds the right shots quickly; add a small semantic graph layer linking assets to shoots, people, places, styles, and color palettes so the user can say “like these, but at the beach” and hop to related looks, while deduping near-identical frames and diversifying results so the inspiration feed isn’t repetitiv

---

### Q3. Your enterprise RAG assistant still hallucinates even with strong retrieval metrics. What’s going wrong, and how do you fix it without swapping the base model?
**T:** Hands-on

**A:** RAG reduces but doesn’t eliminate hallucinations because LLMs treat retrieved text as suggestions, not binding truth; fix this with retrieval discipline and guardrails: use grounded prompts that require answering only from provided passages and allow abstention; gate outputs on retrieval coverage/confidence with a safe fallback instead of forcing an answer; prefer extractive or cite-then-summarize generation to anchor claims; run a quick post-validation (entailment/string-match) to catch unsupported sentences; improve retrieval (chunking/overlap, reranking, multi-hop, dedup) so the right evidence is present; and surface citations/confidence while closing the loop with human feedback. Hallucinations here are system design issues, not just prompt bugs.

---

### Q4. Your RAG pipeline produces accurate answers, but users complain it’s too slow. What do you do?
**T:** Hands-on

**A:** Start by profiling end-to-end latency (retrieval, rerank, LLM) and set a time budget (e.g., p95 < 3s with first token < 500ms). Then trade redundant context for useful context: cap retrieval to the fewest high-quality chunks that preserve accuracy (often 3–6), shrink chunk size (≈200–400 tokens) with light overlap, dedupe near-identical passages, and pull only the sentence window that supports the claim. Use result caching for recurring or semantically similar queries, and reserve re-ranking for complex questions (route simple ones directly). Add hierarchical retrieval/summary embeddings to compress long sources, and prefer structured/context-aware tools (function calls to fetch just a field) over pasting long documents. On the vector side, tune ANN parameters and embeddings for latency (pre-warm indexes, right nprobe/efSearch, smaller dims or quantization), and parallelize retrieval + chunk fetch while you stream the answer. Gate outputs on coverage/confidence so the model can ask for clarification instead of padding the prompt. You’ll know you’ve hit the sweet spot when p95 latency is under ~3s without a drop in user trust


---

### Q5. What latency metrics should we track?
**T:** Basic

**A:**
- End-to-end p50/p95/p99: pX is the time by which X\% of requests finish
- TTFT (time to first token): Time from send to the first streamed token; the main driver of perceived snappiness.
- TTLT (time to last token): Time from send to the final token; the full wait to completion.
- Tokens/sec (throughput): Average streaming speed after the first token; higher feels smoother for long answers.

---

### Q6. What is FAISS and where can it be used?
**T:** Basic

**A:** FAISS (Facebook AI Similarity Search) is an open-source library from Meta for fast similarity search and clustering over dense vectors; it’s written in C++ with Python bindings, scales from millions to billions of vectors, and offers optional GPU acceleration.
Where it’s used:
- Semantic search & RAG retrieval: find the most relevant passages or items via embedding similarity.
- Recommendations & personalization: nearest-neighbor lookups in embedding space to surface similar products/content.
- Multimedia search (images/audio/video): power reverse-image or “find similar” searches on large media libraries.
- Clustering & large-scale indexing: k-means and ANN indexes for organizing huge vector collections efficiently.

---
### Q7.Imagine you were working on iPhone. Everytime users open their phones, you want to suggest one app they are most likely to open first. How would you do that?
**T:** Deep

**A:** This can be framed as a prediction problem per-unlock, with the goal of maximizing accuracy against a simple baseline such as suggesting each user’s most-used app, possibly conditioned on time-of-day. Using only on-device data for privacy, features would include personal history (per-app frequency, recency, usage streaks), temporal context (time of day, day of week, weekend vs weekday), and device context (location, battery and netwrok stats etc.).The task is modeled as ranking over a small candidate set (e.g., last N opened plus top frequent apps), using a lightweight model such as gradient-boosted trees or a compact neural network trained on historical “unlock to first app opened” sequences, with the highest-scoring app shown as the suggestion. Performance is validated offline and then via online A/B tests.

---

### Q8.You run an e-commerce website. Sometimes, users want to buy an item that is no longer available. Build a recommendation system to suggest replacement items
**T:** Deep

**A:** The problem can be framed as recommending substitute products conditioned on a specific unavailable item and the current user’s context. First, define candidate space by filtering to in-stock items within the same category and similar price range. Then learn product–product similarity using a combination of content-based features (category, brand, attributes, specs, embeddings of titles/descriptions/images). For a given unavailable item, generate a candidate set of nearest neighbors in this product space and re-rank using a model that incorporates the current user’s history (past purchases, browsing, preferences), popularity and quality signals (conversion rate, rating, return rate), and real-time constraints (stock, shipping speed). Evaluate offline with historical substitution events and click/purchase data, and online through A/B tests

---

### Q9. Each question on Quora often gets many different answers. How do you create a model that ranks all these answers? How computationally intensive is this model?
**T:** Deep

**A:** The task can be framed as a learning-to-rank problem where, for a given question, the model orders its candidate answers by predicted usefulness. Features would combine: (a) relevance signals between question and answer (semantic similarity from a transformer encoder over text, keyword overlap, topic match), (b) engagement and quality signals (upvotes, downvotes, comments, time spent, reports, answer length/structure), and (c) author features (historical answer quality, expertise in topic). A pairwise or listwise ranking model (e.g., gradient-boosted trees over features, or a neural ranking model on top of text embeddings) can be trained on historical data where user interactions indicate preference between answers. At serving time, answers for a question are embedded, their features computed or retrieved from caches, and scores produced and sorted; computation is roughly linear in the number of answers per question and can be kept modest by precomputing embeddings and heavy features offline. Overall, training (especially of text encoders) is computationally intensive and done offline on large corpora, but online inference per request is relatively lightweight and can meet latency constraints with proper indexing, caching, and batching.

---

### Q10.  Given only CIFAR-10 dataset, how to build a model to recognize if an image is in the 10 classes of CIFAR-10 or not?
**T:** Deep

**A:** Train a strong classifier on CIFAR-10 (e.g., a CNN) and treat “in vs out of CIFAR-10” as an out-of-distribution detection problem using only in-distribution data. At test time, use the network’s penultimate-layer features or softmax outputs to compute a confidence/energy score for each image; images whose maximum softmax probability (or energy) falls below a chosen threshold are labeled “not in CIFAR-10.” The threshold is selected on a held-out CIFAR-10 validation set plus synthetically perturbed images (e.g., heavy noise, strong augmentations) used as proxy OOD examples.

---

### Q. Sources/more questions
**A:** 
https://huyenchip.com/machine-learning-systems-design/toc.html

---