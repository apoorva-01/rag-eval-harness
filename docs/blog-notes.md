# Blog Notes: RAG Evaluation Harness

## Post: Building a Modular RAG Evaluation Framework

### NDCG Lift Pattern
- Dense retrieval baseline (naive in-order search): _NDCG@5 to be measured_
- BM25 hybrid fusion (lexical + semantic): _lift vs. dense to be measured_
- Cross-encoder reranking (fine-tuned ranking): _final NDCG@5 and lift to be measured_
- Target story: Show that reranking captures the long tail of relevant-but-not-top-k documents

### Chunking Strategy Tradeoffs
- **Fixed-size (512 tokens):** Consistent boundaries, reproducible splits, stable embeddings across runs
  - Observed impact on retrieval: _to be measured_
  - Observed impact on faithfulness: _to be measured_
- **Semantic (sentence-greedy, no LLM):** Packs whole sentences up to a ~700-char cap, breaks on section boundaries; higher variance in chunk size than fixed-size
  - Observed impact on retrieval: _to be measured_
  - Observed impact on faithfulness: _to be measured_
- Decision logic: When does semantic chunking improve recall? When does fixed-size win on speed/consistency?

### Embedding Model Comparison
- **bge-small-en-v1.5 (384-dim, lightweight):**
  - Performance: _to be measured_
  - Model size and inference speed: _to be measured_
  - Tradeoffs: Fast but lower semantic capacity
- **e5-large-v2 (1024-dim, larger):**
  - Performance: _to be measured_
  - Model size and inference speed: _to be measured_
  - Tradeoffs: Slower but richer representations
- Which wins on Context Precision vs. Context Recall?

### Surprises & Failure Modes
- _(Document unexpected behaviors or edge cases discovered during evaluation)_
- Did any embedding/chunking combo degrade unexpectedly?
- Did the gold standard reveal annotation issues?
- Did cross-encoder reranking sometimes hurt NDCG on out-of-domain queries?

### What's Next
- Extend the evaluation to multi-document retrieval (how well does the harness handle cross-paper references?)
- A/B test with different reranker models (e.g., mmarco vs. nli-deberta)
- Measure end-to-end latency per config; quantify the speed/quality frontier
- Add batch evaluation of in-the-wild user queries (if available)
- Integrate human-in-the-loop feedback loop for gold standard refinement
