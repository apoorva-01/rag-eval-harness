# Blog Notes: A RAG Evaluation Harness (for apoorvaverma.in)

The chatbot is table stakes. The point of this project is the **evaluation harness** —
and the one piece that makes it stand out from the 10,000 chat-with-PDF repos is a
**custom citation-faithfulness metric** that scores the system's actual standout feature:
inline `[n]` page/section citations.

Corpus: 15 retrieval/RAG/embeddings papers from arXiv (Attention, BERT, DPR, ColBERT,
Contriever, E5, BGE, RAG, BEIR, MS MARCO, …).

## 1. The retrieval lift is real (and not inflated)

Measured over 50 synthetic-but-de-lexicalized questions, single labeled chunk each:

| retriever | NDCG@5 | Recall@5 | MRR |
|---|---|---|---|
| dense (naive baseline) | 0.333 | 0.44 | 0.299 |
| + BM25 hybrid (RRF fusion) | 0.433 | 0.58 | 0.384 |
| + cross-encoder rerank | **0.618** | **0.76** | **0.570** |

Headline: **NDCG@5 0.33 → 0.62, a +86% lift** from naive dense to hybrid+rerank;
Recall@5 0.44 → 0.76. Two reasons this is believable rather than a vanity number:
- The absolutes are modest (0.33–0.62), not pinned near 1.0. The synthesis prompt
  forbade copying phrases from the source passage, so dense embedding match alone only
  gets 0.33 — BM25 recovers lexical signal, the cross-encoder recovers the rest.
- Ground truth is a single positive chunk, so Recall@5 is a **conservative lower bound** —
  when other chunks are also genuinely relevant, recall is under-counted.

## 2. The custom metric is the centerpiece — and it's gold-independent

`CitationFaithfulnessMetric` audits the answer against its *cited* sources only; it never
touches the (circular) synthetic gold set. Framed in attribution-eval (ALCE) terms:
- **citation precision** — of the claims that carry a `[n]`, the fraction whose cited
  source actually states the claim. (When you cite, are you right?)
- **citation recall** — of all factual claims, the fraction that carry ≥1 citation.
  (Do you cite everything you assert?)
Honest refusals ("not in the sources") have no claims to cite and are *excluded*, not
scored 0. Validated by construction: a correctly-cited answer scores 1.0/1.0; an answer
that cites the wrong source and leaves a claim uncited scores precision 0.0 / recall 0.5.

Generation matrix numbers (2 chunking × 2 embedding): _fill from results/generation.md_.

## 3. Chunking × embedding tradeoffs
- fixed vs semantic, bge-small (384-d) vs e5-large (1024-d): _interpret from the matrix —
  which cell wins on faithfulness vs. context recall, and the cost story (e5-large is
  ~3× the embedding time of bge-small)._

## 4. The honest limitation I'd lead with in an interview
The gold set is **synthetic and not hand-verified**, and structurally **circular for
retrieval**: each question is generated from the chunk that is then its only ground truth,
so that chunk tends to rank high under every retriever. I de-lexicalized the questions to
cut keyword leakage, but the structural coupling remains. That's exactly why the headline
signal is the *gold-independent* citation metric — naming this tradeoff (rather than hiding
a suspiciously-high number) is the senior move.

## 5. What's next
- Hand-label a small ground-truth set with multiple positives per query to de-bias Recall@k.
- Add latency/cost per config to quantify the speed↔quality frontier (e5-large vs bge-small).
- A/B alternative rerankers; try semantic chunking with real embeddings-based breakpoints.
