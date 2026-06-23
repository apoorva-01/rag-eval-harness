# RAG System + Evaluation Harness — Design

**Date:** 2026-06-24
**Status:** Approved (design), pending implementation plan

## Goal

A "chat with a research-paper set" RAG system whose real deliverable is a **systematic
evaluation harness**. The chatbot is table stakes; the signal layer — measured retrieval
and answer-quality metrics across a comparison matrix, reported against a baseline — is the
project. The README's results tables are what distinguish this from the field of
chat-with-PDF repos.

## Locked decisions

| Axis | Choice |
|---|---|
| Corpus | Research papers — a set of arXiv ML PDFs |
| Eval framework | DeepEval (pytest-native) |
| UI | Streamlit |
| LLM provider | Claude (Anthropic API) |
| Pipeline | Thin custom pipeline over sentence-transformers, rank-bm25, Chroma, Anthropic SDK |
| Embedding models (matrix) | `BAAI/bge-small-en-v1.5` vs `intfloat/e5-large-v2` |
| Chunking strategies (matrix) | Fixed-size vs Semantic |
| Gold set | LLM-synthesized with Claude, hand-corrected, committed to repo |
| Deploy | Hugging Face Spaces |

## Models

- **LLM-judge metrics (DeepEval):** `claude-opus-4-8` — faithfulness and relevance judging
  needs the strongest judge.
- **Answer generation:** defaults to `claude-opus-4-8`, exposed as a single config constant
  (`GEN_MODEL`) so it can drop to `claude-sonnet-4-6` if eval-run cost climbs.
- **Embeddings + reranker:** local via sentence-transformers — `bge-small`, `e5-large`, and a
  `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker. No API cost on the retrieval path.

Anthropic calls use the SDK with `thinking={"type": "adaptive"}` where reasoning helps
(synthesis, judging) and default settings otherwise. Model IDs are bare (no date suffix).

## Architecture

A thin pipeline of swappable components. The eval matrix *is* the project, so every stage is
cleanly substitutable.

```
rag/
  ingest.py      # PDF -> page-aware text blocks (pymupdf); keeps page + section
  chunk.py       # ChunkStrategy: FixedSize | Semantic -> list[Chunk]
  embed.py       # Embedder: BGEsmall | E5large (sentence-transformers)
  store.py       # Chroma collection wrapper (one collection per config)
  retrieve.py    # Retriever: Dense | BM25 | Hybrid(+cross-encoder rerank)
  generate.py    # Claude answer with inline [p.N] citations, grounded prompt
  pipeline.py    # RAGPipeline: wires the above; one object per experiment cell
config.py        # frozen dataclasses defining each experiment cell
eval/
  gold.py        # synthesize + load gold Q/A/ground-truth-context set
  metrics.py     # DeepEval metric wiring (faithfulness, relevance, ctx p/r)
  test_rag.py    # pytest: per-config assertions + NDCG@5 retrieval comparison
  run_matrix.py  # runs all cells; writes results table (md + csv)
app/
  streamlit_app.py
data/
  papers/        # the corpus (PDFs)
  gold.jsonl     # committed gold set
```

### Core data model

Each `Chunk` carries `{text, paper_id, page, section, chunk_id}`. The page/section provenance
threads from ingest through retrieval into the generation prompt, so every cited claim
resolves to an exact page/section — the standout feature.

### Two orthogonal comparison axes

- **`Retriever`** is the axis for the retrieval/NDCG story: naive Dense -> +BM25 hybrid ->
  +cross-encoder rerank.
- **`ChunkStrategy` x `Embedder`** is the 2x2 answer-quality matrix.

Keeping these orthogonal lets the README report both stories cleanly without confounding them.

## The eval matrix (the deliverable)

Two tables in the README, both produced by `run_matrix.py` from the same gold set.

**Table A — Retrieval (baseline -> senior), held at one chunk/embed config:**

| Retriever | NDCG@5 | Recall@5 | MRR |
|---|---|---|---|
| Dense only (naive) | baseline |
| + BM25 hybrid | |
| + cross-encoder rerank | best |

NDCG@5 / Recall@5 / MRR are computed directly against gold ground-truth chunk IDs —
deterministic, fast, no LLM.

**Table B — Generation quality (2 chunking x 2 embedding = 4 cells), held at Hybrid+rerank:**

| Chunking x Embedding | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|---|
| Fixed x BGE-small | |
| Fixed x E5-large | |
| Semantic x BGE-small | |
| Semantic x E5-large | |

The four DeepEval metrics use the Claude judge.

## Gold set & DeepEval/pytest integration

- `gold.py` synthesizes ~40–60 `(question, reference_answer, ground_truth_chunk_ids,
  source_pages)` triples with Claude from the ingested chunks, written to `data/gold.jsonl`
  for a one-time hand-correction pass. Committed so runs are reproducible.
- `test_rag.py` is real pytest: parametrized over configs, asserting each metric clears a
  threshold (e.g. `faithfulness >= 0.8`) — a CI-able quality gate.
- `run_matrix.py` is the reporting path (full table, no assertions).

## Build order

Get a measurable baseline fast, then layer:

1. ingest -> chunk -> embed -> Chroma -> Dense retrieve -> generate -> **one gold question
   end-to-end**
2. BM25 + hybrid + cross-encoder rerank
3. gold-set synthesis + hand-correction
4. DeepEval metrics + pytest gate
5. `run_matrix.py` + both result tables
6. Streamlit app
7. README (tables + chunking/embedding comparison + "what I learned") + blog post
8. HF Spaces deploy

## Constraints & conventions

- Secrets via `.env` (`ANTHROPIC_API_KEY`); never committed.
- One Chroma collection per `(chunk, embed)` config to keep experiment cells isolated.
- Corpus: a fixed, committed set of arXiv ML PDFs (count finalized at ingest; target ~10–20
  papers so the gold set is meaningful but runs stay fast).
- Local models (embedders, reranker) cached on first download; HF Spaces gives enough RAM for
  them (the reason for choosing Spaces over Streamlit Community Cloud).

## What I learned (README section — to fill during build)

Reserved for the empirical findings: the baseline->hybrid+rerank NDCG lift, which
chunk/embed cell wins on faithfulness vs. recall, and the failure modes observed.
