# RAG Evaluation Harness

A modular evaluation framework for retrieval-augmented generation (RAG) systems, measuring retrieval quality (NDCG, Recall, MRR) and generation fidelity (faithfulness, relevance, context precision/recall) across chunking and embedding strategies.

## Results

### Retrieval Performance

| Retriever | NDCG@5 | Recall@5 | MRR |
|-----------|--------|----------|-----|
| Dense (naive baseline) | — | — | — |
| Dense + BM25 hybrid | — | — | — |
| Hybrid + cross-encoder rerank | — | — | — |

*Populated by `python eval/run_matrix.py` after adding PDFs and setting ANTHROPIC_API_KEY.*

The evaluation harness is designed to surface the retrieval lift pattern: Dense retrieval establishes a baseline, BM25 hybrid fusion improves recall, and cross-encoder reranking optimizes NDCG@5. Concrete numbers will appear after a full run on your corpus.

### Generation Quality (Chunking × Embedding Matrix)

| Config (chunk × embed) | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|------------------------|--------------|------------------|--------------------|-----------------|
| fixed × bge_small | — | — | — | — |
| fixed × e5_large | — | — | — | — |
| semantic × bge_small | — | — | — | — |
| semantic × e5_large | — | — | — | — |

*Populated by `python eval/run_matrix.py` after adding PDFs and setting ANTHROPIC_API_KEY.*

## Architecture

The codebase is organized into ingestion/chunking/embedding (`rag/`) and evaluation (`eval/`) stages:

```
rag/
├── ingest.py           # Load PDFs from data/papers/
├── chunk.py            # Fixed-size and semantic (LLM-driven) chunking
├── embed.py            # Sentence-Transformers embeddings (bge-small, e5-large)
├── store.py            # Chroma vector database write/read
├── retrieve.py         # Dense, BM25 hybrid, cross-encoder reranking
├── generate.py         # Claude context synthesis and answer generation
└── pipeline.py         # Orchestrate ingest→chunk→embed→store→retrieve→generate

eval/
├── gold.py             # Gold standard test set (LLM-synthesized, hand-checked)
├── metrics.py          # DeepEval integration for faithfulness/relevance/precision/recall
├── retrieval_metrics.py # NDCG, Recall@k, MRR computation
├── run_matrix.py       # Run all 2 chunk × 2 embed configs; aggregate results
├── claude_judge.py     # Judge implementation for DeepEval
└── test_rag.py         # Integration tests
```

## Chunking × Embedding Comparison

Fixed-size chunking enforces consistent boundaries (e.g., 512 tokens), yielding stable, reproducible splits. Semantic chunking uses an LLM to identify logical breakpoints, potentially capturing paragraph-level semantics at the cost of variable chunk sizes. The embedding model choice (bge-small: 384-dim, lightweight; e5-large: 1024-dim, higher capacity) affects both retrieval precision and generation context quality. The matrix evaluates all four combinations to identify the optimal tradeoff for your use case: smaller models train faster and consume less memory, while larger models often capture richer semantic relationships. Watch for the faithfulness × recall tension — more context improves answer relevance but can dilute critical facts, especially with verbose chunks.

## What I Learned

_Findings and failure modes to be populated after running the evaluation harness:_

- Baseline accuracy and failure modes
- Chunking strategy tradeoffs (size, semantics, retrieval quality)
- Embedding model impact (model size vs. semantic capacity)
- Reranking lift and generalization
- Gold set reliability and bias sources

## Run It

### 1. Create a Virtual Environment

```bash
cd /path/to/rag-eval-harness
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 3. Prepare Data

Add your PDF papers to `data/papers/`:

```bash
# Example: copy papers to the data directory
cp your_papers/*.pdf data/papers/
```

### 4. Set Up Environment Variables

Copy `.env.example` and add your Anthropic API key:

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Build the Index

```bash
python scripts/build_index.py
```

This ingests all PDFs, chunks them (fixed and semantic strategies), embeds with both models (bge-small and e5-large), and stores in Chroma.

### 6. Create Gold Standard Test Set

```bash
python scripts/synthesize_gold.py
```

This generates synthetic Q&A pairs from your papers. Review and edit `data/gold.jsonl` to hand-correct any incorrect answers before evaluation.

### 7. Run Evaluation Matrix

```bash
python eval/run_matrix.py
```

This runs all 4 chunk×embed combinations across the retrieval ladder (dense → hybrid → hybrid+rerank) and generation quality metrics, producing:
- `results/retrieval.md` – NDCG, Recall@5, MRR by retriever
- `results/generation.md` – Faithfulness, Relevance, Context Precision/Recall by config

### 8. Run Tests

```bash
pytest -v
```

### 9. Launch the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Opens an interactive UI to query the retrieval index and see generated answers with citations.

## Deploy

### Hugging Face Spaces (Streamlit)

1. **Create a Space:** Go to [huggingface.co/new-space](https://huggingface.co/new-space), select "Streamlit" SDK, name it `rag-eval-harness`.

2. **Add Secret:** In the Space settings, create a new secret:
   - Name: `ANTHROPIC_API_KEY`
   - Value: `sk-ant-...` (your Claude API key)

3. **Configure Entrypoint:** Edit the Space's README to set the app file:
   ```yaml
   ---
   title: RAG Eval Harness
   emoji: 🔍
   colorFrom: blue
   colorTo: purple
   sdk: streamlit
   sdk_version: "1.36"
   app_file: app/streamlit_app.py
   ---
   ```

4. **Commit Data and Index:** Push your papers and gold set to the Space repo so the index is built on first run:
   ```bash
   # In your Space repo
   git add data/papers/*.pdf data/gold.jsonl
   git commit -m "add papers and gold standard set"
   git push
   ```

5. **Space Builds Index on Startup:** When the Space first loads, `scripts/build_index.py` is called (as configured in the Space's initialization). The index is cached, so subsequent queries are fast.

Alternatively, commit a prebuilt `chroma_db/` directory if the PDF load is large and exceeds Space build-time limits.
