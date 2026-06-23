# RAG Eval Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-paper RAG chatbot with exact page/section citations, and a systematic evaluation harness that compares retrieval strategies and chunk×embedding configs with DeepEval metrics in pytest.

**Architecture:** A thin pipeline of swappable components (`ingest → chunk → embed → store → retrieve → generate`), each a small class behind a narrow interface so experiment cells are substitutable. The eval layer drives every cell over a committed gold set and emits two result tables: a retrieval comparison (Dense → Hybrid+rerank, NDCG/Recall/MRR) and a 2×2 chunk×embed answer-quality matrix (DeepEval faithfulness/relevance/context-precision/recall).

**Tech Stack:** Python 3.11, sentence-transformers (BGE-small, E5-large, cross-encoder reranker), rank-bm25, ChromaDB, pymupdf, Anthropic SDK (Claude), DeepEval, pytest, Streamlit. Deploy to Hugging Face Spaces.

## Global Constraints

- Python 3.11; manage deps in `pyproject.toml` (or `requirements.txt`) — pin major versions.
- LLM provider is Claude only. Judge + generation model: `claude-opus-4-8` (bare ID, no date suffix), exposed as `GEN_MODEL` / `JUDGE_MODEL` constants in `config.py`. `GEN_MODEL` may drop to `claude-sonnet-4-6`.
- Anthropic calls use the official `anthropic` SDK. Use `thinking={"type": "adaptive"}` for synthesis/judging calls; default otherwise. Never pass `temperature`/`top_p`/`budget_tokens` (they 400 on Opus 4.8). Set `max_tokens` explicitly.
- Secrets via `.env` → `ANTHROPIC_API_KEY`. `.env` is git-ignored; never commit it or print the key.
- Embedders and reranker are local (sentence-transformers); no API on the retrieval path.
- One Chroma collection per `(chunk_strategy, embedder)` config; collection name encodes the config.
- `Chunk` provenance fields `{text, paper_id, page, section, chunk_id}` thread end-to-end so citations resolve to an exact page/section.
- Corpus: a fixed, committed set of ~10–20 arXiv ML PDFs under `data/papers/`.
- Gold set: ~40–60 triples in `data/gold.jsonl`, committed.
- DRY, YAGNI, TDD where logic is pure; real-dependency verification for I/O and LLM paths. Commit per task.

---

### Task 1: Project scaffolding & config

**Files:**
- Create: `pyproject.toml`, `config.py`, `rag/__init__.py`, `eval/__init__.py`, `data/papers/.gitkeep`, `.env.example`, `tests/__init__.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `config.GEN_MODEL`, `config.JUDGE_MODEL`, `config.EMBED_MODELS` (dict name→HF id), `config.RERANKER_ID`, `config.CHROMA_DIR`, `config.PAPERS_DIR`, `config.GOLD_PATH`; `config.ExperimentConfig` frozen dataclass `{chunk_strategy: str, embedder: str, retriever: str}` with `.collection_name -> str`; `config.MATRIX: list[ExperimentConfig]` (4 cells); `config.RETRIEVER_LADDER: list[str]` = `["dense", "hybrid", "hybrid_rerank"]`; `config.anthropic_client() -> anthropic.Anthropic`.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "rag-eval-harness"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "anthropic>=0.40",
  "pymupdf>=1.24",
  "sentence-transformers>=3.0",
  "rank-bm25>=0.2.2",
  "chromadb>=0.5",
  "deepeval>=1.0",
  "streamlit>=1.36",
  "python-dotenv>=1.0",
  "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.pytest.ini_options]
testpaths = ["tests", "eval"]
```

- [ ] **Step 2: Write `config.py`**

```python
import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "data" / "papers"
GOLD_PATH = ROOT / "data" / "gold.jsonl"
CHROMA_DIR = ROOT / "chroma_db"

GEN_MODEL = "claude-opus-4-8"
JUDGE_MODEL = "claude-opus-4-8"

EMBED_MODELS = {
    "bge_small": "BAAI/bge-small-en-v1.5",
    "e5_large": "intfloat/e5-large-v2",
}
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_STRATEGIES = ["fixed", "semantic"]
RETRIEVER_LADDER = ["dense", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class ExperimentConfig:
    chunk_strategy: str
    embedder: str
    retriever: str = "hybrid_rerank"

    @property
    def collection_name(self) -> str:
        return f"{self.chunk_strategy}__{self.embedder}"


MATRIX = [
    ExperimentConfig(c, e)
    for c in CHUNK_STRATEGIES
    for e in EMBED_MODELS
]


def anthropic_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")
    return anthropic.Anthropic(api_key=key)
```

- [ ] **Step 3: Write `.env.example`**

```
ANTHROPIC_API_KEY=sk-ant-...
```

- [ ] **Step 4: Write `tests/test_config.py`**

```python
from config import MATRIX, ExperimentConfig, EMBED_MODELS


def test_matrix_is_2x2():
    assert len(MATRIX) == 4
    names = {c.collection_name for c in MATRIX}
    assert names == {
        "fixed__bge_small", "fixed__e5_large",
        "semantic__bge_small", "semantic__e5_large",
    }


def test_embed_models_are_the_chosen_pair():
    assert EMBED_MODELS["bge_small"] == "BAAI/bge-small-en-v1.5"
    assert EMBED_MODELS["e5_large"] == "intfloat/e5-large-v2"
```

- [ ] **Step 5: Run tests, expect PASS**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml config.py .env.example rag/__init__.py eval/__init__.py tests/ data/papers/.gitkeep
git commit -m "feat: project scaffolding and experiment config"
```

---

### Task 2: PDF ingest (page-aware blocks)

**Files:**
- Create: `rag/ingest.py`, `tests/test_ingest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ingest.Block` dataclass `{paper_id: str, page: int, section: str, text: str}`; `ingest.ingest_pdf(path: Path) -> list[Block]`; `ingest.ingest_dir(papers_dir: Path) -> list[Block]`. `page` is 1-indexed. `section` is a best-effort heading string ("" if unknown). `paper_id` is the PDF filename stem.

- [ ] **Step 1: Write `rag/ingest.py`**

```python
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Block:
    paper_id: str
    page: int
    section: str
    text: str


_HEADING = re.compile(r"^(\d+(\.\d+)*)\s+[A-Z].{0,80}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    return bool(_HEADING.match(line)) or (line.isupper() and len(line.split()) <= 8)


def ingest_pdf(path: Path) -> list[Block]:
    paper_id = path.stem
    doc = fitz.open(path)
    blocks: list[Block] = []
    section = ""
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        lines = [ln for ln in text.splitlines()]
        buffer: list[str] = []

        def flush():
            joined = " ".join(buffer).strip()
            if joined:
                blocks.append(Block(paper_id, page_index + 1, section, joined))
            buffer.clear()

        for line in lines:
            if _looks_like_heading(line):
                flush()
                section = line.strip()
            else:
                buffer.append(line)
        flush()
    doc.close()
    return blocks


def ingest_dir(papers_dir: Path) -> list[Block]:
    out: list[Block] = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        out.extend(ingest_pdf(pdf))
    return out
```

- [ ] **Step 2: Add a small real fixture PDF**

Drop at least one real arXiv PDF into `data/papers/` for tests (the corpus you'll use). Verify the dir is non-empty:

Run: `ls data/papers/*.pdf | head`
Expected: at least one path printed.

- [ ] **Step 3: Write `tests/test_ingest.py` (real dependency — the PDF)**

```python
from pathlib import Path

from config import PAPERS_DIR
from rag.ingest import ingest_dir


def test_ingest_produces_provenanced_blocks():
    blocks = ingest_dir(PAPERS_DIR)
    assert len(blocks) > 0
    b = blocks[0]
    assert b.paper_id and isinstance(b.paper_id, str)
    assert b.page >= 1
    assert isinstance(b.section, str)
    assert b.text.strip()
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_ingest.py -v`
Expected: 1 passed (requires a PDF in `data/papers/`).

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingest.py data/papers/
git commit -m "feat: page- and section-aware PDF ingest"
```

---

### Task 3: Chunking strategies

**Files:**
- Create: `rag/chunk.py`, `tests/test_chunk.py`

**Interfaces:**
- Consumes: `ingest.Block`.
- Produces: `chunk.Chunk` dataclass `{chunk_id: str, paper_id: str, page: int, section: str, text: str}`; `chunk.chunk_blocks(blocks: list[Block], strategy: str) -> list[Chunk]` where `strategy in {"fixed","semantic"}`. `chunk_id` is stable `f"{paper_id}:{page}:{ordinal}"`. Fixed = ~500-char windows with ~80-char overlap, never crossing pages. Semantic = greedy sentence packing up to ~700 chars, breaking on section change.

- [ ] **Step 1: Write the failing test `tests/test_chunk.py`**

```python
from rag.ingest import Block
from rag.chunk import chunk_blocks


def _blocks():
    return [
        Block("p1", 1, "1 Intro", "Sentence one. " * 60),
        Block("p1", 2, "2 Method", "Method detail. " * 60),
    ]


def test_fixed_chunks_have_provenance_and_ids():
    chunks = chunk_blocks(_blocks(), "fixed")
    assert len(chunks) > 2
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))            # unique
    assert all(c.page in (1, 2) for c in chunks)
    assert all(c.paper_id == "p1" for c in chunks)


def test_fixed_chunks_never_cross_pages():
    chunks = chunk_blocks(_blocks(), "fixed")
    for c in chunks:
        assert c.section in ("1 Intro", "2 Method")


def test_semantic_breaks_on_section():
    chunks = chunk_blocks(_blocks(), "semantic")
    sections = {c.section for c in chunks}
    assert sections == {"1 Intro", "2 Method"}
```

- [ ] **Step 2: Run, expect FAIL (module missing)**

Run: `pytest tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: rag.chunk`.

- [ ] **Step 3: Write `rag/chunk.py`**

```python
import re
from dataclasses import dataclass

from rag.ingest import Block

FIXED_SIZE = 500
FIXED_OVERLAP = 80
SEMANTIC_MAX = 700

_SENT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    paper_id: str
    page: int
    section: str
    text: str


def _fixed(blocks: list[Block]) -> list[Chunk]:
    out: list[Chunk] = []
    for b in blocks:
        start, ordinal = 0, 0
        text = b.text
        while start < len(text):
            piece = text[start:start + FIXED_SIZE].strip()
            if piece:
                out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                                 b.paper_id, b.page, b.section, piece))
                ordinal += 1
            start += FIXED_SIZE - FIXED_OVERLAP
    return out


def _semantic(blocks: list[Block]) -> list[Chunk]:
    out: list[Chunk] = []
    for b in blocks:
        sentences = _SENT.split(b.text)
        buf, ordinal = "", 0
        for s in sentences:
            if buf and len(buf) + len(s) > SEMANTIC_MAX:
                out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                                 b.paper_id, b.page, b.section, buf.strip()))
                ordinal += 1
                buf = ""
            buf += (" " if buf else "") + s
        if buf.strip():
            out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                             b.paper_id, b.page, b.section, buf.strip()))
    return out


def chunk_blocks(blocks: list[Block], strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return _fixed(blocks)
    if strategy == "semantic":
        return _semantic(blocks)
    raise ValueError(f"unknown strategy {strategy!r}")
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_chunk.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/chunk.py tests/test_chunk.py
git commit -m "feat: fixed-size and semantic chunking with provenance"
```

---

### Task 4: Embedders

**Files:**
- Create: `rag/embed.py`, `tests/test_embed.py`

**Interfaces:**
- Consumes: `config.EMBED_MODELS`.
- Produces: `embed.Embedder` class; `Embedder(name: str)` (name is a key of `EMBED_MODELS`); `.embed_documents(texts: list[str]) -> list[list[float]]`; `.embed_query(text: str) -> list[float]`; `.dim -> int`. E5 requires `"query: "` / `"passage: "` prefixes — applied internally based on `name`.

- [ ] **Step 1: Write `rag/embed.py`**

```python
from sentence_transformers import SentenceTransformer

from config import EMBED_MODELS


class Embedder:
    def __init__(self, name: str):
        if name not in EMBED_MODELS:
            raise ValueError(f"unknown embedder {name!r}")
        self.name = name
        self.model = SentenceTransformer(EMBED_MODELS[name])
        self._e5 = name.startswith("e5")

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def _prep(self, texts: list[str], kind: str) -> list[str]:
        if self._e5:
            return [f"{kind}: {t}" for t in texts]
        return texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepped = self._prep(texts, "passage")
        vecs = self.model.encode(prepped, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        prepped = self._prep([text], "query")
        return self.model.encode(prepped, normalize_embeddings=True)[0].tolist()
```

- [ ] **Step 2: Write `tests/test_embed.py` (real model download on first run)**

```python
from rag.embed import Embedder


def test_bge_embeds_with_expected_dim():
    e = Embedder("bge_small")
    v = e.embed_query("retrieval augmented generation")
    assert len(v) == e.dim == 384


def test_doc_and_query_embeddings_same_dim():
    e = Embedder("bge_small")
    d = e.embed_documents(["a passage about transformers"])
    assert len(d) == 1 and len(d[0]) == e.dim
```

- [ ] **Step 3: Run, expect PASS (downloads model)**

Run: `pytest tests/test_embed.py -v`
Expected: 2 passed.

- [ ] **Step 4: Commit**

```bash
git add rag/embed.py tests/test_embed.py
git commit -m "feat: BGE and E5 embedders with E5 prefixing"
```

---

### Task 5: Chroma store

**Files:**
- Create: `rag/store.py`, `tests/test_store.py`

**Interfaces:**
- Consumes: `chunk.Chunk`, `embed.Embedder`, `config.CHROMA_DIR`.
- Produces: `store.ChunkStore` class; `ChunkStore(collection_name: str)`; `.index(chunks: list[Chunk], embedder: Embedder) -> None` (idempotent rebuild — drops & recreates the collection); `.query(vector: list[float], k: int) -> list[tuple[Chunk, float]]` (score = cosine similarity, higher better); `.all_chunk_ids() -> list[str]`. Chunk metadata stores `paper_id/page/section/text`.

- [ ] **Step 1: Write `rag/store.py`**

```python
import chromadb

from config import CHROMA_DIR
from rag.chunk import Chunk
from rag.embed import Embedder


class ChunkStore:
    def __init__(self, collection_name: str):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.name = collection_name

    def index(self, chunks: list[Chunk], embedder: Embedder) -> None:
        try:
            self.client.delete_collection(self.name)
        except Exception:
            pass
        col = self.client.create_collection(
            self.name, metadata={"hnsw:space": "cosine"}
        )
        vectors = embedder.embed_documents([c.text for c in chunks])
        col.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            metadatas=[
                {"paper_id": c.paper_id, "page": c.page,
                 "section": c.section, "text": c.text}
                for c in chunks
            ],
        )

    def _collection(self):
        return self.client.get_collection(self.name)

    def query(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        res = self._collection().query(
            query_embeddings=[vector], n_results=k,
            include=["metadatas", "distances"],
        )
        out: list[tuple[Chunk, float]] = []
        for cid, meta, dist in zip(
            res["ids"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(cid, meta["paper_id"], int(meta["page"]),
                          meta["section"], meta["text"])
            out.append((chunk, 1.0 - dist))  # cosine distance -> similarity
        return out

    def all_chunk_ids(self) -> list[str]:
        return self._collection().get()["ids"]
```

- [ ] **Step 2: Write `tests/test_store.py`**

```python
from rag.chunk import Chunk
from rag.embed import Embedder
from rag.store import ChunkStore


def test_index_and_query_roundtrip():
    chunks = [
        Chunk("d:1:0", "d", 1, "Intro", "Transformers use self-attention."),
        Chunk("d:1:1", "d", 1, "Intro", "Bananas are yellow fruit."),
    ]
    store = ChunkStore("test_roundtrip")
    emb = Embedder("bge_small")
    store.index(chunks, emb)
    hits = store.query(emb.embed_query("attention mechanism"), k=2)
    assert hits[0][0].chunk_id == "d:1:0"      # relevant chunk ranked first
    assert hits[0][1] > hits[1][1]             # similarity ordered
    assert set(store.all_chunk_ids()) == {"d:1:0", "d:1:1"}
```

- [ ] **Step 3: Run, expect PASS**

Run: `pytest tests/test_store.py -v`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: Chroma per-config chunk store with cosine similarity"
```

---

### Task 6: Retrievers (dense, BM25, hybrid, rerank)

**Files:**
- Create: `rag/retrieve.py`, `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `store.ChunkStore`, `embed.Embedder`, `chunk.Chunk`, `config.RERANKER_ID`.
- Produces: `retrieve.Retriever` class; `Retriever(store, embedder, all_chunks: list[Chunk], mode: str)` where `mode in RETRIEVER_LADDER`; `.retrieve(query: str, k: int) -> list[Chunk]`. Hybrid = reciprocal-rank fusion of dense + BM25 (over `all_chunks` corpus); `hybrid_rerank` reranks the fused top-`3k` with the cross-encoder, returns top-`k`. Reranker loaded lazily and only for `hybrid_rerank`.

- [ ] **Step 1: Write the failing test `tests/test_retrieve.py`**

```python
from rag.chunk import Chunk
from rag.embed import Embedder
from rag.store import ChunkStore
from rag.retrieve import Retriever

CORPUS = [
    Chunk("d:1:0", "d", 1, "S", "Self-attention lets transformers weigh tokens."),
    Chunk("d:1:1", "d", 1, "S", "BM25 is a lexical ranking function."),
    Chunk("d:2:0", "d", 2, "S", "Cross-encoders rerank candidate passages."),
]


def _retriever(mode):
    store = ChunkStore("test_retrieve")
    emb = Embedder("bge_small")
    store.index(CORPUS, emb)
    return Retriever(store, emb, CORPUS, mode)


def test_dense_finds_semantic_match():
    hits = _retriever("dense").retrieve("attention over tokens", k=1)
    assert hits[0].chunk_id == "d:1:0"


def test_bm25_finds_exact_keyword():
    hits = _retriever("hybrid").retrieve("BM25 lexical ranking", k=1)
    assert hits[0].chunk_id == "d:1:1"


def test_rerank_returns_k():
    hits = _retriever("hybrid_rerank").retrieve("how to rerank passages", k=2)
    assert len(hits) == 2
    assert "d:2:0" in {h.chunk_id for h in hits}
```

- [ ] **Step 2: Run, expect FAIL (module missing)**

Run: `pytest tests/test_retrieve.py -v`
Expected: FAIL — `ModuleNotFoundError: rag.retrieve`.

- [ ] **Step 3: Write `rag/retrieve.py`**

```python
import re

from rank_bm25 import BM25Okapi

from config import RERANKER_ID
from rag.chunk import Chunk
from rag.embed import Embedder
from rag.store import ChunkStore

_RRF_K = 60
_TOKEN = re.compile(r"\w+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, store: ChunkStore, embedder: Embedder,
                 all_chunks: list[Chunk], mode: str):
        self.store = store
        self.embedder = embedder
        self.chunks = all_chunks
        self.by_id = {c.chunk_id: c for c in all_chunks}
        self.mode = mode
        self._bm25 = BM25Okapi([_tok(c.text) for c in all_chunks])
        self._reranker = None

    def _dense(self, query: str, k: int) -> list[str]:
        hits = self.store.query(self.embedder.embed_query(query), k)
        return [c.chunk_id for c, _ in hits]

    def _bm25_ids(self, query: str, k: int) -> list[str]:
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].chunk_id for i in ranked[:k]]

    def _rrf(self, query: str, k: int) -> list[str]:
        dense = self._dense(query, k * 3)
        lexical = self._bm25_ids(query, k * 3)
        fused: dict[str, float] = {}
        for ranking in (dense, lexical):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        return sorted(fused, key=lambda c: fused[c], reverse=True)

    def _rerank(self, query: str, candidate_ids: list[str], k: int) -> list[str]:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(RERANKER_ID)
        pairs = [(query, self.by_id[c].text) for c in candidate_ids]
        scores = self._reranker.predict(pairs)
        order = sorted(range(len(candidate_ids)),
                       key=lambda i: scores[i], reverse=True)
        return [candidate_ids[i] for i in order[:k]]

    def retrieve(self, query: str, k: int) -> list[Chunk]:
        if self.mode == "dense":
            ids = self._dense(query, k)
        elif self.mode == "hybrid":
            ids = self._rrf(query, k)[:k]
        elif self.mode == "hybrid_rerank":
            fused = self._rrf(query, k * 3)[:k * 3]
            ids = self._rerank(query, fused, k)
        else:
            raise ValueError(f"unknown mode {self.mode!r}")
        return [self.by_id[c] for c in ids]
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_retrieve.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/retrieve.py tests/test_retrieve.py
git commit -m "feat: dense/BM25/hybrid retrieval with cross-encoder rerank"
```

---

### Task 7: Answer generation with citations

**Files:**
- Create: `rag/generate.py`, `tests/test_generate.py`

**Interfaces:**
- Consumes: `chunk.Chunk`, `config.anthropic_client`, `config.GEN_MODEL`.
- Produces: `generate.format_context(chunks: list[Chunk]) -> str` (numbers sources `[1]`, `[2]`… with `paper_id p.PAGE §section`); `generate.answer(query: str, chunks: list[Chunk], model: str = GEN_MODEL) -> str`. The system prompt requires grounding strictly in context and citing sources inline as `[n]`; if context is insufficient, the model must say so. `format_context` is pure and unit-tested; `answer` is verified against the real API.

- [ ] **Step 1: Write `rag/generate.py`**

```python
from config import GEN_MODEL, anthropic_client
from rag.chunk import Chunk

_SYSTEM = (
    "You answer questions strictly from the provided sources about research "
    "papers. Cite every claim inline with bracketed source numbers like [1]. "
    "Each source is labeled with its paper, page, and section. If the sources "
    "do not contain the answer, say so plainly — do not use outside knowledge."
)


def format_context(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        sec = f" §{c.section}" if c.section else ""
        lines.append(f"[{i}] ({c.paper_id} p.{c.page}{sec})\n{c.text}")
    return "\n\n".join(lines)


def answer(query: str, chunks: list[Chunk], model: str = GEN_MODEL) -> str:
    client = anthropic_client()
    context = format_context(chunks)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Sources:\n\n{context}\n\nQuestion: {query}",
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
```

- [ ] **Step 2: Write `tests/test_generate.py`**

```python
from rag.chunk import Chunk
from rag.generate import format_context, answer


def test_format_context_numbers_and_provenances():
    chunks = [Chunk("d:3:0", "attention", 3, "2 Method", "Self-attention scales.")]
    ctx = format_context(chunks)
    assert "[1]" in ctx and "attention p.3" in ctx and "§2 Method" in ctx


def test_answer_grounds_and_cites():  # real Anthropic API
    chunks = [Chunk("d:3:0", "attn", 3, "Method",
                    "The model uses 8 attention heads.")]
    out = answer("How many attention heads does the model use?", chunks)
    assert "8" in out
    assert "[1]" in out


def test_answer_refuses_when_unsupported():  # real Anthropic API
    chunks = [Chunk("d:3:0", "attn", 3, "Method", "The model uses 8 heads.")]
    out = answer("What is the capital of France?", chunks).lower()
    assert "source" in out or "not" in out
```

- [ ] **Step 3: Run, expect PASS (requires `ANTHROPIC_API_KEY`)**

Run: `pytest tests/test_generate.py -v`
Expected: 3 passed.

- [ ] **Step 4: Commit**

```bash
git add rag/generate.py tests/test_generate.py
git commit -m "feat: grounded Claude answer generation with inline citations"
```

---

### Task 8: Pipeline wiring + end-to-end smoke

**Files:**
- Create: `rag/pipeline.py`, `scripts/build_index.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all of `rag/*`, `config.ExperimentConfig`, `config.PAPERS_DIR`.
- Produces: `pipeline.RAGPipeline` class; `RAGPipeline(cfg: ExperimentConfig)`; `.build() -> None` (ingest→chunk→embed→index, stores `self.chunks`); `.load() -> None` (rebuild in-memory retriever from an already-indexed collection); `.retrieve(query, k=5) -> list[Chunk]`; `.ask(query, k=5) -> tuple[str, list[Chunk]]` (answer + sources). `scripts/build_index.py` builds all four matrix collections from `data/papers/`.

- [ ] **Step 1: Write `rag/pipeline.py`**

```python
from config import ExperimentConfig, PAPERS_DIR
from rag.ingest import ingest_dir
from rag.chunk import chunk_blocks, Chunk
from rag.embed import Embedder
from rag.store import ChunkStore
from rag.retrieve import Retriever
from rag.generate import answer


class RAGPipeline:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedder)
        self.store = ChunkStore(cfg.collection_name)
        self.chunks: list[Chunk] = []
        self.retriever: Retriever | None = None

    def build(self) -> None:
        blocks = ingest_dir(PAPERS_DIR)
        self.chunks = chunk_blocks(blocks, self.cfg.chunk_strategy)
        self.store.index(self.chunks, self.embedder)
        self._make_retriever()

    def load(self) -> None:
        blocks = ingest_dir(PAPERS_DIR)
        self.chunks = chunk_blocks(blocks, self.cfg.chunk_strategy)
        self._make_retriever()

    def _make_retriever(self) -> None:
        self.retriever = Retriever(
            self.store, self.embedder, self.chunks, self.cfg.retriever
        )

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        assert self.retriever is not None, "call build() or load() first"
        return self.retriever.retrieve(query, k)

    def ask(self, query: str, k: int = 5) -> tuple[str, list[Chunk]]:
        sources = self.retrieve(query, k)
        return answer(query, sources), sources
```

- [ ] **Step 2: Write `scripts/build_index.py`**

```python
from config import MATRIX
from rag.pipeline import RAGPipeline

if __name__ == "__main__":
    for cfg in MATRIX:
        print(f"building {cfg.collection_name} ...")
        RAGPipeline(cfg).build()
    print("done")
```

- [ ] **Step 3: Write `tests/test_pipeline.py` (end-to-end, real models + API)**

```python
from config import ExperimentConfig
from rag.pipeline import RAGPipeline


def test_end_to_end_one_question():
    cfg = ExperimentConfig("fixed", "bge_small", "hybrid_rerank")
    pipe = RAGPipeline(cfg)
    pipe.build()
    out, sources = pipe.ask("What problem does this paper address?", k=5)
    assert out.strip()
    assert 1 <= len(sources) <= 5
    assert all(s.page >= 1 for s in sources)
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_pipeline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add rag/pipeline.py scripts/build_index.py tests/test_pipeline.py
git commit -m "feat: end-to-end RAG pipeline and index builder"
```

---

### Task 9: Retrieval metrics (NDCG / Recall / MRR)

**Files:**
- Create: `eval/retrieval_metrics.py`, `tests/test_retrieval_metrics.py`

**Interfaces:**
- Consumes: nothing (pure functions over id lists).
- Produces: `retrieval_metrics.ndcg_at_k(retrieved_ids, relevant_ids, k) -> float`; `recall_at_k(retrieved_ids, relevant_ids, k) -> float`; `mrr(retrieved_ids, relevant_ids) -> float`. Binary relevance; ideal DCG uses min(|relevant|, k).

- [ ] **Step 1: Write the failing test `tests/test_retrieval_metrics.py`**

```python
import math
from eval.retrieval_metrics import ndcg_at_k, recall_at_k, mrr


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5


def test_mrr_first_relevant_at_rank_2():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_relevant_at_rank_2():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))
```

- [ ] **Step 2: Run, expect FAIL**

Run: `pytest tests/test_retrieval_metrics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `eval/retrieval_metrics.py`**

```python
import math


def recall_at_k(retrieved_ids, relevant_ids, k) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit = sum(1 for cid in retrieved_ids[:k] if cid in relevant)
    return hit / len(relevant)


def mrr(retrieved_ids, relevant_ids) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k) -> float:
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_retrieval_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/retrieval_metrics.py tests/test_retrieval_metrics.py
git commit -m "feat: NDCG@k, Recall@k, MRR retrieval metrics"
```

---

### Task 10: Gold set synthesis & loader

**Files:**
- Create: `eval/gold.py`, `scripts/synthesize_gold.py`, `tests/test_gold.py`

**Interfaces:**
- Consumes: `chunk.Chunk`, `config.anthropic_client`, `config.JUDGE_MODEL`, `config.GOLD_PATH`.
- Produces: `gold.GoldItem` dataclass `{question: str, reference_answer: str, ground_truth_chunk_ids: list[str], source_pages: list[int]}`; `gold.synthesize(chunks: list[Chunk], n: int) -> list[GoldItem]` (samples chunks, asks Claude with `thinking adaptive` + structured output for a question answerable from that chunk, records the chunk id/page as ground truth); `gold.save(items, path)` / `gold.load(path) -> list[GoldItem]` (JSONL). `scripts/synthesize_gold.py` builds ~50 items from the `fixed__bge_small` chunking and writes `data/gold.jsonl` for hand-correction.

- [ ] **Step 1: Write `eval/gold.py`**

```python
import json
import random
from dataclasses import dataclass, asdict

from config import JUDGE_MODEL, anthropic_client
from rag.chunk import Chunk

_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
    },
    "required": ["question", "reference_answer"],
    "additionalProperties": False,
}

_SYS = (
    "Given a passage from a research paper, write ONE specific question that is "
    "fully answerable from this passage alone, and its concise reference answer. "
    "Avoid questions needing outside context."
)


@dataclass
class GoldItem:
    question: str
    reference_answer: str
    ground_truth_chunk_ids: list[str]
    source_pages: list[int]


def _one(client, chunk: Chunk) -> GoldItem:
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=_SYS,
        messages=[{"role": "user", "content": chunk.text}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = json.loads(text)
    return GoldItem(data["question"], data["reference_answer"],
                    [chunk.chunk_id], [chunk.page])


def synthesize(chunks: list[Chunk], n: int) -> list[GoldItem]:
    client = anthropic_client()
    pool = [c for c in chunks if len(c.text) > 200]
    random.seed(0)
    sample = random.sample(pool, min(n, len(pool)))
    return [_one(client, c) for c in sample]


def save(items: list[GoldItem], path) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def load(path) -> list[GoldItem]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(GoldItem(**json.loads(line)))
    return items
```

- [ ] **Step 2: Write `scripts/synthesize_gold.py`**

```python
from config import ExperimentConfig, GOLD_PATH, PAPERS_DIR
from rag.ingest import ingest_dir
from rag.chunk import chunk_blocks
from eval.gold import synthesize, save

if __name__ == "__main__":
    chunks = chunk_blocks(ingest_dir(PAPERS_DIR), "fixed")
    items = synthesize(chunks, n=50)
    save(items, GOLD_PATH)
    print(f"wrote {len(items)} gold items to {GOLD_PATH}")
    print("NOW: hand-correct data/gold.jsonl before committing.")
```

- [ ] **Step 3: Write `tests/test_gold.py` (save/load is pure; synth needs API)**

```python
from eval.gold import GoldItem, save, load


def test_save_load_roundtrip(tmp_path):
    items = [GoldItem("Q?", "A.", ["d:1:0"], [1])]
    p = tmp_path / "g.jsonl"
    save(items, p)
    back = load(p)
    assert back == items
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_gold.py -v`
Expected: 1 passed.

- [ ] **Step 5: Generate and hand-correct the gold set**

Run: `python scripts/synthesize_gold.py`
Then open `data/gold.jsonl`, fix wrong/ambiguous questions, delete bad rows (target ~40–60 good ones).

- [ ] **Step 6: Commit**

```bash
git add eval/gold.py scripts/synthesize_gold.py tests/test_gold.py data/gold.jsonl
git commit -m "feat: gold set synthesis, JSONL loader, and committed gold set"
```

---

### Task 11: DeepEval metrics wiring (Claude judge)

**Files:**
- Create: `eval/metrics.py`, `eval/claude_judge.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `config.JUDGE_MODEL`, `gold.GoldItem`, `chunk.Chunk`, `generate.answer`.
- Produces: `claude_judge.ClaudeJudge` — a `deepeval.models.DeepEvalBaseLLM` subclass wrapping Claude as the metric judge; `metrics.evaluate_case(question, generated_answer, reference_answer, retrieved_chunks) -> dict[str,float]` returning `{faithfulness, answer_relevance, context_precision, context_recall}` via DeepEval's `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextualPrecisionMetric`, `ContextualRecallMetric`, each constructed with `ClaudeJudge`.

- [ ] **Step 1: Write `eval/claude_judge.py`**

```python
from deepeval.models import DeepEvalBaseLLM

from config import JUDGE_MODEL, anthropic_client


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic_client()

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        msg = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return JUDGE_MODEL
```

> Note: confirm the installed DeepEval's `DeepEvalBaseLLM` signature for `generate` (some versions pass a `schema`). If it requires schema-constrained JSON, return `client.messages.create(..., output_config={"format": {"type":"json_schema","schema": schema.model_json_schema()}})` parsed into the schema. Adjust the two methods to match the installed version — verify with `python -c "import deepeval, inspect; from deepeval.models import DeepEvalBaseLLM; print(inspect.getsource(DeepEvalBaseLLM))"`.

- [ ] **Step 2: Write `eval/metrics.py`**

```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric,
)

from eval.claude_judge import ClaudeJudge

_judge = ClaudeJudge()


def evaluate_case(question, generated_answer, reference_answer,
                  retrieved_chunks) -> dict:
    context = [c.text for c in retrieved_chunks]
    case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=reference_answer,
        retrieval_context=context,
    )
    metrics = {
        "faithfulness": FaithfulnessMetric(model=_judge),
        "answer_relevance": AnswerRelevancyMetric(model=_judge),
        "context_precision": ContextualPrecisionMetric(model=_judge),
        "context_recall": ContextualRecallMetric(model=_judge),
    }
    scores = {}
    for name, metric in metrics.items():
        metric.measure(case)
        scores[name] = metric.score
    return scores
```

- [ ] **Step 3: Write `tests/test_metrics.py` (real judge — one case)**

```python
from rag.chunk import Chunk
from eval.metrics import evaluate_case


def test_faithful_answer_scores_high():
    chunks = [Chunk("d:1:0", "d", 1, "M", "The model has 12 layers.")]
    scores = evaluate_case(
        "How many layers does the model have?",
        "The model has 12 layers [1].",
        "12 layers.",
        chunks,
    )
    assert set(scores) == {"faithfulness", "answer_relevance",
                           "context_precision", "context_recall"}
    assert scores["faithfulness"] >= 0.5
```

- [ ] **Step 4: Run, expect PASS**

Run: `pytest tests/test_metrics.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add eval/claude_judge.py eval/metrics.py tests/test_metrics.py
git commit -m "feat: Claude-judge DeepEval metric wiring"
```

---

### Task 12: pytest quality gate

**Files:**
- Create: `eval/test_rag.py`

**Interfaces:**
- Consumes: `config.MATRIX`, `gold.load`, `pipeline.RAGPipeline`, `metrics.evaluate_case`, `retrieval_metrics.*`.
- Produces: a parametrized pytest module asserting, for the `fixed__bge_small` config over a small gold subset, that `faithfulness >= 0.7` and mean `ndcg_at_k >= baseline`. This is the CI-able gate (small subset for speed); `run_matrix.py` (Task 13) does the full sweep.

- [ ] **Step 1: Write `eval/test_rag.py`**

```python
import statistics

import pytest

from config import ExperimentConfig
from eval.gold import load
from eval.retrieval_metrics import ndcg_at_k
from eval.metrics import evaluate_case
from config import GOLD_PATH
from rag.pipeline import RAGPipeline

GOLD = load(GOLD_PATH)[:8]  # small subset for the gate


@pytest.fixture(scope="module")
def pipe():
    p = RAGPipeline(ExperimentConfig("fixed", "bge_small", "hybrid_rerank"))
    p.build()
    return p


def test_retrieval_ndcg_reasonable(pipe):
    scores = []
    for item in GOLD:
        ids = [c.chunk_id for c in pipe.retrieve(item.question, k=5)]
        scores.append(ndcg_at_k(ids, item.ground_truth_chunk_ids, 5))
    assert statistics.mean(scores) >= 0.3


def test_faithfulness_gate(pipe):
    item = GOLD[0]
    out, sources = pipe.ask(item.question, k=5)
    scores = evaluate_case(item.question, out, item.reference_answer, sources)
    assert scores["faithfulness"] >= 0.7
```

- [ ] **Step 2: Run, expect PASS**

Run: `pytest eval/test_rag.py -v`
Expected: 2 passed (requires built index, gold set, API key).

- [ ] **Step 3: Commit**

```bash
git add eval/test_rag.py
git commit -m "test: pytest quality gate for retrieval and faithfulness"
```

---

### Task 13: Experiment matrix runner & tables

**Files:**
- Create: `eval/run_matrix.py`, `tests/test_run_matrix.py`

**Interfaces:**
- Consumes: `config.MATRIX`, `config.RETRIEVER_LADDER`, `gold.load`, `pipeline.RAGPipeline`, `retrieval_metrics.*`, `metrics.evaluate_case`.
- Produces: `run_matrix.retrieval_table() -> list[dict]` (one row per `RETRIEVER_LADDER` mode at `fixed__bge_small`: mean NDCG@5/Recall@5/MRR); `run_matrix.generation_table() -> list[dict]` (one row per `MATRIX` cell at `hybrid_rerank`: mean of the four DeepEval metrics); `run_matrix.to_markdown(rows, headers) -> str`; `__main__` writes `results/retrieval.md`, `results/generation.md`, and matching `.csv`. Helper `to_markdown` is unit-tested; the table builders are integration-run by hand.

- [ ] **Step 1: Write `eval/run_matrix.py`**

```python
import csv
import statistics
from pathlib import Path

from config import MATRIX, RETRIEVER_LADDER, GOLD_PATH, ExperimentConfig
from eval.gold import load
from eval.retrieval_metrics import ndcg_at_k, recall_at_k, mrr
from eval.metrics import evaluate_case
from rag.pipeline import RAGPipeline

RESULTS = Path(__file__).parent.parent / "results"


def _mean(xs):
    return round(statistics.mean(xs), 3) if xs else 0.0


def retrieval_table() -> list[dict]:
    gold = load(GOLD_PATH)
    rows = []
    for mode in RETRIEVER_LADDER:
        pipe = RAGPipeline(ExperimentConfig("fixed", "bge_small", mode))
        pipe.build()
        nd, rc, rr = [], [], []
        for item in gold:
            ids = [c.chunk_id for c in pipe.retrieve(item.question, k=5)]
            gt = item.ground_truth_chunk_ids
            nd.append(ndcg_at_k(ids, gt, 5))
            rc.append(recall_at_k(ids, gt, 5))
            rr.append(mrr(ids, gt))
        rows.append({"retriever": mode, "ndcg@5": _mean(nd),
                     "recall@5": _mean(rc), "mrr": _mean(rr)})
    return rows


def generation_table() -> list[dict]:
    gold = load(GOLD_PATH)
    rows = []
    for cfg in MATRIX:
        pipe = RAGPipeline(cfg)
        pipe.build()
        agg = {"faithfulness": [], "answer_relevance": [],
               "context_precision": [], "context_recall": []}
        for item in gold:
            out, sources = pipe.ask(item.question, k=5)
            s = evaluate_case(item.question, out, item.reference_answer, sources)
            for k in agg:
                agg[k].append(s[k])
        row = {"config": cfg.collection_name}
        row.update({k: _mean(v) for k, v in agg.items()})
        rows.append(row)
    return rows


def to_markdown(rows: list[dict], headers: list[str]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write(name, rows, headers):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(to_markdown(rows, headers))
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rt = retrieval_table()
    _write("retrieval", rt, ["retriever", "ndcg@5", "recall@5", "mrr"])
    gt = generation_table()
    _write("generation", gt, ["config", "faithfulness", "answer_relevance",
                              "context_precision", "context_recall"])
    print("wrote results/retrieval.{md,csv} and results/generation.{md,csv}")
```

- [ ] **Step 2: Write `tests/test_run_matrix.py` (pure helper)**

```python
from eval.run_matrix import to_markdown


def test_to_markdown_shapes_table():
    rows = [{"a": 1, "b": 2}]
    md = to_markdown(rows, ["a", "b"])
    assert md.splitlines()[0] == "| a | b |"
    assert md.splitlines()[2] == "| 1 | 2 |"
```

- [ ] **Step 3: Run, expect PASS**

Run: `pytest tests/test_run_matrix.py -v`
Expected: 1 passed.

- [ ] **Step 4: Produce the real tables**

Run: `python eval/run_matrix.py`
Expected: `results/retrieval.{md,csv}` and `results/generation.{md,csv}` written. Inspect them.

- [ ] **Step 5: Commit**

```bash
git add eval/run_matrix.py tests/test_run_matrix.py results/
git commit -m "feat: experiment matrix runner and results tables"
```

---

### Task 14: Streamlit demo

**Files:**
- Create: `app/streamlit_app.py`

**Interfaces:**
- Consumes: `config.MATRIX`, `pipeline.RAGPipeline`.
- Produces: a Streamlit app with a config selector (the 4 matrix cells + retriever mode), a question box, the answer, and an expandable "Sources" panel showing each cited chunk's `paper_id p.page §section` + text. Pipelines cached with `st.cache_resource`. Assumes `scripts/build_index.py` has been run.

- [ ] **Step 1: Write `app/streamlit_app.py`**

```python
import streamlit as st

from config import CHUNK_STRATEGIES, EMBED_MODELS, RETRIEVER_LADDER, ExperimentConfig
from rag.pipeline import RAGPipeline


@st.cache_resource
def get_pipeline(chunk, embed, retr):
    pipe = RAGPipeline(ExperimentConfig(chunk, embed, retr))
    pipe.load()
    return pipe


st.title("Chat with the papers")

col1, col2, col3 = st.columns(3)
chunk = col1.selectbox("Chunking", CHUNK_STRATEGIES)
embed = col2.selectbox("Embedder", list(EMBED_MODELS))
retr = col3.selectbox("Retriever", RETRIEVER_LADDER, index=2)

query = st.text_input("Ask a question about the papers")

if query:
    pipe = get_pipeline(chunk, embed, retr)
    with st.spinner("Retrieving and answering..."):
        out, sources = pipe.ask(query, k=5)
    st.markdown(out)
    with st.expander(f"Sources ({len(sources)})"):
        for i, c in enumerate(sources, 1):
            sec = f" §{c.section}" if c.section else ""
            st.markdown(f"**[{i}] {c.paper_id} p.{c.page}{sec}**")
            st.caption(c.text)
```

- [ ] **Step 2: Manual smoke test**

Run: `python scripts/build_index.py && streamlit run app/streamlit_app.py`
Expected: app loads; asking a question returns an answer with a populated Sources panel. Confirm a citation `[n]` in the answer maps to the right source.

- [ ] **Step 3: Commit**

```bash
git add app/streamlit_app.py
git commit -m "feat: Streamlit demo with config selector and source panel"
```

---

### Task 15: README, deploy config, blog notes

**Files:**
- Create: `README.md`, `requirements.txt` (HF Spaces), `.streamlit/config.toml` (optional), `docs/blog-notes.md`
- Modify: README embeds `results/retrieval.md` and `results/generation.md`.

**Interfaces:**
- Consumes: `results/*.md`.
- Produces: README with (1) the retrieval table showing the baseline→hybrid+rerank NDCG lift, (2) the 2×2 chunk×embed generation table, (3) a "What I learned" section, (4) run/deploy instructions. `requirements.txt` mirrors `pyproject` deps for HF Spaces (Spaces installs from `requirements.txt`). `docs/blog-notes.md` captures findings to expand into the apoorvaverma.in post.

- [ ] **Step 1: Generate `requirements.txt` from pyproject deps**

Run: `pip freeze | grep -Ei 'anthropic|pymupdf|sentence-transformers|rank-bm25|chromadb|deepeval|streamlit|python-dotenv|numpy' > requirements.txt`
Expected: a `requirements.txt` with pinned versions.

- [ ] **Step 2: Write `README.md`**

Sections, in order: project one-liner; the two result tables (paste from `results/retrieval.md` and `results/generation.md`, with the headline baseline→hybrid+rerank NDCG@5 lift called out in prose, e.g. "Dense 0.62 → Hybrid+rerank 0.91"); architecture diagram (the `rag/` + `eval/` tree from the spec); "Chunking × embedding comparison" paragraph interpreting Table B (which cell wins on faithfulness vs. recall and why); "What I learned" (empirical findings + failure modes); "Run it" (env setup, `build_index.py`, `run_matrix.py`, `pytest`, `streamlit run`); "Deploy" (HF Spaces). Include exact commands.

- [ ] **Step 3: Write `docs/blog-notes.md`**

Bullet the findings to expand into the blog post: the NDCG lift numbers, the chunk/embed tradeoffs observed, surprises, and what you'd do next.

- [ ] **Step 4: Configure HF Spaces deploy**

Create the Space (SDK: Streamlit), set `ANTHROPIC_API_KEY` as a Space secret, set the app entrypoint to `app/streamlit_app.py`. Commit `data/papers/*.pdf` and `data/gold.jsonl` so the Space can build the index (or commit a prebuilt `chroma_db/` — decide based on Space build limits). Document the exact steps taken in the README "Deploy" section.

- [ ] **Step 5: Final full-suite verification**

Run: `pytest -v`
Expected: all tests pass. Paste the summary line into the commit message.

- [ ] **Step 6: Commit**

```bash
git add README.md requirements.txt docs/blog-notes.md .streamlit/ 2>/dev/null; git add -A
git commit -m "docs: README with eval tables, deploy config, blog notes"
```

---

## Self-Review

**Spec coverage:**
- Ingest→chunk→embed→Chroma → Tasks 2–5 ✓
- Hybrid retrieval + cross-encoder rerank → Task 6 ✓
- Citations with exact page/section → `Chunk` provenance (Tasks 2–3), `format_context`/`answer` (Task 7) ✓
- DeepEval faithfulness/relevance/context precision+recall → Tasks 11–12 ✓
- Retrieval precision/recall (NDCG/Recall/MRR baseline→hybrid+rerank) → Tasks 9, 13 ✓
- 2 chunking × 2 embedding matrix table → Task 13 ✓
- Gold set (LLM-synthesized, hand-checked, committed) → Task 10 ✓
- Streamlit UI → Task 14 ✓
- README tables + chunking/embedding comparison + "what I learned" → Task 15 ✓
- HF Spaces deploy + blog → Task 15 ✓
- Claude models / no banned params / `.env` secret → Global Constraints + Tasks 1, 7, 11 ✓

**Placeholder scan:** No "TBD/handle edge cases/similar to Task N" steps; the one explicit verification note (DeepEval `DeepEvalBaseLLM` signature in Task 11) is an instruction to confirm an external library API at implementation time, with the exact command to check it — not a deferred design decision.

**Type consistency:** `Chunk` fields and `chunk_id` format (`paper_id:page:ordinal`) are consistent across ingest/chunk/store/retrieve/generate. `ExperimentConfig(chunk_strategy, embedder, retriever)` and `.collection_name` used consistently. `GoldItem` fields match between `gold.py`, the gate, and the matrix runner. Metric dict keys (`faithfulness/answer_relevance/context_precision/context_recall`) match between `metrics.evaluate_case`, the gate, and `run_matrix.generation_table`.
