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
