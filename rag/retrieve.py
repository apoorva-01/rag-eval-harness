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
