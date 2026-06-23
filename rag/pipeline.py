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
