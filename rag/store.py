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
