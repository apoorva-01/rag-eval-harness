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
