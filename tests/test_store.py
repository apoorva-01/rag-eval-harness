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
