from rag.embed import Embedder


def test_bge_embeds_with_expected_dim():
    e = Embedder("bge_small")
    v = e.embed_query("retrieval augmented generation")
    assert len(v) == e.dim == 384


def test_doc_and_query_embeddings_same_dim():
    e = Embedder("bge_small")
    d = e.embed_documents(["a passage about transformers"])
    assert len(d) == 1 and len(d[0]) == e.dim
