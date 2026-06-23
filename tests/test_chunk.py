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


def test_chunk_ids_unique_across_blocks_on_same_page():
    blocks = [
        Block("p1", 1, "1 Intro", "Intro text. " * 60),
        Block("p1", 1, "2 Setup", "Setup text. " * 60),  # same paper+page, different section
    ]
    for strat in ("fixed", "semantic"):
        ids = [c.chunk_id for c in chunk_blocks(blocks, strat)]
        assert len(ids) == len(set(ids)), f"{strat} produced duplicate chunk_ids"
