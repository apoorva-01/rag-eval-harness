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
