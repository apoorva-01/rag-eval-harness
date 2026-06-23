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
    page_ordinals: dict[tuple[str, int], int] = {}
    for b in blocks:
        start = 0
        text = b.text
        key = (b.paper_id, b.page)
        while start < len(text):
            piece = text[start:start + FIXED_SIZE].strip()
            if piece:
                ordinal = page_ordinals.get(key, 0)
                out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                                 b.paper_id, b.page, b.section, piece))
                page_ordinals[key] = ordinal + 1
            start += FIXED_SIZE - FIXED_OVERLAP
    return out


def _semantic(blocks: list[Block]) -> list[Chunk]:
    out: list[Chunk] = []
    page_ordinals: dict[tuple[str, int], int] = {}
    for b in blocks:
        sentences = _SENT.split(b.text)
        buf = ""
        key = (b.paper_id, b.page)
        for s in sentences:
            if buf and len(buf) + len(s) > SEMANTIC_MAX:
                ordinal = page_ordinals.get(key, 0)
                out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                                 b.paper_id, b.page, b.section, buf.strip()))
                page_ordinals[key] = ordinal + 1
                buf = ""
            buf += (" " if buf else "") + s
        if buf.strip():
            ordinal = page_ordinals.get(key, 0)
            out.append(Chunk(f"{b.paper_id}:{b.page}:{ordinal}",
                             b.paper_id, b.page, b.section, buf.strip()))
            page_ordinals[key] = ordinal + 1
    return out


def chunk_blocks(blocks: list[Block], strategy: str) -> list[Chunk]:
    if strategy == "fixed":
        return _fixed(blocks)
    if strategy == "semantic":
        return _semantic(blocks)
    raise ValueError(f"unknown strategy {strategy!r}")
