import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # pymupdf


@dataclass
class Block:
    paper_id: str
    page: int
    section: str
    text: str


_HEADING = re.compile(r"^(\d+(\.\d+)*)\s+[A-Z].{0,80}$")


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    return bool(_HEADING.match(line)) or (line.isupper() and len(line.split()) <= 8)


def ingest_pdf(path: Path) -> list[Block]:
    paper_id = path.stem
    doc = fitz.open(path)
    blocks: list[Block] = []
    section = ""
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text")
        lines = [ln for ln in text.splitlines()]
        buffer: list[str] = []

        def flush():
            joined = " ".join(buffer).strip()
            if joined:
                blocks.append(Block(paper_id, page_index + 1, section, joined))
            buffer.clear()

        for line in lines:
            if _looks_like_heading(line):
                flush()
                section = line.strip()
            else:
                buffer.append(line)
        flush()
    doc.close()
    return blocks


def ingest_dir(papers_dir: Path) -> list[Block]:
    out: list[Block] = []
    for pdf in sorted(papers_dir.glob("*.pdf")):
        out.extend(ingest_pdf(pdf))
    return out
