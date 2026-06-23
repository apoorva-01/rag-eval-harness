from config import GEN_MODEL, anthropic_client
from rag.chunk import Chunk

_SYSTEM = (
    "You answer questions strictly from the provided sources about research "
    "papers. Cite every claim inline with bracketed source numbers like [1]. "
    "Each source is labeled with its paper, page, and section. If the sources "
    "do not contain the answer, say so plainly — do not use outside knowledge."
)


def format_context(chunks: list[Chunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, start=1):
        sec = f" §{c.section}" if c.section else ""
        lines.append(f"[{i}] ({c.paper_id} p.{c.page}{sec})\n{c.text}")
    return "\n\n".join(lines)


def answer(query: str, chunks: list[Chunk], model: str = GEN_MODEL) -> str:
    client = anthropic_client()
    context = format_context(chunks)
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM,
        messages=[{
            "role": "user",
            "content": f"Sources:\n\n{context}\n\nQuestion: {query}",
        }],
    )
    return "".join(b.text for b in msg.content if b.type == "text")
