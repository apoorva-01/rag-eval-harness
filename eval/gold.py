import json
import random
from dataclasses import dataclass, asdict

from config import JUDGE_MODEL, anthropic_client
from rag.chunk import Chunk

_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
    },
    "required": ["question", "reference_answer"],
    "additionalProperties": False,
}

_SYS = (
    "Given a passage from a research paper, write ONE specific question that is "
    "fully answerable from this passage alone, and its concise reference answer. "
    "Avoid questions needing outside context."
)


@dataclass
class GoldItem:
    question: str
    reference_answer: str
    ground_truth_chunk_ids: list[str]
    source_pages: list[int]


def _one(client, chunk: Chunk) -> GoldItem:
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=_SYS,
        messages=[{"role": "user", "content": chunk.text}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    data = json.loads(text)
    return GoldItem(data["question"], data["reference_answer"],
                    [chunk.chunk_id], [chunk.page])


def synthesize(chunks: list[Chunk], n: int) -> list[GoldItem]:
    client = anthropic_client()
    pool = [c for c in chunks if len(c.text) > 200]
    random.seed(0)
    sample = random.sample(pool, min(n, len(pool)))
    return [_one(client, c) for c in sample]


def save(items: list[GoldItem], path) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")


def load(path) -> list[GoldItem]:
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(GoldItem(**json.loads(line)))
    return items
