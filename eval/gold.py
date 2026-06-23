"""Synthetic gold-set generation and loading.

KNOWN LIMITATION (named, not hidden): each gold item's question is generated FROM a
single source chunk, and that same chunk is recorded as the only ground-truth. This is
structurally circular for *retrieval* evaluation — the labeled-relevant chunk is the one
the question was written from, so it tends to rank highly under every retriever, which
COMPRESSES the separation between Dense / Hybrid / Hybrid+rerank rather than inflating any
single number. Two further caveats: (1) single-positive ground truth means Recall@k is
deflated whenever other chunks are also genuinely relevant; (2) the set is LLM-synthesized
and not hand-verified. We de-lexicalize the questions below to reduce keyword leakage, but
the structural circularity remains — read the retrieval table with that in mind, and treat
the gold-INDEPENDENT citation metric (eval/citation_metric.py) as the headline signal.
"""

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
    "Given a passage from a research paper, write ONE specific question that is fully "
    "answerable from this passage alone, plus its concise reference answer. "
    "Constraints to keep the question a fair retrieval test, not a keyword echo: "
    "(1) do NOT copy distinctive phrases, numbers, or terminology verbatim from the "
    "passage — paraphrase the concept instead; "
    "(2) the question must require understanding the passage's meaning, not matching its "
    "surface words; "
    "(3) avoid questions that need outside context. "
    "Write the question as a reader who has NOT seen this exact passage would phrase it."
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
