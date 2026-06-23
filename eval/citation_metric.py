"""Custom citation-faithfulness metric (ALCE-style).

This is the gold-independent core of the harness: it never touches the (synthetic)
gold set. It audits the answer against the *cited* sources only, so it measures the
standout feature of the system — inline [n] page/section citations — directly.

Two components, in the established attribution-eval terms:
- citation precision: of the claims that carry a citation, the fraction whose cited
  sources actually support them. (When you cite, are you right?)
- citation recall: of all factual claims in the answer, the fraction that carry at
  least one citation. (Do you cite everything you assert?)

Honest refusals / answers with no factual claims are EXCLUDED (score None), not scored
0 — penalizing a model for correctly saying "the sources don't cover this" would be wrong.
"""

import json
import re

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config import JUDGE_MODEL, anthropic_client

_CITE = re.compile(r"\[(\d+)\]")

_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "is_factual_claim": {"type": "boolean"},
                    "cited_sources": {"type": "array", "items": {"type": "integer"}},
                    "supported_by_cited": {"type": "boolean"},
                },
                "required": [
                    "claim", "is_factual_claim", "cited_sources", "supported_by_cited",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_SYS = (
    "You audit inline citations in an answer about research papers. The answer cites "
    "numbered sources like [1], [2] that correspond to the numbered Sources block. "
    "Decompose the answer into individual claims. For each claim report: the claim text; "
    "is_factual_claim (true if it asserts a checkable fact about the papers; false for "
    "hedges, refusals, or meta-commentary such as 'the sources do not say'); cited_sources "
    "(the source numbers cited on that claim, [] if none); supported_by_cited (true ONLY if "
    "the cited sources actually state the claim — false if the claim is uncited, or if a "
    "cited source does not actually support it). Be strict: a topically-related source that "
    "does not state the specific claim is NOT support."
)


def parse_citation_indices(answer: str) -> list[int]:
    """Pure: the source numbers cited inline in the answer, in order (with repeats)."""
    return [int(m) for m in _CITE.findall(answer)]


def _number_texts(texts: list[str]) -> str:
    return "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts, start=1))


def _audit(question: str, answer: str, numbered_context: str) -> list[dict]:
    client = anthropic_client()
    msg = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_SYS,
        messages=[{
            "role": "user",
            "content": (
                f"Sources:\n\n{numbered_context}\n\n"
                f"Question: {question}\n\nAnswer:\n{answer}"
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text)["claims"]


def _precision_recall(claims: list[dict]) -> dict:
    factual = [c for c in claims if c.get("is_factual_claim")]
    if not factual:
        # No factual claims (e.g. an honest "not in the sources" refusal): undefined.
        return {"citation_precision": None, "citation_recall": None}
    cited = [c for c in factual if c.get("cited_sources")]
    precision = (
        sum(1 for c in cited if c.get("supported_by_cited")) / len(cited)
        if cited else None  # made claims but cited nothing -> precision undefined
    )
    recall = len(cited) / len(factual)
    return {"citation_precision": precision, "citation_recall": recall}


def citation_scores(question: str, answer: str, retrieved_chunks) -> dict:
    """ALCE-style citation precision & recall for one (answer, sources) pair.

    `retrieved_chunks` is the ordered list of Chunks the answer was generated from;
    [n] maps to retrieved_chunks[n-1] via the same numbering as generation. Returns
    {citation_precision, citation_recall}, each a float in [0,1] or None when undefined.
    """
    from rag.generate import format_context
    claims = _audit(question, answer, format_context(retrieved_chunks))
    return _precision_recall(claims)


class CitationFaithfulnessMetric(BaseMetric):
    """DeepEval-compatible wrapper exposing citation precision as the score.

    Lets the metric be used in pytest assertions like the built-in DeepEval metrics.
    Score = citation precision; an answer with no factual claims (a clean refusal)
    is treated as success (score 1.0) rather than penalized.
    """

    _required_params = [
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ]

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        numbered = _number_texts(test_case.retrieval_context or [])
        claims = _audit(test_case.input, test_case.actual_output, numbered)
        pr = _precision_recall(claims)
        precision = pr["citation_precision"]
        if precision is None:
            self.score, self.success = 1.0, True
            self.reason = "No cited factual claims to verify (e.g. a refusal)."
        else:
            self.score = precision
            self.success = precision >= self.threshold
            self.reason = (
                f"Citation precision {precision:.2f} "
                f"(recall {pr['citation_recall']:.2f})."
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Citation Faithfulness"
