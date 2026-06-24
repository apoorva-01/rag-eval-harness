from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric,
)

from eval.claude_judge import ClaudeJudge
from eval.citation_metric import citation_scores

_judge = None


def evaluate_case(question, generated_answer, reference_answer,
                  retrieved_chunks) -> dict:
    """Run the four DeepEval metrics plus the custom citation metric.

    Returns the six metric keys. `citation_precision`/`citation_recall` may be None
    when undefined (an answer with no cited factual claims, e.g. an honest refusal) —
    callers aggregating into means must skip None.
    """
    global _judge
    if _judge is None:
        _judge = ClaudeJudge()
    context = [c.text for c in retrieved_chunks]
    case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        expected_output=reference_answer,
        retrieval_context=context,
    )
    metrics = {
        "faithfulness": FaithfulnessMetric(model=_judge),
        "answer_relevance": AnswerRelevancyMetric(model=_judge),
        "context_precision": ContextualPrecisionMetric(model=_judge),
        "context_recall": ContextualRecallMetric(model=_judge),
    }
    scores = {}
    for name, metric in metrics.items():
        metric.measure(case)
        scores[name] = metric.score
    scores.update(citation_scores(question, generated_answer, retrieved_chunks))
    return scores
