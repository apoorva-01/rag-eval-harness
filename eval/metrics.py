from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric, AnswerRelevancyMetric,
    ContextualPrecisionMetric, ContextualRecallMetric,
)

from eval.claude_judge import ClaudeJudge

_judge = ClaudeJudge()


def evaluate_case(question, generated_answer, reference_answer,
                  retrieved_chunks) -> dict:
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
    return scores
