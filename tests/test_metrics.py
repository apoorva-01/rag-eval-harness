from rag.chunk import Chunk
from eval.metrics import evaluate_case


def test_faithful_answer_scores_high():
    chunks = [Chunk("d:1:0", "d", 1, "M", "The model has 12 layers.")]
    scores = evaluate_case(
        "How many layers does the model have?",
        "The model has 12 layers [1].",
        "12 layers.",
        chunks,
    )
    assert set(scores) == {"faithfulness", "answer_relevance",
                           "context_precision", "context_recall"}
    assert scores["faithfulness"] >= 0.5
