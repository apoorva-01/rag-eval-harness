import statistics

import pytest

from config import ExperimentConfig
from eval.gold import load
from eval.retrieval_metrics import ndcg_at_k
from eval.metrics import evaluate_case
from config import GOLD_PATH
from rag.pipeline import RAGPipeline

GOLD = load(GOLD_PATH)[:8]  # small subset for the gate


@pytest.fixture(scope="module")
def pipe():
    p = RAGPipeline(ExperimentConfig("fixed", "bge_small", "hybrid_rerank"))
    p.build()
    return p


def test_retrieval_ndcg_reasonable(pipe):
    scores = []
    for item in GOLD:
        ids = [c.chunk_id for c in pipe.retrieve(item.question, k=5)]
        scores.append(ndcg_at_k(ids, item.ground_truth_chunk_ids, 5))
    assert statistics.mean(scores) >= 0.3


def test_faithfulness_gate(pipe):
    item = GOLD[0]
    out, sources = pipe.ask(item.question, k=5)
    scores = evaluate_case(item.question, out, item.reference_answer, sources)
    assert scores["faithfulness"] >= 0.7
