import math
from eval.retrieval_metrics import ndcg_at_k, recall_at_k, mrr


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 3) == 0.5


def test_mrr_first_relevant_at_rank_2():
    assert mrr(["x", "a", "b"], {"a"}) == 0.5


def test_ndcg_perfect_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_ndcg_relevant_at_rank_2():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))
