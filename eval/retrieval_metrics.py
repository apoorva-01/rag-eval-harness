import math


def recall_at_k(retrieved_ids, relevant_ids, k) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit = sum(1 for cid in retrieved_ids[:k] if cid in relevant)
    return hit / len(relevant)


def mrr(retrieved_ids, relevant_ids) -> float:
    relevant = set(relevant_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids, relevant_ids, k) -> float:
    relevant = set(relevant_ids)
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0
