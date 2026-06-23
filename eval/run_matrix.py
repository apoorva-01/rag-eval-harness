import csv
import statistics
from pathlib import Path

from config import MATRIX, RETRIEVER_LADDER, GOLD_PATH, ExperimentConfig
from eval.gold import load
from eval.retrieval_metrics import ndcg_at_k, recall_at_k, mrr
from eval.metrics import evaluate_case
from rag.pipeline import RAGPipeline

RESULTS = Path(__file__).parent.parent / "results"


def _mean(xs):
    return round(statistics.mean(xs), 3) if xs else 0.0


def retrieval_table() -> list[dict]:
    gold = load(GOLD_PATH)
    rows = []
    for mode in RETRIEVER_LADDER:
        pipe = RAGPipeline(ExperimentConfig("fixed", "bge_small", mode))
        pipe.build()
        nd, rc, rr = [], [], []
        for item in gold:
            ids = [c.chunk_id for c in pipe.retrieve(item.question, k=5)]
            gt = item.ground_truth_chunk_ids
            nd.append(ndcg_at_k(ids, gt, 5))
            rc.append(recall_at_k(ids, gt, 5))
            rr.append(mrr(ids, gt))
        rows.append({"retriever": mode, "ndcg@5": _mean(nd),
                     "recall@5": _mean(rc), "mrr": _mean(rr)})
    return rows


def generation_table() -> list[dict]:
    gold = load(GOLD_PATH)
    rows = []
    for cfg in MATRIX:
        pipe = RAGPipeline(cfg)
        pipe.build()
        agg = {"faithfulness": [], "answer_relevance": [],
               "context_precision": [], "context_recall": []}
        for item in gold:
            out, sources = pipe.ask(item.question, k=5)
            s = evaluate_case(item.question, out, item.reference_answer, sources)
            for k in agg:
                agg[k].append(s[k])
        row = {"config": cfg.collection_name}
        row.update({k: _mean(v) for k, v in agg.items()})
        rows.append(row)
    return rows


def to_markdown(rows: list[dict], headers: list[str]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(r[h]) for h in headers) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _write(name, rows, headers):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.md").write_text(to_markdown(rows, headers))
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rt = retrieval_table()
    _write("retrieval", rt, ["retriever", "ndcg@5", "recall@5", "mrr"])
    gt = generation_table()
    _write("generation", gt, ["config", "faithfulness", "answer_relevance",
                              "context_precision", "context_recall"])
    print("wrote results/retrieval.{md,csv} and results/generation.{md,csv}")
