import csv
import os
import statistics
from pathlib import Path

from config import MATRIX, RETRIEVER_LADDER, GOLD_PATH, ExperimentConfig
from eval.gold import load
from eval.retrieval_metrics import ndcg_at_k, recall_at_k, mrr
from eval.metrics import evaluate_case
from rag.pipeline import RAGPipeline

RESULTS = Path(__file__).parent.parent / "results"


GEN_METRIC_KEYS = [
    "faithfulness", "answer_relevance", "context_precision", "context_recall",
    "citation_precision", "citation_recall",
]


def _mean(xs):
    # Skip None (undefined metric values, e.g. citation scores on a refusal).
    vals = [x for x in xs if x is not None]
    return round(statistics.mean(vals), 3) if vals else None


def retrieval_table() -> list[dict]:
    # Reuse the prebuilt collection via load() — do NOT re-embed per retriever mode.
    # Full gold set: retrieval scoring is local + cheap (no LLM calls).
    gold = load(GOLD_PATH)
    rows = []
    for mode in RETRIEVER_LADDER:
        print(f"[retrieval] {mode} over {len(gold)} questions ...", flush=True)
        pipe = RAGPipeline(ExperimentConfig("fixed", "bge_small", mode))
        pipe.load()
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
    # Reuse the four prebuilt collections via load() — no re-embedding.
    # Each (config, question) costs ~9 LLM calls (generation + 4 DeepEval metrics +
    # citation audit), so the expensive generation sweep is capped at MATRIX_GEN_SAMPLE
    # questions (default 20) for affordability; raise/unset it for the full set.
    gold = load(GOLD_PATH)
    sample = os.environ.get("MATRIX_GEN_SAMPLE", "20")  # default 20; set "0"/"" for all
    if sample and int(sample) > 0:
        gold = gold[:int(sample)]
    rows = []
    for cfg in MATRIX:
        print(f"[generation] {cfg.collection_name} over {len(gold)} questions ...",
              flush=True)
        pipe = RAGPipeline(cfg)
        pipe.load()
        agg = {k: [] for k in GEN_METRIC_KEYS}
        for item in gold:
            out, sources = pipe.ask(item.question, k=5)
            s = evaluate_case(item.question, out, item.reference_answer, sources)
            for k in agg:
                agg[k].append(s[k])
        row = {"config": cfg.collection_name}
        row.update({k: _mean(v) for k, v in agg.items()})
        rows.append(row)
    return rows


def _cell(v):
    return "—" if v is None else str(v)


def to_markdown(rows: list[dict], headers: list[str]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_cell(r[h]) for h in headers) + " |" for r in rows]
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
    _write("generation", gt, ["config", *GEN_METRIC_KEYS])
    print("wrote results/retrieval.{md,csv} and results/generation.{md,csv}")
