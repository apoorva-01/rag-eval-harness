from config import ExperimentConfig, GOLD_PATH, PAPERS_DIR
from rag.ingest import ingest_dir
from rag.chunk import chunk_blocks
from eval.gold import synthesize, save

if __name__ == "__main__":
    chunks = chunk_blocks(ingest_dir(PAPERS_DIR), "fixed")
    items = synthesize(chunks, n=50)
    save(items, GOLD_PATH)
    print(f"wrote {len(items)} gold items to {GOLD_PATH}")
    print("NOW: hand-correct data/gold.jsonl before committing.")
