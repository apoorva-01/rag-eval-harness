from config import MATRIX
from rag.pipeline import RAGPipeline

if __name__ == "__main__":
    for cfg in MATRIX:
        print(f"building {cfg.collection_name} ...")
        RAGPipeline(cfg).build()
    print("done")
