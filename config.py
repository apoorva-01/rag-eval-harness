import os
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
PAPERS_DIR = ROOT / "data" / "papers"
GOLD_PATH = ROOT / "data" / "gold.jsonl"
CHROMA_DIR = ROOT / "chroma_db"

GEN_MODEL = "claude-opus-4-8"
JUDGE_MODEL = "claude-opus-4-8"

EMBED_MODELS = {
    "bge_small": "BAAI/bge-small-en-v1.5",
    "e5_large": "intfloat/e5-large-v2",
}
RERANKER_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_STRATEGIES = ["fixed", "semantic"]
RETRIEVER_LADDER = ["dense", "hybrid", "hybrid_rerank"]


@dataclass(frozen=True)
class ExperimentConfig:
    chunk_strategy: str
    embedder: str
    retriever: str = "hybrid_rerank"

    @property
    def collection_name(self) -> str:
        return f"{self.chunk_strategy}__{self.embedder}"


MATRIX = [
    ExperimentConfig(c, e)
    for c in CHUNK_STRATEGIES
    for e in EMBED_MODELS
]


def anthropic_client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set (see .env.example)")
    return anthropic.Anthropic(api_key=key)
