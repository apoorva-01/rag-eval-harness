from config import ExperimentConfig
from rag.pipeline import RAGPipeline


def test_end_to_end_one_question():
    cfg = ExperimentConfig("fixed", "bge_small", "hybrid_rerank")
    pipe = RAGPipeline(cfg)
    pipe.build()
    out, sources = pipe.ask("What problem does this paper address?", k=5)
    assert out.strip()
    assert 1 <= len(sources) <= 5
    assert all(s.page >= 1 for s in sources)
