from rag.chunk import Chunk
from rag.generate import format_context, answer


def test_format_context_numbers_and_provenances():
    chunks = [Chunk("d:3:0", "attention", 3, "2 Method", "Self-attention scales.")]
    ctx = format_context(chunks)
    assert "[1]" in ctx and "attention p.3" in ctx and "§2 Method" in ctx


def test_answer_grounds_and_cites():  # real Anthropic API
    chunks = [Chunk("d:3:0", "attn", 3, "Method",
                    "The model uses 8 attention heads.")]
    out = answer("How many attention heads does the model use?", chunks)
    assert "8" in out
    assert "[1]" in out


def test_answer_refuses_when_unsupported():  # real Anthropic API
    chunks = [Chunk("d:3:0", "attn", 3, "Method", "The model uses 8 heads.")]
    out = answer("What is the capital of France?", chunks).lower()
    assert "source" in out or "not" in out
