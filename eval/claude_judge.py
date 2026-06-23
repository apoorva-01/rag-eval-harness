from deepeval.models import DeepEvalBaseLLM

from config import JUDGE_MODEL, anthropic_client


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic_client()

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:
        # NOTE: verify DeepEvalBaseLLM.generate signature against installed deepeval version before running
        msg = self.client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=2048,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return JUDGE_MODEL
