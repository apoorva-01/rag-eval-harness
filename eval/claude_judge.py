"""Claude as the DeepEval metric judge.

DeepEval drives metric judging through `generate_with_schema(prompt, schema_cls)` /
`a_generate_with_schema(...)`, where `schema_cls` is a pydantic model. It prefers a
returned schema INSTANCE (which it uses directly); otherwise it falls back to parsing
the returned string as JSON, which is brittle. So we return a validated instance, using
Anthropic structured outputs to force schema-conforming JSON, with a plain-call +
JSON-extraction fallback for any schema Anthropic's constrained decoding rejects.
"""

import json
import re

from deepeval.models import DeepEvalBaseLLM

from config import JUDGE_MODEL, anthropic_client

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def _strictify(node):
    """Recursively enforce Anthropic structured-output requirements on a JSON schema:
    every object gets additionalProperties=false and all its properties required."""
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
    return node


def _extract_json(text: str) -> dict:
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return json.loads(text)


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic_client()

    def load_model(self):
        return self.client

    def _text(self, msg) -> str:
        # Only the final text blocks — thinking blocks are separate and excluded.
        return "".join(b.text for b in msg.content if b.type == "text")

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self.generate_with_schema(prompt, schema)
        msg = self.client.messages.create(
            model=JUDGE_MODEL, max_tokens=2048,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return self._text(msg)

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def generate_with_schema(self, prompt: str, schema):
        try:
            json_schema = _strictify(schema.model_json_schema())
            msg = self.client.messages.create(
                model=JUDGE_MODEL, max_tokens=2048,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": json_schema}},
            )
            return schema.model_validate_json(self._text(msg))
        except Exception:
            # Fallback: ask plainly for JSON, then extract + validate.
            msg = self.client.messages.create(
                model=JUDGE_MODEL, max_tokens=2048,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": (
                    prompt + "\n\nRespond with ONLY a JSON object matching the requested "
                    "schema. No markdown, no commentary."
                )}],
            )
            return schema.model_validate(_extract_json(self._text(msg)))

    async def a_generate_with_schema(self, prompt: str, schema):
        return self.generate_with_schema(prompt, schema)

    def get_model_name(self) -> str:
        return JUDGE_MODEL
