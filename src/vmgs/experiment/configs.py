"""比較実験の構成指定（例: baseline / noop / llm:ollama:qwen3:14b:v9）の解釈。"""

import re
from dataclasses import dataclass

PROVIDERS = {"openai", "anthropic", "ollama"}
_PROMPT = re.compile(r"v\d+|free|hybrid")


@dataclass(frozen=True)
class ConfigSpec:
    kind: str  # baseline / noop / llm
    provider: str | None = None
    model: str | None = None
    prompt: str | None = None

    @property
    def name(self) -> str:
        """実行IDに使える安全な名前。"""
        if self.kind != "llm":
            return self.kind
        parts = [p for p in ("llm", self.provider, self.model, self.prompt) if p]
        return re.sub(r"[^A-Za-z0-9.\-]+", "_", "_".join(parts))


def parse_config_spec(text: str) -> ConfigSpec:
    """`llm:<provider>:<model>:<prompt>` のモデル名は `:` を含みうる（qwen3:14b）ので、末尾が v数字 / free / hybrid ならプロンプト。"""
    if text in ("baseline", "noop", "llm"):
        return ConfigSpec(text)
    tokens = text.split(":")
    if tokens[0] != "llm" or len(tokens) < 3 or tokens[1] not in PROVIDERS:
        raise ValueError(
            f"構成の指定が不正です: {text!r}（baseline / noop / llm / llm:<provider>:<model>[:v数字]）"
        )
    prompt = tokens[-1] if len(tokens) > 3 and _PROMPT.fullmatch(tokens[-1]) else None
    model_tokens = tokens[2:-1] if prompt else tokens[2:]
    return ConfigSpec("llm", provider=tokens[1], model=":".join(model_tokens), prompt=prompt)
