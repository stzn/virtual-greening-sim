"""比較実験の構成指定（例: baseline / noop / llm:ollama:qwen3:14b:v9）の解釈。"""

import re
from dataclasses import dataclass

from vmgs.agent.prompts import PROMPT_REQUIRES_PLAN_OBS

PROVIDERS = {"openai", "anthropic", "ollama"}
_PROMPT = re.compile(
    r"(?P<prompt>v\d+|free|hybrid|plan_early_move|plan_early|plan|early)?(?P<think>\+think)?"
)
THINK_TIMEOUT_S = 600.0  # 思考モードの1リクエストの上限秒数（ハング対策）


@dataclass(frozen=True)
class ConfigSpec:
    kind: str  # baseline / noop / llm
    provider: str | None = None
    model: str | None = None
    prompt: str | None = None
    reasoning: bool = False  # Ollama の思考モード（+think）

    @property
    def planning_observations(self) -> bool:
        """このプロンプトは世界側の計画観測を要する（plan を選んだ構成にだけ有効にする）。"""
        return self.kind == "llm" and self.prompt in PROMPT_REQUIRES_PLAN_OBS

    @property
    def name(self) -> str:
        """実行IDに使える安全な名前。"""
        if self.kind != "llm":
            return self.kind
        parts = [
            p
            for p in (
                "llm",
                self.provider,
                self.model,
                self.prompt,
                "think" if self.reasoning else None,
            )
            if p
        ]
        return re.sub(r"[^A-Za-z0-9.\-]+", "_", "_".join(parts))


def parse_config_spec(text: str) -> ConfigSpec:
    """`llm:<provider>:<model>:<prompt>` のモデル名は `:` を含みうる（qwen3:14b）ので、末尾が v数字 / free / hybrid / plan / early / plan_early / plan_early_move（任意で +think）ならプロンプト。"""
    if text in ("baseline", "noop", "llm"):
        return ConfigSpec(text)
    tokens = text.split(":")
    if tokens[0] != "llm" or len(tokens) < 3 or tokens[1] not in PROVIDERS:
        raise ValueError(
            f"構成の指定が不正です: {text!r}（baseline / noop / llm / llm:<provider>:<model>[:v数字]）"
        )
    last = _PROMPT.fullmatch(tokens[-1]) if len(tokens) > 3 else None
    if last is not None and (last["prompt"] or last["think"]):
        prompt, reasoning = last["prompt"], bool(last["think"])
        model_tokens = tokens[2:-1]
    else:
        prompt, reasoning, model_tokens = None, False, tokens[2:]
    if reasoning and tokens[1] not in ("ollama", "anthropic"):
        raise ValueError(
            f"+think（思考モード）は Ollama と Anthropic のみ対応です: {text!r}（プロバイダごとに思考の仕組みが違うため）"
        )
    return ConfigSpec(
        "llm", provider=tokens[1], model=":".join(model_tokens), prompt=prompt, reasoning=reasoning
    )
