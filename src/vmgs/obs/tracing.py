"""LangSmith トレースの設定と、各ターンの実行情報。

トレースはプロンプトと応答を外部サービス（LangSmith）へ送る。そのため既定は無効で、
明示的な指定（--trace または LANGSMITH_TRACING=true）とAPIキーがそろったときだけ有効にする。
"""

import os
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any

from vmgs.config import Settings

SEND_WARNING = (
    "トレースを有効にしました。各ターンのプロンプトと応答が LangSmith（外部サービス）へ送信されます。"
    "機密情報を含めないでください。"
)


@dataclass(frozen=True)
class TracingConfig:
    enabled: bool
    env: dict[str, str] = field(default_factory=dict)  # 有効なときにプロセスへ設定する環境変数
    reason: str | None = None  # 無効の理由
    warnings: list[str] = field(default_factory=list)
    project: str | None = None

    def describe(self) -> str:
        """人が読む説明。APIキーの値は含めない。"""
        if self.enabled:
            return f"LangSmith トレース: 有効（プロジェクト {self.project}）"
        return f"LangSmith トレース: 無効（{self.reason}）"


def resolve_tracing(
    settings: Settings, *, enable: bool = False, project: str | None = None
) -> TracingConfig:
    requested = enable or settings.langsmith_tracing
    if not requested:
        return TracingConfig(
            enabled=False,
            reason="明示的に有効化されていません（--trace または LANGSMITH_TRACING=true）",
        )
    if not settings.langsmith_api_key:
        return TracingConfig(
            enabled=False, reason="LANGSMITH_API_KEY（APIキー）が未設定のため無効にしました"
        )
    proj = project or settings.langsmith_project
    return TracingConfig(
        enabled=True,
        env={
            "LANGSMITH_TRACING": "true",
            "LANGSMITH_PROJECT": proj,
            "LANGSMITH_API_KEY": settings.langsmith_api_key,
        },
        warnings=[SEND_WARNING],
        project=proj,
    )


def apply_tracing(cfg: TracingConfig, environ: MutableMapping[str, str] | None = None) -> None:
    """有効なときだけ、トレース用の環境変数をプロセスへ設定する。"""
    target = os.environ if environ is None else environ
    if cfg.enabled:
        target.update(cfg.env)


def build_run_config(
    *,
    run_id: str,
    scenario: str,
    step: int,
    sim_time: str,
    prompt_version: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """LangChain の config（run_name / tags / metadata）。トレースで実行を特定できるようにする。"""
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "scenario": scenario,
        "step": step,
        "sim_time": sim_time,
    }
    tags = [f"scenario:{scenario}"]
    if prompt_version:
        metadata["prompt_version"] = prompt_version
        tags.append(f"prompt:{prompt_version}")
    if model:
        metadata["model"] = model
        tags.append(f"model:{model}")
    return {"run_name": f"turn-{step}", "tags": tags, "metadata": metadata}
