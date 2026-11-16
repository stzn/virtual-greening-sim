"""評価用フィクスチャ: 実LLMの準備（未設定ならスキップ）とスコア記録。"""

import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.llm import MissingApiKeyError, make_chat_model
from vmgs.agent.prompts import PROMPT_VERSION
from vmgs.config import Settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def eval_model(settings: Settings) -> BaseChatModel:
    if os.environ.get("EVAL_FAKE"):
        # 配線確認用のスモークモード: 状態確認 → 現状維持 を毎ターン繰り返すFake（実LLMは使わない）
        return ScriptedChatModel(
            responses=[
                tool_call_message("get_plant_state", {}, "c"),
                AIMessage(content='{"action_summary": "現状維持", "rationale": "fake"}'),
            ],
            cycle=True,
        )
    if settings.llm_provider == "ollama":
        try:
            tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3).json()
        except httpx.HTTPError:
            pytest.skip(
                f"Ollama に接続できません（{settings.ollama_base_url}）。`ollama serve` を起動してください"
            )
        names = {m["name"] for m in tags.get("models", [])}
        wanted = settings.llm_model
        if wanted not in names and f"{wanted}:latest" not in names:
            pytest.skip(f"Ollama にモデルがありません: `ollama pull {wanted}`")
    try:
        return make_chat_model(settings)
    except MissingApiKeyError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="session")
def scoreboard(settings: Settings) -> Iterator[list[dict[str, Any]]]:
    """全シナリオの結果を集め、セッション終了時に runs/evals/ へ JSON で保存する。"""
    rows: list[dict[str, Any]] = []
    yield rows
    if not rows:
        return
    out_dir = Path("runs/evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    provider, model = (
        ("fake", "scripted")
        if os.environ.get("EVAL_FAKE")
        else (settings.llm_provider, settings.llm_model)
    )
    safe_model = model.replace(":", "_").replace("/", "_")
    path = out_dir / f"{stamp}_{provider}_{safe_model}.json"
    payload = {
        "provider": provider,
        "model": model,
        "prompt_version": os.environ.get("EVAL_PROMPT", PROMPT_VERSION),
        "seed": int(os.environ.get("EVAL_SEED", "0")),
        "scenarios": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    passed = sum(r["passed"] for r in rows)
    print(f"\n[eval] {passed}/{len(rows)} シナリオ合格 → {path}")
