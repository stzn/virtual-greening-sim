import pytest

from vmgs.agent.graph import parse_decision
from vmgs.agent.llm import MissingApiKeyError, make_chat_model
from vmgs.agent.prompts import SYSTEM_PROMPT_V1, TurnSummary, build_turn_input
from vmgs.config import Settings


def test_parse_decision_はJSONを構造化する() -> None:
    d = parse_decision('{"action_summary": "Bへ退避", "rationale": "猛暑", "risk_notes": "電池"}')
    assert (d.action_summary, d.rationale, d.risk_notes) == ("Bへ退避", "猛暑", "電池")


def test_parse_decision_はコードフェンス付きJSONも読める() -> None:
    d = parse_decision('```json\n{"action_summary": "維持"}\n```')
    assert d.action_summary == "維持"


@pytest.mark.parametrize("text", ["散水した", "[1, 2]", ""])
def test_parse_decision_はJSONでなければ本文を要約にする(text: str) -> None:
    assert parse_decision(text).action_summary == text.strip()


def test_ターン入力に履歴が入り_なければなしと書く() -> None:
    with_hist = build_turn_input(
        "13:00", "C", "歩道", [TurnSummary(time="12:00", action_summary="散水")]
    )
    assert "12:00 散水" in with_hist
    assert "直近の判断: なし" in build_turn_input("13:00", "C", "歩道", [])


def test_システムプロンプトは役割と最終回答の形式を含む() -> None:
    assert "移動緑化ユニットの自律制御AI" in SYSTEM_PROMPT_V1
    assert "action_summary" in SYSTEM_PROMPT_V1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_APIキーが無ければ明確なエラー(provider: str) -> None:
    settings = Settings(_env_file=None, llm_provider=provider)  # type: ignore[call-arg, arg-type]
    with pytest.raises(MissingApiKeyError):
        make_chat_model(settings)


def test_ollamaはAPIキー無しでモデルを作れる() -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="qwen3:14b")  # type: ignore[call-arg]
    model = make_chat_model(settings)
    assert type(model).__name__ == "ChatOllama"
