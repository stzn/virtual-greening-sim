"""Claude（Anthropic）の拡張思考の構成指定・配線・思考のトレース。"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli
from vmgs.agent.llm import make_chat_model
from vmgs.config import Settings
from vmgs.experiment.configs import parse_config_spec
from vmgs.obs.telemetry import message_record

scenarios("agent/reasoning_anthropic.feature")


class AnthropicCtx:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.chat: Any = None
        self.message: AIMessage | None = None
        self.record: dict[str, Any] = {}


@pytest.fixture
def ac() -> AnthropicCtx:
    return AnthropicCtx()


def _make(
    ac: AnthropicCtx,
    provider: str,
    model: str,
    reasoning: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    monkeypatch.chdir(str(tmp_path))  # 実際の .env を読まない
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # 通信しない。生成だけ
    for name in ("OLLAMA_REASONING", "OLLAMA_TIMEOUT_S", "ANTHROPIC_THINKING"):
        monkeypatch.delenv(name, raising=False)
    ac.settings = cli._llm_settings(provider, model, reasoning)
    ac.chat = make_chat_model(ac.settings)


@when(parsers.parse('思考モード有効で Anthropic の "{model}" の LLM 設定を作る'))
def _w_think(
    ac: AnthropicCtx, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(ac, "anthropic", model, True, monkeypatch, tmp_path)


@when(parsers.parse('思考モード無効で Anthropic の "{model}" の LLM 設定を作る'))
def _w_plain(
    ac: AnthropicCtx, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(ac, "anthropic", model, False, monkeypatch, tmp_path)


@when("思考モード有効で Ollama の LLM 設定を作る")
def _w_ollama(ac: AnthropicCtx, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _make(ac, "ollama", "qwen3:14b", True, monkeypatch, tmp_path)


@then("生成されるチャットモデルの thinking は adaptive で要約つきである")
def _t_adaptive(ac: AnthropicCtx) -> None:
    assert ac.chat.thinking == {"type": "adaptive", "display": "summarized"}


@then("生成されるチャットモデルの thinking は 無効 である")
def _t_disabled(ac: AnthropicCtx) -> None:
    assert ac.chat.thinking == {"type": "disabled"}


@then("生成されるチャットモデルの温度は 指定なし である")
def _t_temp_none(ac: AnthropicCtx) -> None:
    assert ac.chat.temperature is None


@then(parsers.parse("生成されるチャットモデルの温度は {n:d} である"))
def _t_temp(ac: AnthropicCtx, n: int) -> None:
    assert ac.chat.temperature == n


@then("設定の Anthropic の思考は 指定なし である")
def _t_setting_none(ac: AnthropicCtx) -> None:
    assert ac.settings is not None
    assert ac.settings.anthropic_thinking is None


@then("思考モード有効の Anthropic の識別情報に reasoning=true が含まれる")
def _t_info(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    info = cli._agent_info("llm", "anthropic", "claude-haiku-5-5", "hybrid", reasoning=True)
    assert info is not None
    assert info["reasoning"] == "true"


# ---------------- 構成の指定（test_prompt_free.py の同名ステップと同じ内容） ----------------
@then(
    parsers.parse(
        '構成の指定 "{text}" は プロバイダ "{p}"・モデル "{m}"・プロンプト "{v}" と解釈される'
    )
)
def _t_spec(text: str, p: str, m: str, v: str) -> None:
    spec = parse_config_spec(text)
    assert (spec.provider, spec.model, spec.prompt) == (p, m, v)


@then(parsers.parse('構成の指定 "{text}" の思考モードは {state} である'))
def _t_spec_reasoning(text: str, state: str) -> None:
    assert parse_config_spec(text).reasoning is (state == "有効")


@then(parsers.parse('構成 "{text}" の名前は "{name}" になる'))
def _t_named_spec(text: str, name: str) -> None:
    assert parse_config_spec(text).name == name


@then(parsers.parse('構成の指定 "{text}" はエラーになり、メッセージに "{msg}" が含まれる'))
def _t_spec_error(text: str, msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_config_spec(text)
    assert msg in str(e.value)


# ---------------- 思考ブロックのトレース ----------------
def _blocks(*items: tuple[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for kind, text in items:
        if kind == "thinking":
            out.append({"type": "thinking", "thinking": text, "signature": "sig"})
        else:
            out.append({"type": "text", "text": text})
    return out


@given(parsers.re(r'本文 "(?P<text>.*)" と思考 "(?P<thinking>.*)" を持つ AI メッセージがある'))
def _g_text_thinking(ac: AnthropicCtx, text: str, thinking: str) -> None:
    ac.message = AIMessage(content=_blocks(("thinking", thinking), ("text", text)))


@given(parsers.parse('本文 "{text}" だけの AI メッセージがある'))
def _g_text_only(ac: AnthropicCtx, text: str) -> None:
    ac.message = AIMessage(content=_blocks(("text", text)))


@given(parsers.parse('思考 "{a}" と思考 "{b}" と本文 "{text}" を持つ AI メッセージがある'))
def _g_two_thinking(ac: AnthropicCtx, a: str, b: str, text: str) -> None:
    ac.message = AIMessage(content=_blocks(("thinking", a), ("thinking", b), ("text", text)))


@then(parsers.parse('トレースの記録の reasoning は "{expected}" である'))
def _t_reasoning(ac: AnthropicCtx, expected: str) -> None:
    assert ac.message is not None
    assert message_record(ac.message)["reasoning"] == expected.replace("\\n", "\n")


@then(parsers.parse('トレースの記録の本文は "{expected}" である'))
def _t_content(ac: AnthropicCtx, expected: str) -> None:
    assert ac.message is not None
    assert message_record(ac.message)["content"] == expected


@then("トレースの記録に reasoning のキーは含まれない")
def _t_no_reasoning(ac: AnthropicCtx) -> None:
    assert ac.message is not None
    assert "reasoning" not in message_record(ac.message)
