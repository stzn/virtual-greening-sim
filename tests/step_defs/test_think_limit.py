"""思考の長さの制限（構成の +think=<値>）の解釈・配線・識別情報。"""

from typing import Any

import pytest
from pytest_bdd import parsers, scenarios, then, when

from vmgs import cli
from vmgs.agent.llm import make_chat_model
from vmgs.experiment.configs import parse_config_spec

scenarios("agent/think_limit.feature")


class LimitCtx:
    def __init__(self) -> None:
        self.chat: Any = None


@pytest.fixture
def lc() -> LimitCtx:
    return LimitCtx()


def _make(
    lc: LimitCtx,
    provider: str,
    model: str,
    limit: str | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    monkeypatch.chdir(str(tmp_path))  # 実際の .env を読まない
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")  # 通信しない。生成だけ
    for name in ("OLLAMA_REASONING", "OLLAMA_TIMEOUT_S", "ANTHROPIC_THINKING"):
        monkeypatch.delenv(name, raising=False)
    lc.chat = make_chat_model(cli._llm_settings(provider, model, True, limit))


@when(parsers.parse('思考の制限 "{limit}" つきで Anthropic の "{model}" の LLM 設定を作る'))
def _w_anthropic(
    lc: LimitCtx, limit: str, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(lc, "anthropic", model, limit, monkeypatch, tmp_path)


@when(parsers.parse('思考の制限なしで Anthropic の "{model}" の LLM 設定を作る'))
def _w_anthropic_none(
    lc: LimitCtx, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(lc, "anthropic", model, None, monkeypatch, tmp_path)


@when(parsers.parse('思考の制限 "{limit}" つきで Ollama の "{model}" の LLM 設定を作る'))
def _w_ollama(
    lc: LimitCtx, limit: str, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(lc, "ollama", model, limit, monkeypatch, tmp_path)


@when(parsers.parse('思考の制限なしで Ollama の "{model}" の LLM 設定を作る'))
def _w_ollama_none(
    lc: LimitCtx, model: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    _make(lc, "ollama", model, None, monkeypatch, tmp_path)


@then(parsers.parse('生成されるチャットモデルの reasoning_effort は "{v}" である'))
def _t_effort(lc: LimitCtx, v: str) -> None:
    assert lc.chat.reasoning_effort == v


@then("生成されるチャットモデルの reasoning_effort は 指定なし である")
def _t_effort_none(lc: LimitCtx) -> None:
    assert lc.chat.reasoning_effort is None


@then("生成されるチャットモデルの thinking は adaptive で要約つきである")
def _t_adaptive(lc: LimitCtx) -> None:
    assert lc.chat.thinking == {"type": "adaptive", "display": "summarized"}


@then(parsers.parse("生成される Ollama のチャットモデルの num_predict は {n:d} である"))
def _t_num_predict(lc: LimitCtx, n: int) -> None:
    assert lc.chat.num_predict == n


@then("生成される Ollama のチャットモデルの num_predict は 指定なし である")
def _t_num_predict_none(lc: LimitCtx) -> None:
    assert lc.chat.num_predict is None


@then("生成される Ollama のチャットモデルの reasoning は 有効 である")
def _t_ollama_reasoning(lc: LimitCtx) -> None:
    assert lc.chat.reasoning is True


@then(
    parsers.parse(
        '構成の指定 "{text}" は プロバイダ "{p}"・モデル "{m}"・プロンプト "{v}" と解釈される'
    )
)
def _t_spec(text: str, p: str, m: str, v: str) -> None:
    spec = parse_config_spec(text)
    assert (spec.provider, spec.model, spec.prompt) == (p, m, v)


@then(parsers.parse('構成の指定 "{text}" の思考の制限は "{limit}" である'))
def _t_spec_limit(text: str, limit: str) -> None:
    assert parse_config_spec(text).think_limit == limit


@then(parsers.parse('構成の指定 "{text}" の思考の制限は なし である'))
def _t_spec_limit_none(text: str) -> None:
    assert parse_config_spec(text).think_limit is None


@then(parsers.parse('構成 "{text}" の名前は "{name}" になる'))
def _t_named_spec(text: str, name: str) -> None:
    assert parse_config_spec(text).name == name


@then(parsers.parse('構成の指定 "{text}" はエラーになり、メッセージに "{msg}" が含まれる'))
def _t_spec_error(text: str, msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_config_spec(text)
    assert msg in str(e.value)


@then(parsers.parse('思考の制限 "{limit}" の識別情報に think_limit={v} が含まれる'))
def _t_info(limit: str, v: str, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    info = cli._agent_info("llm", "anthropic", "claude-haiku-5-5", "plan_early", True, limit)
    assert info is not None
    assert info["think_limit"] == v


@then("思考の制限なしの識別情報に think_limit のキーは含まれない")
def _t_info_none(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    info = cli._agent_info("llm", "anthropic", "claude-haiku-5-5", "plan_early", True)
    assert info is not None
    assert "think_limit" not in info
