"""チェックリストなしのプロンプト（free）。"""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli
from vmgs.agent.graph import parse_decision
from vmgs.agent.llm import make_chat_model
from vmgs.agent.prompts import (
    PLAN_OBSERVATION_HINT,
    PLANNING_HINTS,
    PROMPTS,
    SYSTEM_PROMPT,
    build_free_prompt,
    build_hybrid_prompt,
)
from vmgs.config import Settings
from vmgs.experiment.configs import parse_config_spec
from vmgs.world.params import Params

scenarios(
    "agent/prompt_free.feature",
    "agent/prompt_hybrid.feature",
    "agent/prompt_plan.feature",
    "agent/reasoning.feature",
)


class PromptCtx:
    def __init__(self) -> None:
        self.text = ""
        self.params = Params()
        self.settings: Settings | None = None
        self.chat: object = None


@pytest.fixture
def pc() -> PromptCtx:
    return PromptCtx()


@given(parsers.parse('プロンプト "{name}" を取得する'))
def _g_prompt(pc: PromptCtx, name: str) -> None:
    pc.text = PROMPTS[name]


@given(parsers.parse("世界の設定で適正水分の範囲が {lo:g}〜{hi:g}% である"))
def _g_params(pc: PromptCtx, lo: float, hi: float) -> None:
    pc.params = Params(moisture_ok_low=float(lo), moisture_ok_high=float(hi))


@when("free プロンプトを生成する")
def _w_build(pc: PromptCtx) -> None:
    pc.text = build_free_prompt(pc.params)


@then(parsers.parse('プロンプトに "{t}" は含まれない'))
def _t_absent(pc: PromptCtx, t: str) -> None:
    assert t not in pc.text


@then(parsers.parse('プロンプトに "{t}" が含まれる'))
def _t_present(pc: PromptCtx, t: str) -> None:
    assert t in pc.text


@then("プロンプトに表形式の行は含まれない")
def _t_no_table(pc: PromptCtx) -> None:
    assert not [ln for ln in pc.text.splitlines() if ln.lstrip().startswith("|")]


@then(parsers.parse('プロンプトに "#" で始まる行は含まれない'))
def _t_no_hash_lines(pc: PromptCtx) -> None:
    assert not [ln for ln in pc.text.splitlines() if ln.lstrip().startswith("#")]


@then("プロンプトに次の判断基準は含まれない")
def _t_no_criteria(pc: PromptCtx, datatable: list[list[str]]) -> None:
    present = [row[0] for row in datatable[1:] if row[0] in pc.text]
    assert not present, present


@then(parsers.parse('プロンプトから計画観測の説明を取り除くと "{name}" と完全に一致する'))
def _t_only_obs_hint_differs(pc: PromptCtx, name: str) -> None:
    assert PLAN_OBSERVATION_HINT in pc.text
    assert pc.text.replace(PLAN_OBSERVATION_HINT, "") == PROMPTS[name]


@then("計画観測の説明に数字は含まれない")
def _t_obs_hint_no_digits(pc: PromptCtx) -> None:
    import re

    assert not re.search(r"[0-9０-９]", PLAN_OBSERVATION_HINT)


@then("計画観測の説明に次の語が含まれる")
def _t_obs_hint_words(pc: PromptCtx, datatable: list[list[str]]) -> None:
    missing = [row[0] for row in datatable[1:] if row[0] not in PLAN_OBSERVATION_HINT]
    assert not missing, missing


@then(parsers.parse('構成の指定 "{text}" の計画観測は {state} である'))
def _t_spec_plan_obs(text: str, state: str) -> None:
    assert parse_config_spec(text).planning_observations is (state == "有効")


@then("プロンプトに次の語が含まれる")
def _t_words(pc: PromptCtx, datatable: list[list[str]]) -> None:
    missing = [row[0] for row in datatable[1:] if row[0] not in pc.text]
    assert not missing, missing


@then(
    parsers.parse(
        "プロンプトは土壌水分の適正を {lo:g}〜{hi:g}%、過湿を {w:g}% 超、高温ストレスを {h:g}℃ 超と述べている"
    )
)
def _t_facts(pc: PromptCtx, lo: float, hi: float, w: float, h: float) -> None:
    p = Params()
    assert (p.moisture_ok_low, p.moisture_ok_high, p.moisture_wet, p.heat_stress_temp_c) == (
        lo,
        hi,
        w,
        h,
    )
    assert f"{lo:g}〜{hi:g}%" in pc.text
    assert f"{w:g}% 超" in pc.text
    assert f"{h:g}℃ 超" in pc.text


@then(parsers.parse("プロンプトは移動を {w:g}時間に {n:g}回までと述べている"))
def _t_move_budget(pc: PromptCtx, w: float, n: float) -> None:
    p = Params()
    assert (p.move_budget_window_hours, p.move_budget_max) == (w, n)
    assert f"{w:g}時間に {n:g}回まで" in pc.text


@then("プロンプトの最終回答の例から行動要約を取り出せる")
def _t_example(pc: PromptCtx) -> None:
    d = parse_decision(pc.text)
    assert d.raw_fallback is False
    assert d.action_summary


@then(parsers.parse('既定のプロンプトは "{name}" ではない'))
def _t_default(name: str) -> None:
    assert SYSTEM_PROMPT != PROMPTS[name]


@then(parsers.parse('プロンプト名 "{name}" は選択肢に含まれる'))
def _t_choice(name: str) -> None:
    assert name in PROMPTS


@then(
    parsers.parse(
        '構成の指定 "{text}" は プロバイダ "{p}"・モデル "{m}"・プロンプト "{v}" と解釈される'
    )
)
def _t_spec(text: str, p: str, m: str, v: str) -> None:
    spec = parse_config_spec(text)
    assert (spec.provider, spec.model, spec.prompt) == (p, m, v)


@when("hybrid プロンプトを生成する")
def _w_build_hybrid(pc: PromptCtx) -> None:
    pc.text = build_hybrid_prompt(pc.params)


@then(parsers.parse('プロンプトから資源計画のヒントを取り除くと "{name}" と完全に一致する'))
def _t_only_hints_differ(pc: PromptCtx, name: str) -> None:
    assert PLANNING_HINTS in pc.text
    assert pc.text.replace(PLANNING_HINTS, "") == PROMPTS[name]


@then("資源計画のヒントに数字は含まれない")
def _t_hints_no_digits(pc: PromptCtx) -> None:
    import re

    assert not re.search(r"[0-9０-９]", PLANNING_HINTS)


@then("資源計画のヒントに次の語が含まれる")
def _t_hint_words(pc: PromptCtx, datatable: list[list[str]]) -> None:
    missing = [row[0] for row in datatable[1:] if row[0] not in PLANNING_HINTS]
    assert not missing, missing


@then(parsers.parse('構成 "{text}" の名前は "{name}" になる'))
def _t_named_spec(text: str, name: str) -> None:
    assert parse_config_spec(text).name == name


@then(parsers.parse('構成の名前は "{name}" になる'))
def _t_spec_name(name: str) -> None:
    assert parse_config_spec("llm:ollama:qwen3:14b:free").name == name


# ---------------- 思考モード（+think） ----------------
@then(parsers.parse('構成の指定 "{text}" の思考モードは {state} である'))
def _t_spec_reasoning(text: str, state: str) -> None:
    assert parse_config_spec(text).reasoning is (state == "有効")


@then(
    parsers.parse(
        '構成の指定 "{text}" は プロバイダ "{p}"・モデル "{m}" で思考モードが有効と解釈される'
    )
)
def _t_spec_think_only(text: str, p: str, m: str) -> None:
    spec = parse_config_spec(text)
    assert (spec.provider, spec.model, spec.prompt, spec.reasoning) == (p, m, None, True)


@then(parsers.parse('構成の指定 "{text}" はエラーになり、メッセージに "{msg}" が含まれる'))
def _t_spec_error(text: str, msg: str) -> None:
    with pytest.raises(ValueError) as e:
        parse_config_spec(text)
    assert msg in str(e.value)


def _make_settings(
    pc: PromptCtx, reasoning: bool, monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    monkeypatch.chdir(str(tmp_path))  # 実際の .env を読まない
    for name in ("OLLAMA_REASONING", "OLLAMA_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)
    pc.settings = cli._llm_settings("ollama", "qwen3:14b", reasoning)
    pc.chat = make_chat_model(pc.settings)


@when("思考モード有効で Ollama の LLM 設定を作る")
def _w_settings_think(pc: PromptCtx, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _make_settings(pc, True, monkeypatch, tmp_path)


@when("思考モード無効で Ollama の LLM 設定を作る")
def _w_settings_plain(pc: PromptCtx, monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    _make_settings(pc, False, monkeypatch, tmp_path)


def _settings(pc: PromptCtx) -> Settings:
    assert pc.settings is not None
    return pc.settings


@then("設定の思考モードは有効である")
def _t_setting_on(pc: PromptCtx) -> None:
    assert _settings(pc).ollama_reasoning is True


@then("設定の思考モードは無効である")
def _t_setting_off(pc: PromptCtx) -> None:
    assert _settings(pc).ollama_reasoning is False


@then(parsers.parse("設定のタイムアウトは {n:d} 秒である"))
def _t_timeout(pc: PromptCtx, n: int) -> None:
    assert _settings(pc).ollama_timeout_s == n


@then("設定のタイムアウトは なし である")
def _t_timeout_none(pc: PromptCtx) -> None:
    assert _settings(pc).ollama_timeout_s is None


@then(parsers.parse("生成されるチャットモデルの reasoning は {v} である"))
def _t_chat_reasoning(pc: PromptCtx, v: str) -> None:
    assert pc.chat.reasoning is (v == "True")  # type: ignore[attr-defined]


@then(parsers.parse("生成されるチャットモデルのタイムアウトは {n:d} 秒である"))
def _t_chat_timeout(pc: PromptCtx, n: int) -> None:
    assert pc.chat.client_kwargs["timeout"] == n  # type: ignore[attr-defined]


@then("生成されるチャットモデルのタイムアウトは なし である")
def _t_chat_timeout_none(pc: PromptCtx) -> None:
    assert "timeout" not in pc.chat.client_kwargs  # type: ignore[attr-defined]


@then("思考モード有効の LLM の識別情報に reasoning=true が含まれる")
def _t_info_think(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    info = cli._agent_info("llm", "ollama", "qwen3:14b", "hybrid", reasoning=True)
    assert info is not None
    assert info["reasoning"] == "true"


@then("思考モード無効の LLM の識別情報に reasoning のキーは含まれない")
def _t_info_plain(monkeypatch: pytest.MonkeyPatch, tmp_path: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    info = cli._agent_info("llm", "ollama", "qwen3:14b", "hybrid")
    assert info is not None
    assert "reasoning" not in info
