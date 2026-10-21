"""チェックリストなしのプロンプト（free）。"""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs.agent.graph import parse_decision
from vmgs.agent.prompts import PROMPTS, SYSTEM_PROMPT, build_free_prompt
from vmgs.experiment.configs import parse_config_spec
from vmgs.world.params import Params

scenarios("agent/prompt_free.feature", "agent/prompt_hybrid.feature")


class PromptCtx:
    def __init__(self) -> None:
        self.text = ""
        self.params = Params()


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


@then(parsers.parse('構成の名前は "{name}" になる'))
def _t_spec_name(name: str) -> None:
    assert parse_config_spec("llm:ollama:qwen3:14b:free").name == name
