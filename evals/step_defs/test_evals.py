"""LLM判断品質の評価シナリオ。実行: uv run pytest evals -m eval

実LLMを複数回呼ぶため、既定の `pytest` からは除外されている。
"""

import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, TurnSummary
from vmgs.agent.runner import TurnResult
from vmgs.eval.trials import (
    DEFAULT_TRIALS,
    ForecastPoint,
    Predicate,
    TrialSetup,
    any_of,
    called,
    checked_state_before_acting,
    final_answer_is_json,
    first_time_consistent,
    irrigation_within_tank,
    moved_to,
    not_attempted_move_to,
    not_called,
    not_moved_to,
    run_trials,
    success_count,
    summary_matches_actions,
)

pytestmark = pytest.mark.eval

FEATURES = Path(__file__).resolve().parent.parent / "features"
for _name in ("rain", "heat", "resources", "loop_guard", "answer_format", "hard"):
    scenarios(str(FEATURES / f"{_name}.feature"))


class EvalCtx:
    def __init__(self) -> None:
        self.setup = TrialSetup()
        self.predicates: list[Predicate] = []
        self.results: list[TurnResult] = []


@pytest.fixture
def ec() -> EvalCtx:
    return EvalCtx()


def _rows(datatable: list[list[str]]) -> list[dict[str, str]]:
    header, *rows = datatable
    return [dict(zip(header, r, strict=True)) for r in rows]


# ---------------- Given ----------------
@given(
    parsers.re(
        r'スポット "(?P<spot>[ABC])"・土壌水分 (?P<m>[\d.]+)%・水タンク (?P<t>\d+)ml'
        r"(?:・バッテリー (?P<b>[\d.]+)%)? の状況である"
    )
)
def _g_situation(ec: EvalCtx, spot: str, m: str, t: str, b: str | None) -> None:
    ec.setup.spot, ec.setup.moisture, ec.setup.tank_ml = spot, float(m), int(t)
    if b is not None:
        ec.setup.battery_pct = float(b)


@given("次の気象予報が与えられている")
def _g_forecast(ec: EvalCtx, datatable: list[list[str]]) -> None:
    ec.setup.forecast = [
        ForecastPoint(int(r["何時間後"]), float(r["降水確率"]), float(r["降水量"]))
        for r in _rows(datatable)
    ]


@given(parsers.parse("今後6時間は降水確率 {p:g}% の晴天である"))
def _g_clear(ec: EvalCtx, p: float) -> None:
    ec.setup.base_precip_prob_pct = float(p)


@given(parsers.parse("現在の気象は 気温 {t:g}℃・日射 {r:g}W/m² である"))
def _g_weather(ec: EvalCtx, t: float, r: float) -> None:
    ec.setup.temperature_c, ec.setup.shortwave_w_m2 = float(t), float(r)


@given(parsers.parse('{n:d}時間前に "{a}" から "{b}" へ移動している'))
def _g_last_move(ec: EvalCtx, n: int, a: str, b: str) -> None:
    ec.setup.last_move = (a, b, n)


@given(parsers.parse("直近の移動が {a:d}時間前と {b:d}時間前にある"))
def _g_recent_moves(ec: EvalCtx, a: int, b: int) -> None:
    ec.setup.recent_moves_hours_ago = [a, b]


@given("前のターンの判断要約が次のとおり1件ある")
def _g_history(ec: EvalCtx, datatable: list[list[str]]) -> None:
    ec.setup.history = [
        TurnSummary(time=r["時刻"], action_summary=r["行動要約"]) for r in _rows(datatable)
    ]


# ---------------- When ----------------
@when("エージェントに1ターン判断させる")
def _w_run(ec: EvalCtx, eval_model: BaseChatModel) -> None:
    version = os.environ.get("EVAL_PROMPT", PROMPT_VERSION)
    ec.results = run_trials(eval_model, ec.setup, DEFAULT_TRIALS, PROMPTS[version])


# ---------------- Then（述語を登録し、最後の「N回中M回」でまとめて判定） ----------------
@then("irrigate は呼ばれない")
def _t_no_irrigate(ec: EvalCtx) -> None:
    ec.predicates.append(not_called("irrigate"))


@then("irrigate が呼ばれる")
def _t_irrigate(ec: EvalCtx) -> None:
    ec.predicates.append(called("irrigate"))


@then("move_to_spot は呼ばれない")
def _t_no_move(ec: EvalCtx) -> None:
    ec.predicates.append(not_called("move_to_spot"))


@then(parsers.parse('move_to_spot で "{spot}" へ移動しない'))
def _t_not_moved(ec: EvalCtx, spot: str) -> None:
    ec.predicates.append(not_moved_to(spot))


@then(parsers.parse('move_to_spot で "{spot}" へ移動する'))
def _t_moved(ec: EvalCtx, spot: str) -> None:
    ec.predicates.append(moved_to(spot))


@then(parsers.parse('move_to_spot で "{spot}" へ移動するか irrigate が呼ばれる'))
def _t_move_or_irrigate(ec: EvalCtx, spot: str) -> None:
    ec.predicates.append(any_of(moved_to(spot), called("irrigate")))


@then(parsers.parse('move_to_spot で "{spot}" への移動を試みない'))
def _t_not_attempted(ec: EvalCtx, spot: str) -> None:
    ec.predicates.append(not_attempted_move_to(spot))


@then("最終回答はJSONで解釈できる")
def _t_json(ec: EvalCtx) -> None:
    ec.predicates.append(final_answer_is_json)


@then("行動要約は実際のツール呼び出しと矛盾しない")
def _t_consistent(ec: EvalCtx) -> None:
    ec.predicates.append(summary_matches_actions)


@then("再確認なしで要約が実際の行動と一致する")
def _t_first_time(ec: EvalCtx) -> None:
    ec.predicates.append(first_time_consistent)


@then("副作用のあるツールの前に get_plant_state が呼ばれている")
def _t_state_first(ec: EvalCtx) -> None:
    ec.predicates.append(checked_state_before_acting)


@then("散水量は水タンク残量を超えない")
def _t_within_tank(ec: EvalCtx) -> None:
    ec.predicates.append(irrigation_within_tank(ec.setup.tank_ml))


def _trace(r: TurnResult) -> dict[str, Any]:
    return {
        "calls": [{"name": c.name, "args": c.args, "ok": c.result["ok"]} for c in r.tool_calls],
        "action_summary": r.decision.action_summary,
        "failed": r.decision.failed,
        "error": r.decision.error,
    }


@then(parsers.parse("{n:d}回中{m:d}回以上この振る舞いになる"))
def _t_threshold(
    ec: EvalCtx, request: pytest.FixtureRequest, scoreboard: list[dict[str, Any]], n: int, m: int
) -> None:
    assert n == DEFAULT_TRIALS, f"試行回数は {DEFAULT_TRIALS} 回です（featureは {n} 回）"
    ok = success_count(ec.results, ec.predicates)
    scoreboard.append(
        {
            "name": request.node.name,
            "successes": ok,
            "trials": n,
            "threshold": m,
            "passed": ok >= m,
            "traces": [_trace(r) for r in ec.results],
        }
    )
    assert ok >= m, f"{n}回中{ok}回しか期待どおりになりませんでした（基準 {m} 回）"
