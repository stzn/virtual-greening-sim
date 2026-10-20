"""ベースライン(ルールベース)エージェントの振る舞い。"""

import copy

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import Ctx
from vmgs.agent.baseline import BaselineAgent
from vmgs.agent.runner import TurnResult

scenarios("agent/baseline.feature")


class BaselineCtx:
    def __init__(self) -> None:
        self.results: list[TurnResult] = []


@pytest.fixture
def bl() -> BaselineCtx:
    return BaselineCtx()


@given(parsers.parse("3時間以内に降水確率 {p:g}% の予報がある"))
def _g_rain_forecast(ctx: Ctx, p: float) -> None:
    ctx.weather.precipitation_prob_pct = float(p)


@when("ベースラインエージェントが1ターン判断する")
def _w_turn(ctx: Ctx, bl: BaselineCtx) -> None:
    ctx.before = ctx.snapshot()
    bl.results = [BaselineAgent(ctx.world).run_turn()]


@when("ベースラインエージェントが2回独立に判断する")
def _w_twice(ctx: Ctx, bl: BaselineCtx) -> None:
    twin = copy.deepcopy(ctx.world)  # まったく同じ状況をもう1つ作る
    bl.results = [BaselineAgent(ctx.world).run_turn(), BaselineAgent(twin).run_turn()]


def _irrigations(bl: BaselineCtx) -> list[int]:
    return [
        int(c.args["amount_ml"])
        for c in bl.results[0].tool_calls
        if c.name == "irrigate" and c.result["ok"]
    ]


@then("散水が1回行われる")
def _t_irrigated(bl: BaselineCtx) -> None:
    assert len(_irrigations(bl)) == 1


@then("散水は行われない")
def _t_not_irrigated(bl: BaselineCtx) -> None:
    assert _irrigations(bl) == []


@then(parsers.parse("散水量は {n:d}ml 以下である"))
def _t_amount(bl: BaselineCtx, n: int) -> None:
    amounts = _irrigations(bl)
    assert amounts
    assert sum(amounts) <= n


@then("2回の行動は同一である")
def _t_same(bl: BaselineCtx) -> None:
    a, b = bl.results
    assert [(c.name, c.args) for c in a.tool_calls] == [(c.name, c.args) for c in b.tool_calls]
    assert a.decision == b.decision
