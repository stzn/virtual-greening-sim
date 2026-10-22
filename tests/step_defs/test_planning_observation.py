"""計画観測（残り時間と、残り時間あたりに使える水・電池）。"""

from datetime import datetime, timedelta
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import START, Ctx
from tests.step_defs.sim_steps import SimCtx
from vmgs.tools.tools import ToolBox

scenarios("world/planning_observation.feature", "sim/planning_observation_sim.feature")

KEYS = {
    "残り時間": "hours_remaining",
    "1時間あたりの水": "water_per_remaining_hour_ml",
    "1時間あたりの電池": "battery_per_remaining_hour_pct",
}
OBS_KEYS = tuple(KEYS.values())


class PlanCtx:
    def __init__(self) -> None:
        self.result: dict[str, Any] = {}


@pytest.fixture
def pl() -> PlanCtx:
    return PlanCtx()


# ---------------- world ----------------
def _end_of_day() -> datetime:
    return (START + timedelta(days=1)).replace(hour=0, minute=0)


@given("終了時刻が 24:00 で、計画観測が有効である")
def _g_enabled(ctx: Ctx) -> None:
    ctx.world.horizon_end = _end_of_day()
    ctx.world.planning_observations_enabled = True


@given("終了時刻が 24:00 で、計画観測は無効である")
def _g_disabled(ctx: Ctx) -> None:
    ctx.world.horizon_end = _end_of_day()
    ctx.world.planning_observations_enabled = False


@given("終了時刻は不明で、計画観測が有効である")
def _g_unknown_end(ctx: Ctx) -> None:
    ctx.world.horizon_end = None
    ctx.world.planning_observations_enabled = True


@when("get_plant_state を呼ぶ")
def _w_state(ctx: Ctx, pl: PlanCtx) -> None:
    pl.result = ToolBox(ctx.world).call("get_plant_state")


@then(parsers.parse("結果に {names} は含まれない"))
def _t_absent(pl: PlanCtx, names: str) -> None:
    for n in names.split("・"):
        assert KEYS[n.strip()] not in pl.result


@then(parsers.parse("残り時間は {n:d} である"))
def _t_remaining(pl: PlanCtx, n: int) -> None:
    assert pl.result["hours_remaining"] == n


@then(parsers.parse("1時間あたりの水は {v:g}ml である"))
def _t_water(pl: PlanCtx, v: float) -> None:
    assert pl.result["water_per_remaining_hour_ml"] == pytest.approx(v)


@then(parsers.parse("1時間あたりの電池は {v:g}% である"))
def _t_battery(pl: PlanCtx, v: float) -> None:
    assert pl.result["battery_per_remaining_hour_pct"] == pytest.approx(v)


@then("観測の3つの値はすべて null である")
def _t_nulls(pl: PlanCtx) -> None:
    assert all(k in pl.result and pl.result[k] is None for k in OBS_KEYS)


@then("観測を除いた結果は、計画観測が無効のときの結果と同じである")
def _t_same_without_obs(ctx: Ctx, pl: PlanCtx) -> None:
    with_obs = {k: v for k, v in pl.result.items() if k not in OBS_KEYS}
    ctx.world.planning_observations_enabled = False
    assert ToolBox(ctx.world).call("get_plant_state") == with_obs


# ---------------- sim ----------------
@given("計画観測を有効にして実行する")
def _g_sim_enabled(sc: SimCtx) -> None:
    sc.overrides["planning_observations"] = True


def _state_result(sc: SimCtx, hour: int) -> dict[str, Any]:
    step = sc.last.steps[hour - 1]
    call = next(c for c in step.tool_calls if c.name == "get_plant_state")
    return call.result


@then(parsers.parse("{h:d}時間目の get_plant_state の残り時間は {n:d} である"))
def _t_step_remaining(sc: SimCtx, h: int, n: int) -> None:
    assert _state_result(sc, h)["hours_remaining"] == n


@then("全ステップで 1時間あたりの水は 水タンク残量÷残り時間 である")
def _t_all_division(sc: SimCtx) -> None:
    assert len(sc.last.steps) == 24
    for h in range(1, 25):
        r = _state_result(sc, h)
        assert r["water_per_remaining_hour_ml"] == pytest.approx(
            r["water_tank_ml"] / r["hours_remaining"], abs=0.06
        )


@then("どのステップのツール結果にも計画観測のキーは含まれない")
def _t_no_keys(sc: SimCtx) -> None:
    for step in sc.last.steps:
        for c in step.tool_calls:
            assert not any(k in c.result for k in OBS_KEYS)
