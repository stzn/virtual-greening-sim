"""条件観測（チェックリストの条件の真偽）。"""

import dataclasses
from datetime import timedelta
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import Ctx
from vmgs.tools.tools import ToolBox
from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import SeriesWeather, floor_hour

scenarios("world/condition_observation.feature")

OBS_KEYS = (
    "rain_within_3h",
    "heat_trigger",
    "moisture_band",
    "moisture_below_30",
    "moisture_below_40",
    "moisture_above_70",
    "battery_below_15",
    "in_sunlight",
    "temp_at_least_33",
    "moisture_50_to_70",
    "tank_at_least_4000",
    "rain_leaves_dry",
    "return_to_b_blocked",
)


class CondCtx:
    def __init__(self) -> None:
        self.result: dict[str, Any] = {}


@pytest.fixture
def cc() -> CondCtx:
    return CondCtx()


@given("条件観測が有効である")
def _g_enabled(ctx: Ctx) -> None:
    ctx.world.condition_observations_enabled = True


@given("条件観測は無効である")
def _g_disabled(ctx: Ctx) -> None:
    ctx.world.condition_observations_enabled = False


@given("行観測が有効である")
def _g_row_enabled(ctx: Ctx) -> None:
    ctx.world.row_observation_enabled = True


@given(parsers.parse("現在の気象は 気温 {t:g}℃・日射 {r:g}W/m² である"))
def _g_weather(ctx: Ctx, t: float, r: float) -> None:
    ctx.weather.temperature_c, ctx.weather.shortwave_w_m2 = float(t), float(r)


@given("次の予報が与えられている")
def _g_forecast(ctx: Ctx, datatable: list[list[str]]) -> None:
    head, *rows = datatable
    base = floor_hour(ctx.world.clock.now)
    over = {int(r[0]): (float(r[1]), float(r[2])) for r in rows}
    records = []
    for h in range(7):
        prob, mm = over.get(h, (0.0, 0.0))
        records.append(
            WeatherRecord(
                time=base + timedelta(hours=h),
                temperature_c=ctx.weather.temperature_c,
                humidity_pct=ctx.weather.humidity_pct,
                shortwave_w_m2=ctx.weather.shortwave_w_m2,
                wind_ms=ctx.weather.wind_ms,
                precipitation_mm=mm,
                precipitation_prob_pct=prob,
            )
        )
    assert head[0] == "何時間後"
    ctx.world.weather = SeriesWeather(records)


@given(parsers.parse("降水1mmあたりの水分増加は {v:g}% である"))
def _g_rain_gain(ctx: Ctx, v: float) -> None:
    ctx.world.params = dataclasses.replace(ctx.world.params, rain_gain_per_mm=float(v))


@when("get_plant_state を呼ぶ")
def _w_state(ctx: Ctx, cc: CondCtx) -> None:
    cc.result = ToolBox(ctx.world).call("get_plant_state")


@then("結果に 条件観測のキー は含まれない")
def _t_absent(cc: CondCtx) -> None:
    assert not any(k in cc.result for k in OBS_KEYS)


@then("結果に 条件観測のキー が含まれる")
def _t_present(cc: CondCtx) -> None:
    assert all(k in cc.result for k in OBS_KEYS)


@then("条件観測を除いた結果は、無効のときの結果と同じである")
def _t_same(ctx: Ctx, cc: CondCtx) -> None:
    rest = {k: v for k, v in cc.result.items() if k not in OBS_KEYS}
    ctx.world.condition_observations_enabled = False
    assert ToolBox(ctx.world).call("get_plant_state") == rest


@then(
    "条件観測のキーは rain_within_3h・heat_trigger・moisture_band・moisture_below_30・"
    "moisture_below_40・moisture_above_70・battery_below_15・in_sunlight・"
    "temp_at_least_33・moisture_50_to_70・tank_at_least_4000・"
    "rain_leaves_dry・return_to_b_blocked の13個だけである"
)
def _t_only_seven(ctx: Ctx, cc: CondCtx) -> None:
    ctx.world.condition_observations_enabled = False
    base = ToolBox(ctx.world).call("get_plant_state")
    assert set(cc.result) - set(base) == set(OBS_KEYS)


@then(parsers.parse("{key} は {value} である"))
def _t_bool(cc: CondCtx, key: str, value: str) -> None:
    if key == "moisture_band":
        assert cc.result[key] == value.strip('"')
    else:
        assert cc.result[key] is (value == "true")


@then("結果に checklist_row は含まれない")
def _t_row_absent(cc: CondCtx) -> None:
    assert "checklist_row" not in cc.result


@then(parsers.parse("最初に当てはまる行は {n:d} である"))
def _t_row(cc: CondCtx, n: int) -> None:
    assert cc.result["checklist_row"] == n


@then("行観測を除いた結果は、無効のときの結果と同じである")
def _t_row_same(ctx: Ctx, cc: CondCtx) -> None:
    rest = {k: v for k, v in cc.result.items() if k != "checklist_row"}
    ctx.world.row_observation_enabled = False
    assert ToolBox(ctx.world).call("get_plant_state") == rest
