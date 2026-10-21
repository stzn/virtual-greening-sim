"""全featureで共有するステップ定義とフィクスチャ。"""

import re
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, then, when

from vmgs.tools.tools import ToolBox
from vmgs.weather.providers import ConstantWeather
from vmgs.world.clock import SimClock
from vmgs.world.plant import LastMove, UnitStatus
from vmgs.world.spots import SPOTS
from vmgs.world.world import World, WorldError

START = datetime(2025, 8, 1, 13, 0)

# 日本語ラベル → ツール結果のキー
KEY_MAP = {
    "気温": "temperature_c",
    "日射": "shortwave_w_m2",
    "風速": "wind_ms",
    "湿度": "humidity_pct",
    "降水": "precipitation_mm",
    "土壌水分": "soil_moisture_pct",
    "散水後の土壌水分": "soil_moisture_pct",
    "健康度": "health",
    "現在スポット": "spot_id",
    "水タンク残量": "water_tank_ml",
    "バッテリー残量": "battery_pct",
    "ステータス": "status",
    "冷却効果": "cooling_c",
    "直近の移動回数": "moves_in_window",
    "残りの移動予算": "move_budget_remaining",
}


class Ctx:
    """1シナリオ分の共有状態。"""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.weather = ConstantWeather()
        self.world = self.new_world()
        self.toolbox = ToolBox(self.world)
        self.before: dict[str, Any] = {}
        self.error: WorldError | None = None
        self.result: dict[str, Any] = {}
        self.results: list[dict[str, Any]] = []
        self.value: float = 0.0
        self.compare: dict[str, float] = {}
        self.irrigated_ml = 0

    def new_world(self, spot: str = "A") -> World:
        return World(
            self.weather,
            SimClock(START),
            notes_path=self.tmp_path / "notes.md",
            start_spot=spot,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.world.state().model_dump()

    def act(self, fn: Callable[[], object]) -> None:
        """行動前の状態を保存し、WorldError は捕捉して ctx.error に残す。"""
        self.before = self.snapshot()
        self.error = None
        try:
            fn()
        except WorldError as e:
            self.error = e

    def call_tool(self, name: str, **kwargs: Any) -> None:
        self.before = self.snapshot()
        self.result = self.toolbox.call(name, **kwargs)


@pytest.fixture
def ctx(tmp_path: Path) -> Ctx:
    return Ctx(tmp_path)


def _num(v: str) -> float:
    return float(v)


# ---------------- Given ----------------
@given(parsers.parse('植物ユニットがスポット "{spot}" にいる'))
def _at_spot(ctx: Ctx, spot: str) -> None:
    ctx.world.plant.spot_id = spot


@given(parsers.parse("仮想時刻が {hh}:{mm} の World が用意されている"))
def _clock(ctx: Ctx, hh: str, mm: str) -> None:
    ctx.world.clock.now = ctx.world.clock.now.replace(hour=int(hh), minute=int(mm))


@given(parsers.parse("土壌水分は {v:g}% である"))
def _g_moisture(ctx: Ctx, v: float) -> None:
    ctx.world.plant.moisture = float(v)


@given(parsers.parse("健康度は {v:g} である"))
def _g_health(ctx: Ctx, v: float) -> None:
    ctx.world.plant.health = float(v)


@given(parsers.re(r"水タンク残量[はが] (?P<n>\d+)ml である"), converters={"n": int})
def _g_tank(ctx: Ctx, n: int) -> None:
    ctx.world.plant.water_tank_ml = n


@given(parsers.parse("バッテリー残量は {v:g}% である"))
def _g_battery(ctx: Ctx, v: float) -> None:
    ctx.world.plant.battery_pct = float(v)


@given(parsers.parse('{n:d}時間前に "{a}" から "{b}" へ移動している'))
def _g_last_move(ctx: Ctx, n: int, a: str, b: str) -> None:
    now = ctx.world.clock.now
    when = now - timedelta(hours=n)
    ctx.world.plant.last_move = LastMove(a, b, when)
    ctx.world.plant.move_times.append(when)


@given(parsers.parse("{a:d}時間前と {b:d}時間前に移動している"))
def _g_move_times(ctx: Ctx, a: int, b: int) -> None:
    now = ctx.world.clock.now
    ctx.world.plant.move_times = [now - timedelta(hours=a), now - timedelta(hours=b)]


@given("直近の移動は無い")
def _g_no_recent_moves(ctx: Ctx) -> None:
    ctx.world.plant.move_times = []
    ctx.world.plant.last_move = None


@given("直前の移動の記録が無い")
def _g_no_last_move(ctx: Ctx) -> None:
    ctx.world.plant.last_move = None


@given("ユニットが枯死している")
def _g_dead(ctx: Ctx) -> None:
    ctx.world.plant.status = UnitStatus.DEAD


@given(
    parsers.re(
        r"気温 (?P<t>-?[\d.]+)℃・日射 (?P<r>[\d.]+)W/m²"
        r"(?:・風速 (?P<u>[\d.]+)m/s・湿度 (?P<h>[\d.]+)%・降水 (?P<p>[\d.]+)mm)?"
        r" の1時間が与えられる"
    ),
)
def _g_weather(ctx: Ctx, t: str, r: str, u: str | None, h: str | None, p: str | None) -> None:
    w = ctx.weather
    w.temperature_c, w.shortwave_w_m2 = float(t), float(r)
    # 風速・湿度・降水は省略可（省略時は ConstantWeather の既定値のまま）
    if u is not None and h is not None and p is not None:
        w.wind_ms, w.humidity_pct, w.precipitation_mm = float(u), float(h), float(p)


# ---------------- When（world） ----------------
@when(parsers.parse("{n:d}時間経過する"))
def _w_advance(ctx: Ctx, n: int) -> None:
    ctx.act(lambda: ctx.world.advance(n))


@when(parsers.parse("仮想時刻を 1時間進める"))
def _w_advance_clock(ctx: Ctx) -> None:
    ctx.act(lambda: ctx.world.advance(1))


@when(parsers.parse("{n:d}ml 散水する"))
def _w_irrigate(ctx: Ctx, n: int) -> None:
    ctx.irrigated_ml = n
    ctx.act(lambda: ctx.world.irrigate(n))


@when(parsers.parse('スポット "{spot}" へ移動する'))
def _w_move(ctx: Ctx, spot: str) -> None:
    ctx.act(lambda: ctx.world.move_to(spot))


@when(parsers.parse('スポット "{a}" と "{b}" で土壌水分 {m:g}% から1時間経過させて比較する'))
def _w_compare(ctx: Ctx, a: str, b: str, m: float) -> None:
    for spot in (a, b):
        world = ctx.new_world(spot)
        world.plant.moisture = float(m)
        world.advance(1)
        ctx.compare[spot] = float(m) - world.plant.moisture


@when(parsers.parse('スポット "{spot}" の有効気温を求める'))
def _w_efftemp(ctx: Ctx, spot: str) -> None:
    w = ctx.weather.at(START)
    ctx.value = SPOTS[spot].effective_temp(w.temperature_c, w.shortwave_w_m2, ctx.world.params)


# ---------------- Then（world） ----------------
@then(parsers.parse('スポット "{b}" の水分減少量はスポット "{a}" より小さい'))
def _t_compare(ctx: Ctx, a: str, b: str) -> None:
    assert ctx.compare[b] < ctx.compare[a]


@then(parsers.parse("有効気温は {v:g}℃ である"))
def _t_efftemp(ctx: Ctx, v: float) -> None:
    assert ctx.value == pytest.approx(v)


@then(parsers.parse("土壌水分は {v:g}% より小さい"))
def _t_moisture_lt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.moisture < v


@then(parsers.parse("土壌水分は {v:g}% より大きい"))
def _t_moisture_gt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.moisture > v


@then(parsers.parse("土壌水分は {v:g}% である"))
def _t_moisture_eq(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.moisture == pytest.approx(v, abs=0.01)


@then(parsers.parse("健康度は {v:g} より小さい"))
def _t_health_lt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.health < v


@then(parsers.parse("健康度は {v:g} より大きい"))
def _t_health_gt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.health > v


@then(parsers.parse('ユニットの状態は "{s}" である'))
def _t_status(ctx: Ctx, s: str) -> None:
    assert ctx.world.plant.status.value == s


@then(parsers.re(r"水タンク残量[はが] (?P<n>\d+)ml である"), converters={"n": int})
def _t_tank(ctx: Ctx, n: int) -> None:
    assert ctx.world.plant.water_tank_ml == n


@then(parsers.parse("バッテリー残量は {v:g}% である"))
def _t_battery_eq(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.battery_pct == pytest.approx(v, abs=0.01)


@then(parsers.parse("移動できるようになるまで {n:d} 時間である"))
def _t_budget_retry(ctx: Ctx, n: int) -> None:
    assert ctx.error is not None
    assert ctx.error.details["retry_after_hours"] == n


@then(parsers.parse("直近の移動回数は {n:d} 回である"))
def _t_moves_in_window(ctx: Ctx, n: int) -> None:
    assert ctx.world.state().moves_in_window == n


@then(parsers.parse("残りの移動予算は {n:d} 回である"))
def _t_budget_remaining(ctx: Ctx, n: int) -> None:
    assert ctx.world.state().move_budget_remaining == n


@then("エラーに あと何時間後から戻れるか が含まれる")
def _t_retry_after(ctx: Ctx) -> None:
    assert ctx.error is not None
    retry = ctx.error.details["retry_after_hours"]
    assert isinstance(retry, int)
    assert retry >= 1


@then(parsers.parse("バッテリー残量は {v:g}% より大きい"))
def _t_battery_gt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.battery_pct > v


@then(parsers.parse("バッテリー残量は {v:g}% より小さい"))
def _t_battery_lt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.battery_pct < v


@then(parsers.parse('現在のスポットは "{spot}" である'))
@then(parsers.parse('現在のスポットは "{spot}" のままである'))
def _t_spot(ctx: Ctx, spot: str) -> None:
    assert ctx.world.plant.spot_id == spot


@then(parsers.parse("バッテリー消費は {lo:g}% 以上 {hi:g}% 以下である"))
def _t_battery_used(ctx: Ctx, lo: float, hi: float) -> None:
    used = ctx.before["battery_pct"] - ctx.world.plant.battery_pct
    assert lo - 1e-9 <= used <= hi + 1e-9


@then(parsers.parse('散水はエラー "{code}" で拒否される'))
@then(parsers.parse('移動はエラー "{code}" で拒否される'))
def _t_world_error(ctx: Ctx, code: str) -> None:
    assert ctx.error is not None, "エラーが発生していない"
    assert ctx.error.code == code


@then("土壌水分とタンク残量は変化しない")
def _t_unchanged_world(ctx: Ctx) -> None:
    now = ctx.snapshot()
    assert now["soil_moisture_pct"] == pytest.approx(ctx.before["soil_moisture_pct"])
    assert now["water_tank_ml"] == ctx.before["water_tank_ml"]


@then(parsers.parse("暑熱緩和スコアは {v:g} より大きい"))
def _t_score_gt(ctx: Ctx, v: float) -> None:
    assert ctx.world.plant.heat_mitigation_score > v


@then("猛暑のスポットCでの同量の散水より暑熱緩和スコアが小さい")
def _t_score_cmp(ctx: Ctx) -> None:
    other = ctx.new_world("C")
    other.irrigate(ctx.irrigated_ml)
    assert ctx.world.plant.heat_mitigation_score < other.plant.heat_mitigation_score


# ---------------- 共通ヘルパ（tools用） ----------------
def keys_of(names: str) -> list[str]:
    cleaned = re.sub(r"^現在の\s*", "", names.strip())
    return [KEY_MAP[n.strip()] for n in cleaned.split("・")]


def find_key(result: dict[str, Any], key: str) -> bool:
    return key in result or key in result.get("current", {})


# 24時間シミュレーション用のステップ定義と fixture（sc）を取り込む
from tests.step_defs.experiment_steps import *
from tests.step_defs.sim_steps import *
