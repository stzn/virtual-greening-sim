"""LLM判断品質の評価基盤。同じ状況で N 回判断させ、行動の述語を満たした回数で合否を決める。"""

import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from vmgs.agent.claims import claimed_actions
from vmgs.agent.prompts import SYSTEM_PROMPT, TurnSummary
from vmgs.agent.runner import AgentRunner, TurnResult
from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import SeriesWeather
from vmgs.world.clock import SimClock
from vmgs.world.plant import LastMove
from vmgs.world.world import World

EVAL_START = datetime(2025, 8, 1, 13, 0)
DEFAULT_TRIALS = 5
SIDE_EFFECT_TOOLS = {"move_to_spot", "irrigate"}

# 試行ごとの条件の揺らぎ。判断ルールの閾値（水分30/40/50/70/85・気温33・日射700・バッテリー15）は、
# 元の値と同じ側にとどめる（判断の正解を変えない）。その範囲で入力だけを変え、temperature=0 でも試行が同一にならないようにする
MOISTURE_JITTER = 5.0  # 土壌水分 ± [%]
TEMPERATURE_JITTER = 2.0  # 気温 ± [℃]
SHORTWAVE_JITTER_RATIO = 0.1  # 日射 ± 10%
WIND_JITTER = 1.5  # 風速 ± [m/s]
HUMIDITY_JITTER = 15.0  # 湿度 ± [%]
BATTERY_JITTER = 8.0  # バッテリー ± [%]
MOISTURE_THRESHOLDS = (30.0, 40.0, 50.0, 70.0, 85.0)
TEMPERATURE_THRESHOLDS = (33.0,)
SHORTWAVE_THRESHOLDS = (700.0,)
BATTERY_THRESHOLDS = (15.0,)
_EPS = 0.01


@dataclass
class ForecastPoint:
    hours_ahead: int
    precip_prob_pct: float
    precip_mm: float


@dataclass
class TrialSetup:
    spot: str = "A"
    moisture: float = 60.0
    tank_ml: int = 10_000
    battery_pct: float = 100.0
    temperature_c: float = 28.0
    shortwave_w_m2: float = 400.0
    wind_ms: float = 1.0
    humidity_pct: float = 50.0
    base_precip_prob_pct: float = 5.0
    forecast: list[ForecastPoint] = field(default_factory=list)
    history: list[TurnSummary] = field(default_factory=list)
    last_move: tuple[str, str, int] | None = None  # (移動元, 移動先, 何時間前)
    recent_moves_hours_ago: list[int] = field(default_factory=list)  # 移動予算の集計用


def build_weather(setup: TrialSetup, hours: int = 24) -> SeriesWeather:
    """現在（0時間後）から hours 時間分。forecast で指定した時間だけ降水を上書きする。"""
    overrides = {f.hours_ahead: f for f in setup.forecast}
    records: list[WeatherRecord] = []
    for h in range(hours + 1):
        point = overrides.get(h)
        records.append(
            WeatherRecord(
                time=EVAL_START + timedelta(hours=h),
                temperature_c=setup.temperature_c,
                humidity_pct=setup.humidity_pct,
                shortwave_w_m2=setup.shortwave_w_m2,
                wind_ms=setup.wind_ms,
                precipitation_mm=point.precip_mm if point else 0.0,
                precipitation_prob_pct=point.precip_prob_pct
                if point
                else setup.base_precip_prob_pct,
            )
        )
    return SeriesWeather(records)


def build_world(setup: TrialSetup) -> World:
    world = World(
        build_weather(setup),
        SimClock(EVAL_START),
        notes_path=Path(tempfile.mkdtemp()) / "notes.md",
        start_spot=setup.spot,
    )
    world.plant.moisture = setup.moisture
    world.plant.water_tank_ml = setup.tank_ml
    world.plant.battery_pct = setup.battery_pct
    world.plant.move_times = [EVAL_START - timedelta(hours=h) for h in setup.recent_moves_hours_ago]
    if setup.last_move is not None:
        src, dst, hours_ago = setup.last_move
        world.plant.last_move = LastMove(src, dst, EVAL_START - timedelta(hours=hours_ago))
    return world


def _jitter(
    rng: random.Random,
    base: float,
    amplitude: float,
    thresholds: tuple[float, ...] = (),
    lo: float = 0.0,
    hi: float = float("inf"),
) -> float:
    """base を ±amplitude で揺らす。閾値に対して base と同じ側（>= か <）にとどめ、[lo, hi] に収める。"""
    value = base + rng.uniform(-amplitude, amplitude)
    lower = max((t for t in thresholds if t <= base), default=lo)
    upper = min((t for t in thresholds if t > base), default=hi + _EPS) - _EPS
    return round(min(max(value, lower, lo), upper, hi), 2)


def vary_setup(setup: TrialSetup, seed: int, index: int) -> TrialSetup:
    """index=0 は条件そのまま。それ以外は (seed, index) から決定論的に、気象・水分・バッテリーを少し変える。"""
    if index == 0:
        return setup
    rng = random.Random(f"{seed}:{index}")
    return replace(
        setup,
        moisture=_jitter(rng, setup.moisture, MOISTURE_JITTER, MOISTURE_THRESHOLDS, 0.0, 100.0),
        temperature_c=_jitter(
            rng, setup.temperature_c, TEMPERATURE_JITTER, TEMPERATURE_THRESHOLDS, -50.0, 60.0
        ),
        shortwave_w_m2=_jitter(
            rng,
            setup.shortwave_w_m2,
            setup.shortwave_w_m2 * SHORTWAVE_JITTER_RATIO,
            SHORTWAVE_THRESHOLDS,
        ),
        wind_ms=_jitter(rng, setup.wind_ms, WIND_JITTER),
        humidity_pct=_jitter(rng, setup.humidity_pct, HUMIDITY_JITTER, lo=0.0, hi=100.0),
        battery_pct=_jitter(rng, setup.battery_pct, BATTERY_JITTER, BATTERY_THRESHOLDS, 0.0, 100.0),
    )


def run_trials(
    model: BaseChatModel,
    setup: TrialSetup,
    n: int = DEFAULT_TRIALS,
    system_prompt: str = SYSTEM_PROMPT,
    seed: int = 0,
) -> list[TurnResult]:
    """試行ごとに新しい World・Runner で1ターン判断させる（試行間の影響を排除）。条件は試行ごとに少し揺らす。"""
    results: list[TurnResult] = []
    for i in range(n):
        runner = AgentRunner(
            model, build_world(vary_setup(setup, seed, i)), system_prompt=system_prompt
        )
        runner.history = list(setup.history)
        results.append(runner.run_turn())
    return results


# ---- 述語（1試行の TurnResult → 振る舞いが期待どおりか） ----
Predicate = Callable[[TurnResult], bool]


def called(tool: str) -> Predicate:
    return lambda r: any(c.name == tool for c in r.tool_calls)


def not_called(tool: str) -> Predicate:
    return lambda r: not any(c.name == tool for c in r.tool_calls)


def moved_to(spot: str) -> Predicate:
    return lambda r: any(
        c.name == "move_to_spot" and c.args.get("spot_id") == spot and c.result["ok"]
        for c in r.tool_calls
    )


def not_moved_to(spot: str) -> Predicate:
    inner = moved_to(spot)
    return lambda r: not inner(r)


def not_attempted_move_to(spot: str) -> Predicate:
    """移動の成否によらず、そのスポットへの移動をそもそも試みなかった（LLM自身の判断を測る）。"""
    return lambda r: (
        not any(c.name == "move_to_spot" and c.args.get("spot_id") == spot for c in r.tool_calls)
    )


def any_of(*preds: Predicate) -> Predicate:
    return lambda r: any(p(r) for p in preds)


def checked_state_before_acting(r: TurnResult) -> bool:
    """get_plant_state が呼ばれており、最初の副作用ツールより前であること。"""
    names = [c.name for c in r.tool_calls]
    if "get_plant_state" not in names:
        return False
    first_state = names.index("get_plant_state")
    effects = [i for i, n in enumerate(names) if n in SIDE_EFFECT_TOOLS]
    return not effects or first_state < effects[0]


def irrigation_within_tank(tank_ml: int) -> Predicate:
    """散水の要求量が（初期の）タンク残量を超えない。複数回なら累計で判定する。"""

    def pred(r: TurnResult) -> bool:
        total = sum(int(c.args.get("amount_ml", 0)) for c in r.tool_calls if c.name == "irrigate")
        return total <= tank_ml

    return pred


def final_answer_is_json(r: TurnResult) -> bool:
    """最終回答がJSONとして解釈できた（本文のままのフォールバックではない）。"""
    return not r.decision.raw_fallback and not r.decision.failed


def summary_matches_actions(r: TurnResult) -> bool:
    """行動要約が「した」と述べた行動は、すべて成功したツール呼び出しとして実在する。"""
    executed = {c.name for c in r.tool_calls if c.result.get("ok")}
    return not (claimed_actions(r.decision.action_summary) - executed)


def first_time_consistent(r: TurnResult) -> bool:
    """再確認を促されることなく、要約が実際の行動と一致していた。"""
    return not r.decision.claim_retry and not r.decision.unexecuted_claim


def success_count(results: list[TurnResult], predicates: list[Predicate]) -> int:
    """すべての述語を満たした試行の数。LLM呼び出しが失敗した試行は（述語が空振りで通らないよう）除外する。"""
    return sum((not r.decision.failed) and all(p(r) for p in predicates) for r in results)
