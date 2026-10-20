"""実行結果（StepRecord列）から評価指標を算出する。"""

import itertools
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel

from vmgs.sim.records import StepRecord
from vmgs.weather.providers import WeatherProvider, WeatherUnavailableError

RAIN_HORIZON_HOURS = 3
RAIN_PROB_THRESHOLD = 70.0


class Metrics(BaseModel):
    steps: int
    final_health: float
    min_health: float
    heat_mitigation_score: float
    move_count: int
    wasted_moves: int
    irrigation_total_ml: int
    irrigation_before_rain: int
    final_battery: float
    final_tank_ml: int
    decision_failures: int
    tool_errors: int


def count_wasted_moves(moves: Sequence[tuple[str, str]]) -> int:
    """直前の移動の元のスポットへ戻る移動（往復）の回数。"""
    return sum(1 for prev, cur in itertools.pairwise(moves) if cur[1] == prev[0])


def count_irrigation_before_rain(
    times: Sequence[datetime],
    weather: WeatherProvider,
    *,
    horizon_hours: int = RAIN_HORIZON_HOURS,
    prob_threshold: float = RAIN_PROB_THRESHOLD,
) -> int:
    """散水の直後 horizon_hours 時間以内に、降水確率が閾値以上の時間があった散水の回数。"""
    count = 0
    for t in times:
        try:
            forecast = weather.forecast(t, horizon_hours)
        except WeatherUnavailableError:
            continue
        if any((f.precipitation_prob_pct or 0.0) >= prob_threshold for f in forecast):
            count += 1
    return count


def compute_metrics(steps: Sequence[StepRecord], weather: WeatherProvider | None = None) -> Metrics:
    if not steps:
        raise ValueError("ステップが無いため指標を算出できません")
    moves: list[tuple[str, str]] = []
    irrigation_times: list[datetime] = []
    irrigation_total = 0
    tool_errors = 0
    failures = 0
    for s in steps:
        failures += int(s.decision.failed)
        spot = s.state_before["spot_id"]
        for c in s.tool_calls:
            if not c.result.get("ok"):
                tool_errors += 1
                continue
            if c.name == "move_to_spot":
                new = c.result["new_spot_id"]
                moves.append((spot, new))
                spot = new
            elif c.name == "irrigate":
                irrigation_total += int(c.args["amount_ml"])
                irrigation_times.append(datetime.fromisoformat(s.time))
    last = steps[-1].state_after
    healths = [steps[0].state_before["health"], *(s.state_after["health"] for s in steps)]
    return Metrics(
        steps=len(steps),
        final_health=last["health"],
        min_health=min(healths),
        heat_mitigation_score=last["heat_mitigation_score"],
        move_count=len(moves),
        wasted_moves=count_wasted_moves(moves),
        irrigation_total_ml=irrigation_total,
        irrigation_before_rain=(
            count_irrigation_before_rain(irrigation_times, weather) if weather else 0
        ),
        final_battery=last["battery_pct"],
        final_tank_ml=last["water_tank_ml"],
        decision_failures=failures,
        tool_errors=tool_errors,
    )
