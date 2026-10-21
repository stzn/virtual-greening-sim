"""実行結果（StepRecord列）から評価指標を算出する。"""

import itertools
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel

from vmgs.sim.records import StepRecord
from vmgs.sim.store import RunStore
from vmgs.weather.providers import WeatherProvider, WeatherUnavailableError

RAIN_HORIZON_HOURS = 3
RAIN_PROB_THRESHOLD = 70.0
WASTED_MOVE_WINDOW_HOURS = 4  # 直前の移動の元へ、この時間以内に戻る移動を「無駄」とみなす
GUARD_ERROR_CODES = {"RECENT_REVERSAL", "MOVE_BUDGET_EXCEEDED"}

# 要約が「行動していない」ことを述べている表現（宣言とはみなさない）
_NEGATIONS = (
    "しない",
    "せず",
    "しなかった",
    "試み",
    "不要",
    "必要ない",
    "不可",
    "なし",
    "できな",
    "拒否",
)
_MOVE_WORDS = ("移動", "退避")
_IRRIGATE_WORDS = ("散水", "給水")


class MoveEvent(NamedTuple):
    time: datetime
    from_spot: str
    to_spot: str


class Metrics(BaseModel):
    steps: int
    final_health: float
    min_health: float
    heat_mitigation_score: float
    move_count: int
    wasted_moves: int
    rejected_moves: int  # 拒否された move_to_spot の試行（理由を問わない）
    guard_rejections: int  # うち、往復・移動予算のガードによる拒否
    irrigation_total_ml: int
    irrigation_before_rain: int
    final_battery: float
    final_tank_ml: int
    decision_failures: int
    malformed_summaries: int  # 最終回答がJSONにならず崩れた要約
    unexecuted_claims: int  # 要約は移動/散水したと言うが、成功したツール呼び出しが無い
    tool_errors: int


def count_wasted_moves(
    moves: Sequence[MoveEvent], window_hours: int = WASTED_MOVE_WINDOW_HOURS
) -> int:
    """直前の移動の元のスポットへ window_hours 時間以内に戻る移動（往復）の回数。"""
    return sum(
        1
        for prev, cur in itertools.pairwise(moves)
        if cur.to_spot == prev.from_spot
        and (cur.time - prev.time).total_seconds() <= window_hours * 3600
    )


def claimed_actions(summary: str) -> set[str]:
    """行動要約が「実行した」と述べているツール名。否定・試行の表現は宣言とみなさない。"""
    if any(n in summary for n in _NEGATIONS):
        return set()
    claims: set[str] = set()
    if any(w in summary for w in _MOVE_WORDS):
        claims.add("move_to_spot")
    if any(w in summary for w in _IRRIGATE_WORDS):
        claims.add("irrigate")
    return claims


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
    moves: list[MoveEvent] = []
    irrigation_times: list[datetime] = []
    irrigation_total = 0
    tool_errors = rejected = guard = 0
    failures = malformed = unexecuted = 0
    for s in steps:
        failures += int(s.decision.failed)
        malformed += int(s.decision.raw_fallback)
        at = datetime.fromisoformat(s.time)
        spot = s.state_before["spot_id"]
        executed: set[str] = set()
        for c in s.tool_calls:
            if not c.result.get("ok"):
                tool_errors += 1
                if c.name == "move_to_spot":
                    rejected += 1
                    guard += int(c.result["error"]["code"] in GUARD_ERROR_CODES)
                continue
            executed.add(c.name)
            if c.name == "move_to_spot":
                new = c.result["new_spot_id"]
                moves.append(MoveEvent(at, spot, new))
                spot = new
            elif c.name == "irrigate":
                irrigation_total += int(c.args["amount_ml"])
                irrigation_times.append(at)
        # 崩れた要約は本文そのものなので、宣言のみの判定からは除く
        if not (s.decision.failed or s.decision.raw_fallback):
            unexecuted += int(bool(claimed_actions(s.decision.action_summary) - executed))
    last = steps[-1].state_after
    healths = [steps[0].state_before["health"], *(s.state_after["health"] for s in steps)]
    return Metrics(
        steps=len(steps),
        final_health=last["health"],
        min_health=min(healths),
        heat_mitigation_score=last["heat_mitigation_score"],
        move_count=len(moves),
        wasted_moves=count_wasted_moves(moves),
        rejected_moves=rejected,
        guard_rejections=guard,
        irrigation_total_ml=irrigation_total,
        irrigation_before_rain=(
            count_irrigation_before_rain(irrigation_times, weather) if weather else 0
        ),
        final_battery=last["battery_pct"],
        final_tank_ml=last["water_tank_ml"],
        decision_failures=failures,
        malformed_summaries=malformed,
        unexecuted_claims=unexecuted,
        tool_errors=tool_errors,
    )


def metrics_from_run(run_dir: Path, weather: WeatherProvider | None = None) -> Metrics:
    """保存済みの実行（steps.jsonl）から指標を再計算する。指標の定義を変えた後の再評価に使う。"""
    return compute_metrics(RunStore(run_dir).read_steps(), weather)
