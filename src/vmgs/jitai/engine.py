"""4 段階を時刻順に回し、介入の記録と集計を作る。

判定点: 過緊張を検知した窓のうち、前の判定から十分に時間がたったもの。
介入した後は COOLDOWN_AFTER_INTERVENTION_MIN、見送った後は RETRY_AFTER_SKIP_MIN の間は判定しない
（通知の負担を抑え、見送った人は空きができたら拾い直す）。全員を同じ時刻で進め、席の予約は早い者勝ち。
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from vmgs.analysis.recording import FloatArray
from vmgs.analysis.synthetic import TrueParams
from vmgs.contracts import Dispatch
from vmgs.fleet.stage_world import StageWorld
from vmgs.jitai.calendar import MIN_FREE_MIN, free_minutes
from vmgs.jitai.detect import (
    SUSTAIN_WINDOWS,
    Z_THRESHOLD,
    detection_scores,
    flags_from_z,
    worker_profile,
)
from vmgs.jitai.pods import DWELL_MIN, Occupancy, PodSchedule, walk_minutes
from vmgs.jitai.recovery import CARRYOVER_MIN, recovery_amplitude, recovery_curve
from vmgs.jitai.safety import PetFn, pet_check, rain_check, rain_limit
from vmgs.jitai.workers import STEP_MIN, Worker, time_grid_min

COOLDOWN_AFTER_INTERVENTION_MIN = 90
RETRY_AFTER_SKIP_MIN = 15
DEMAND_MIN_WORKERS = 3  # 需要の集計で、これ未満の人数の区画・時刻は出さない（個人の特定を防ぐ）

Outcome = Literal["intervened", "skipped"]
# "pet" は天候の段階（雨・PET）。契約（jitai-summary.json の skipped_pet）のキーに合わせて名前は変えない
Stage = Literal["calendar", "pods", "pet"]
STAGES: tuple[Stage, ...] = ("calendar", "pods", "pet")
REASONS: dict[str, Stage] = {
    "in_meeting": "calendar",
    "short_gap": "calendar",
    "no_pod_nearby": "pods",
    "pod_leaving": "pods",
    "pod_full": "pods",
    "pet_destination": "pet",
    "pet_route": "pet",
    "rain": "pet",
}


@dataclass(frozen=True)
class Record:
    """1 回の判定。個人の計測値（ln(HF) など）は持たない。"""

    time_min: int
    worker_id: str
    outcome: Outcome
    stage: Stage | None  # 見送った段階（介入したら None）
    reason: str | None
    pod_id: str | None
    spot_id: str | None
    walk_m: float | None
    pet_destination_c: float | None
    pet_route_c: float | None

    def to_json(self) -> dict[str, Any]:
        h, m = divmod(self.time_min, 60)
        return {
            "time": f"{h:02d}:{m:02d}",
            "worker_id": self.worker_id,
            "outcome": self.outcome,
            "stage": self.stage,
            "reason": self.reason,
            "pod_id": self.pod_id,
            "spot_id": self.spot_id,
            "walk_m": self.walk_m,
            "pet_destination_c": self.pet_destination_c,
            "pet_route_c": self.pet_route_c,
        }


class TimedDecision(Protocol):
    """需要の集計に要る、判定の時刻とワーカーだけを持つもの（Record・DecisionPoint）。"""

    @property
    def time_min(self) -> int: ...

    @property
    def worker_id(self) -> str: ...


@dataclass(frozen=True)
class DecisionPoint:
    """介入しない場合の判定点（需要の元）。"""

    time_min: int
    worker_id: str


@dataclass(frozen=True)
class JitaiResult:
    records: list[Record]
    workers: list[Worker]
    corrected: bool
    recovery: bool = False
    carryover_min: float = CARRYOVER_MIN
    hyperstress_windows: int = 0  # 回復を反映した後に過緊張と判定した窓の数（全員の合計）
    episode_burden: float = 0.0  # 真のエピソード中の「下げ幅 − 回復」の正の部分 × 分（全員の合計）


def _skip(t: int, w: Worker, reason: str, **kw: Any) -> Record:
    fields: dict[str, Any] = {
        "pod_id": None,
        "spot_id": None,
        "walk_m": None,
        "pet_destination_c": None,
        "pet_route_c": None,
    }
    fields.update(kw)
    return Record(t, w.id, "skipped", REASONS[reason], reason, **fields)


def _decide(
    t: int,
    w: Worker,
    world: StageWorld,
    schedule: PodSchedule,
    occupancy: Occupancy,
    pet: PetFn,
    rain_mm: Mapping[int, float] | None = None,
    rain_cover: bool = False,
) -> Record:
    # 段階 2: 予定の空き枠
    free = free_minutes(w.meetings, t)
    if free == 0:
        return _skip(t, w, "in_meeting")
    if free < MIN_FREE_MIN:
        return _skip(t, w, "short_gap")
    # 段階 3: 徒歩圏のポッドの空き（滞在の終わりまで同じ地点にいて、席が空いている）
    near = schedule.nearby(world, w.lat, w.lon, t // 60)
    if not near:
        return _skip(t, w, "no_pod_nearby")
    plans = []
    for c in near:
        arrive = t + walk_minutes(c.walk_m)
        leave = arrive + DWELL_MIN
        plans.append((c, arrive, leave, leave + walk_minutes(c.walk_m)))
    staying = [p for p in plans if schedule.stays(p[0].pod_id, p[0].spot_id, p[1], p[2])]
    if not staying:
        return _skip(t, w, "pod_leaving")
    seated = [p for p in staying if occupancy.free(p[0].pod_id, p[1], p[2])]
    if not seated:
        return _skip(t, w, "pod_full")
    # 段階 4: 天候の安全判定（雨 → 二段階の PET。近いポッドから順に、通るものを選ぶ）
    first: Record | None = None
    for c, arrive, leave, back in seated:
        hours = range(t // 60, (back - 1) // 60 + 1)
        pet_reason, dest, route = pet_check(pet, c.spot_id, {w.home_spot_id, c.spot_id}, hours)
        rainy = rain_mm is not None and rain_check(rain_mm, hours) >= rain_limit(
            rain_cover=rain_cover
        )
        reason: str | None = "rain" if rainy else pet_reason
        if reason is None:
            occupancy.book(c.pod_id, arrive, leave)
            return Record(
                t, w.id, "intervened", None, None, c.pod_id, c.spot_id, c.walk_m, dest, route
            )
        if first is None:
            first = _skip(
                t,
                w,
                reason,
                pod_id=c.pod_id,
                spot_id=c.spot_id,
                walk_m=c.walk_m,
                pet_destination_c=dest,
                pet_route_c=route,
            )
    assert first is not None
    return first


def run_jitai(
    workers: Sequence[Worker],
    world: StageWorld,
    dispatch: Dispatch,
    pet: PetFn,
    *,
    corrected: bool = True,
    recovery: bool = True,
    carryover_min: float = CARRYOVER_MIN,
    rain_mm: Mapping[int, float] | None = None,
    rain_cover: bool = False,
) -> JitaiResult:
    """時刻順に判定する。recovery なら、介入のたびに回復分を後の窓の ln(HF) に足し、次の判定に反映する。

    rain_mm（時刻 → 降水量 [mm/h]）を渡すと、行きから帰りまでに基準以上の雨がある判定を見送る。
    基準は RAIN_MAX_MM_H、rain_cover（ポッドに雨よけあり）なら RAIN_COVER_MAX_MM_H。
    """
    t_grid = time_grid_min()
    params = TrueParams()
    profiles = {w.id: worker_profile(w, corrected=corrected) for w in workers}
    # 判定は z（個人の基準からのずれ ÷ 揺らぎ）で行う。回復は z に「回復分 ÷ 揺らぎ」を足す
    z = {w.id: (w.ln_hf - profiles[w.id].predict(t_grid)) / profiles[w.id].sd for w in workers}
    gained = {w.id: np.zeros(t_grid.size) for w in workers}  # 足した回復分（ln 単位）
    schedule = PodSchedule(dispatch)
    occupancy = Occupancy()
    next_eval = {w.id: 0 for w in workers}
    records: list[Record] = []
    for i, t_np in enumerate(t_grid):
        t = int(t_np)
        for w in workers:
            if t < next_eval[w.id] or not _flagged(z[w.id], i):
                continue
            rec = _decide(t, w, world, schedule, occupancy, pet, rain_mm, rain_cover)
            records.append(rec)
            if rec.outcome == "intervened" and recovery:
                assert rec.walk_m is not None and rec.pet_destination_c is not None
                arrive = t + walk_minutes(rec.walk_m)
                r = recovery_curve(
                    t_grid,
                    arrive,
                    arrive + DWELL_MIN,
                    amplitude=recovery_amplitude(rec.pet_destination_c, params),
                    tau_s=params.tau_s,
                    carryover_min=carryover_min,
                )
                gained[w.id] += r
                z[w.id] = z[w.id] + r / profiles[w.id].sd
            wait = (
                COOLDOWN_AFTER_INTERVENTION_MIN
                if rec.outcome == "intervened"
                else RETRY_AFTER_SKIP_MIN
            )
            next_eval[w.id] = t + wait
    windows = sum(int(flags_from_z(z[w.id]).sum()) for w in workers)
    burden = sum(_burden(w, gained[w.id], t_grid) for w in workers)
    return JitaiResult(records, list(workers), corrected, recovery, carryover_min, windows, burden)


def latent_decisions(
    workers: Sequence[Worker], *, corrected: bool = True, require_free: bool = False
) -> list[DecisionPoint]:
    """介入しない場合の判定点。需要はこれで数える。

    介入が無ければ判定は毎回見送りになり、次の判定は RETRY_AFTER_SKIP_MIN 後、回復も入らない。
    そのため判定点はワーカー（ln(HF) の系列）だけで決まり、配車・PET・予定に左右されない。
    実際の記録（介入あり）で需要を数えると、ポッドを置いた区画ほど判定が減って次の周で
    ポッドが離れ、配車と需要の反復が往復する（docs/jitai.md）。

    require_free なら、予定の空きが MIN_FREE_MIN に足りない判定点（段階 2 で見送るもの）を除く。
    予定は配車によらないので、除いても配車に左右されない。判定の間隔は除いた点でも変わらない
    （予定で見送っても次の判定は RETRY_AFTER_SKIP_MIN 後）。
    """
    t_grid = time_grid_min()
    points: list[DecisionPoint] = []
    for w in workers:
        prof = worker_profile(w, corrected=corrected)
        z = (w.ln_hf - prof.predict(t_grid)) / prof.sd
        next_eval = 0
        for i, t_np in enumerate(t_grid):
            t = int(t_np)
            if t < next_eval or not _flagged(z, i):
                continue
            if not require_free or free_minutes(w.meetings, t) >= MIN_FREE_MIN:
                points.append(DecisionPoint(t, w.id))
            next_eval = t + RETRY_AFTER_SKIP_MIN
    # run_jitai の記録と同じ順（時刻、同じ時刻ならワーカーの順）
    order = {w.id: k for k, w in enumerate(workers)}
    return sorted(points, key=lambda p: (p.time_min, order[p.worker_id]))


def _flagged(z: FloatArray, i: int) -> bool:
    """窓 i で過緊張か（直近 SUSTAIN_WINDOWS 窓の z がすべて閾値以下。detect と同じ規則）。"""
    if i < SUSTAIN_WINDOWS - 1:
        return False
    recent = z[i - SUSTAIN_WINDOWS + 1 : i + 1]
    return bool(np.all(np.where(np.isnan(recent), False, recent <= Z_THRESHOLD)))


def _burden(w: Worker, gained: FloatArray, t_grid: npt.NDArray[np.int64]) -> float:
    total = 0.0
    for e in w.episodes:
        span = (t_grid >= e.start_min) & (t_grid < e.end_min)
        total += float(np.clip(-e.depth - gained[span], 0.0, None).sum()) * STEP_MIN
    return total


def demand_cells(
    records: Sequence[TimedDecision],
    home_spot: Mapping[str, str],
    zone_of: Mapping[str, tuple[int, int]],
    min_workers: int = DEMAND_MIN_WORKERS,
) -> list[dict[str, Any]]:
    """過緊張の判定点を、勤務地に近い地点の区画・時刻ごとに数え、区画内の地点に等分する。

    min_workers 人未満の区画・時刻は区画としては出さず、時刻ごとに合算して min_workers 人以上なら
    全地点に等分して足す（少人数の所の需要を 0 にすると、配車がそこを無視するため）。
    """
    weight: dict[tuple[str, int], float] = {}
    for z in demand_zone_counts(records, home_spot, zone_of, min_workers):
        for sid in z["spot_ids"]:
            weight[(sid, z["hour"])] = weight.get((sid, z["hour"]), 0.0) + z["count"] / len(
                z["spot_ids"]
            )
    all_spots = sorted(zone_of)
    for r in demand_residual_counts(records, home_spot, zone_of, min_workers):
        for sid in all_spots:
            weight[(sid, r["hour"])] = weight.get((sid, r["hour"]), 0.0) + r["count"] / len(
                all_spots
            )
    return [
        {"spot_id": sid, "hour": h, "weight": w}
        for (sid, h), w in sorted(weight.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    ]


def _count_by_zone_hour(
    records: Sequence[TimedDecision],
    home_spot: Mapping[str, str],
    zone_of: Mapping[str, tuple[int, int]],
) -> tuple[Counter[tuple[tuple[int, int], int]], dict[tuple[tuple[int, int], int], set[str]]]:
    counts: Counter[tuple[tuple[int, int], int]] = Counter()
    people: dict[tuple[tuple[int, int], int], set[str]] = {}
    for r in records:
        key = (zone_of[home_spot[r.worker_id]], r.time_min // 60)
        counts[key] += 1
        people.setdefault(key, set()).add(r.worker_id)
    return counts, people


def demand_zone_counts(
    records: Sequence[TimedDecision],
    home_spot: Mapping[str, str],
    zone_of: Mapping[str, tuple[int, int]],
    min_workers: int = DEMAND_MIN_WORKERS,
) -> list[dict[str, Any]]:
    """区画・時刻ごとの判定点の件数（min_workers 人以上のものだけ）と、区画内の地点 ID。"""
    spots_in: dict[tuple[int, int], list[str]] = {}
    for sid, zone in sorted(zone_of.items()):
        spots_in.setdefault(zone, []).append(sid)
    counts, people = _count_by_zone_hour(records, home_spot, zone_of)
    return [
        {"zone": zone, "hour": h, "count": n, "spot_ids": spots_in[zone]}
        for (zone, h), n in sorted(counts.items())
        if len(people[(zone, h)]) >= min_workers
    ]


def demand_residual_counts(
    records: Sequence[TimedDecision],
    home_spot: Mapping[str, str],
    zone_of: Mapping[str, tuple[int, int]],
    min_workers: int = DEMAND_MIN_WORKERS,
) -> list[dict[str, Any]]:
    """min_workers 人未満で落とした区画・時刻を時刻ごとに合算した件数と人数（合算でも
    min_workers 人以上の時刻だけ）。spots は配る先の地点の数（全地点）。"""
    counts, people = _count_by_zone_hour(records, home_spot, zone_of)
    n: Counter[int] = Counter()
    who: dict[int, set[str]] = {}
    for (zone, h), c in counts.items():
        if len(people[(zone, h)]) < min_workers:
            n[h] += c
            who.setdefault(h, set()).update(people[(zone, h)])
    return [
        {"hour": h, "count": n[h], "workers": len(who[h]), "spots": len(zone_of)}
        for h in sorted(n)
        if len(who[h]) >= min_workers
    ]


def zone_hour_summary(
    records: Sequence[Record],
    home_spot: Mapping[str, str],
    zone_of: Mapping[str, tuple[int, int]],
    min_workers: int = DEMAND_MIN_WORKERS,
) -> list[dict[str, Any]]:
    """区画・時刻ごとの判定・介入・段階別の見送りの件数（min_workers 人以上のものだけ）。"""
    counts: dict[tuple[tuple[int, int], int], Counter[str]] = {}
    people: dict[tuple[tuple[int, int], int], set[str]] = {}
    for r in records:
        key = (zone_of[home_spot[r.worker_id]], r.time_min // 60)
        c = counts.setdefault(key, Counter())
        c["decisions"] += 1
        c["interventions" if r.outcome == "intervened" else f"skipped_{r.stage}"] += 1
        people.setdefault(key, set()).add(r.worker_id)
    return [
        {
            "zone": zone,
            "hour": h,
            "decisions": c["decisions"],
            "interventions": c["interventions"],
            "skipped_calendar": c["skipped_calendar"],
            "skipped_pods": c["skipped_pods"],
            "skipped_pet": c["skipped_pet"],
        }
        for (zone, h), c in sorted(counts.items())
        if len(people[(zone, h)]) >= min_workers
    ]


def summarize(result: JitaiResult) -> dict[str, Any]:
    records = result.records
    n = len(records)
    done = [r for r in records if r.outcome == "intervened"]
    by_stage = Counter(r.stage for r in records if r.stage is not None)
    by_reason = Counter(r.reason for r in records if r.reason is not None)
    return {
        "workers": len(result.workers),
        "decisions": n,
        "interventions": len(done),
        "intervention_rate": len(done) / n if n else 0.0,
        "workers_with_intervention": len({r.worker_id for r in done}),
        "skipped_by_stage": {s: by_stage.get(s, 0) for s in STAGES},
        "skipped_by_reason": {k: by_reason.get(k, 0) for k in REASONS},
        "detection": {
            "corrected": detection_scores(result.workers, corrected=True).to_json(),
            "naive": detection_scores(result.workers, corrected=False).to_json(),
            "used": "corrected" if result.corrected else "naive",
        },
        "recovery": {
            "enabled": result.recovery,
            "carryover_minutes": result.carryover_min,
            "hyperstress_windows": result.hyperstress_windows,
            "episode_burden": round(result.episode_burden, 3),
        },
    }
