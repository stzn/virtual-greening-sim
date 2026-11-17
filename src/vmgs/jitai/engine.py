"""4 段階を時刻順に回し、介入の記録と集計を作る。

判定点: 過緊張を検知した窓のうち、前の判定から十分に時間がたったもの。
介入した後は COOLDOWN_AFTER_INTERVENTION_MIN、見送った後は RETRY_AFTER_SKIP_MIN の間は判定しない
（通知の負担を抑え、見送った人は空きができたら拾い直す）。全員を同じ時刻で進め、席の予約は早い者勝ち。
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from vmgs.contracts import Dispatch
from vmgs.fleet.stage_world import StageWorld
from vmgs.jitai.calendar import MIN_FREE_MIN, free_minutes
from vmgs.jitai.detect import detection_scores, worker_flags
from vmgs.jitai.pods import DWELL_MIN, Occupancy, PodSchedule, walk_minutes
from vmgs.jitai.safety import PetFn, pet_check
from vmgs.jitai.workers import Worker, time_grid_min

COOLDOWN_AFTER_INTERVENTION_MIN = 90
RETRY_AFTER_SKIP_MIN = 15
DEMAND_MIN_WORKERS = 3  # 需要の集計で、これ未満の人数の地点・時刻は出さない（個人の特定を防ぐ）

Outcome = Literal["intervened", "skipped"]
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


@dataclass(frozen=True)
class JitaiResult:
    records: list[Record]
    workers: list[Worker]
    corrected: bool


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
    # 段階 4: 二段階の PET 安全判定（近いポッドから順に、通るものを選ぶ）
    first: Record | None = None
    for c, arrive, leave, back in seated:
        hours = range(t // 60, (back - 1) // 60 + 1)
        reason, dest, route = pet_check(pet, c.spot_id, {w.home_spot_id, c.spot_id}, hours)
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
) -> JitaiResult:
    t_grid = time_grid_min()
    flags = {w.id: worker_flags(w, corrected=corrected) for w in workers}
    schedule = PodSchedule(dispatch)
    occupancy = Occupancy()
    next_eval = {w.id: 0 for w in workers}
    records: list[Record] = []
    for i, t_np in enumerate(t_grid):
        t = int(t_np)
        for w in workers:
            if t < next_eval[w.id] or not flags[w.id][i]:
                continue
            rec = _decide(t, w, world, schedule, occupancy, pet)
            records.append(rec)
            wait = (
                COOLDOWN_AFTER_INTERVENTION_MIN
                if rec.outcome == "intervened"
                else RETRY_AFTER_SKIP_MIN
            )
            next_eval[w.id] = t + wait
    return JitaiResult(records, list(workers), corrected)


def demand_cells(
    records: Sequence[Record],
    home_spot: Mapping[str, str],
    min_workers: int = DEMAND_MIN_WORKERS,
) -> list[dict[str, Any]]:
    """過緊張の判定点を、勤務地に近い地点・時刻ごとに数える。min_workers 人未満のセルは出さない。"""
    counts: Counter[tuple[str, int]] = Counter()
    people: dict[tuple[str, int], set[str]] = {}
    for r in records:
        key = (home_spot[r.worker_id], r.time_min // 60)
        counts[key] += 1
        people.setdefault(key, set()).add(r.worker_id)
    return [
        {"spot_id": s, "hour": h, "weight": n}
        for (s, h), n in sorted(counts.items())
        if len(people[(s, h)]) >= min_workers
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
    }
