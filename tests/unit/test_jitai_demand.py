"""JITAI の需要の集計を stage.json の demand へ書き戻す（vmgs.jitai.demand）振る舞い。"""

import json
from pathlib import Path
from typing import Any

import pytest

from vmgs.contracts import DemandHour, Dispatch, Pod, PodStep, Stage, StageRef
from vmgs.fleet.stage_world import StageWorld, load_stage
from vmgs.jitai.demand import apply_demand

SAMPLE_STAGE = Path("contracts/fixtures/stage.sample.json")


def _jitai_out(
    cells: list[dict[str, Any]], area: str = "otemachi", date: str = "2025-08-30"
) -> dict[str, Any]:
    return {
        "stage_ref": {"area_id": area, "date": date},
        "demand": {"min_workers": 3, "cells": cells},
    }


def test_集計した需要を地点ごとに時刻順で書き込み_他の地点は空にする() -> None:
    # Given 地点データのサンプルと、2 地点の集計 / When 書き戻す
    stage = load_stage(SAMPLE_STAGE)
    out = apply_demand(
        stage,
        _jitai_out(
            [
                {"spot_id": "OTM-002", "hour": 15, "weight": 4},
                {"spot_id": "OTM-001", "hour": 12, "weight": 5},
                {"spot_id": "OTM-001", "hour": 10, "weight": 3},
            ]
        ),
    )
    # Then 地点ごと・時刻順に入り、集計の無い地点は空
    demand = {s.id: s.demand for s in out.spots}
    assert demand["OTM-001"] == [DemandHour(hour=10, weight=3), DemandHour(hour=12, weight=5)]
    assert demand["OTM-002"] == [DemandHour(hour=15, weight=4)]
    assert demand["OTM-003"] == []
    # 需要以外（地点・時間別の遮蔽）は変えない
    assert [s.hourly for s in out.spots] == [s.hourly for s in stage.spots]


def test_既存の需要は足し合わせず置き換える() -> None:
    data = json.loads(SAMPLE_STAGE.read_text(encoding="utf-8"))
    data["spots"][0]["demand"] = [{"hour": 12, "weight": 100}]
    data["spots"][2]["demand"] = [{"hour": 9, "weight": 7}]
    out = apply_demand(
        Stage.model_validate(data), _jitai_out([{"spot_id": "OTM-001", "hour": 12, "weight": 2}])
    )
    assert out.spots[0].demand == [DemandHour(hour=12, weight=2)]
    assert out.spots[2].demand == []


def test_書き戻した地点データでは配車が既定の需要を使わない() -> None:
    out = apply_demand(
        load_stage(SAMPLE_STAGE), _jitai_out([{"spot_id": "OTM-001", "hour": 12, "weight": 2}])
    )
    world = StageWorld.from_stage(out)
    assert not world.uses_default_demand
    assert world.demand("OTM-001", 12) == 2
    assert world.demand("OTM-002", 12) == 0


def test_書き戻した地点データは契約のスキーマで検証できる() -> None:
    out = apply_demand(
        load_stage(SAMPLE_STAGE), _jitai_out([{"spot_id": "OTM-003", "hour": 8, "weight": 1}])
    )
    assert Stage.model_validate_json(out.model_dump_json()) == out


@pytest.mark.parametrize(
    ("jitai", "needle"),
    [
        (_jitai_out([{"spot_id": "OTM-001", "hour": 12, "weight": 1}], area="shibuya"), "shibuya"),
        (
            _jitai_out([{"spot_id": "OTM-001", "hour": 12, "weight": 1}], date="2025-07-15"),
            "2025-07-15",
        ),
        (_jitai_out([{"spot_id": "NOPE", "hour": 12, "weight": 1}]), "NOPE"),
        (_jitai_out([]), "空"),
    ],
    ids=["別の区域", "別の日付", "地点データにない地点", "集計が空"],
)
def test_地点データと食い違う集計や空の集計は書き戻さない(
    jitai: dict[str, Any], needle: str
) -> None:
    # 集計が空のまま書き戻すと、配車が黙って既定の需要に戻るので拒否する
    with pytest.raises(ValueError, match=needle):
        apply_demand(load_stage(SAMPLE_STAGE), jitai)


def test_同じ地点と時刻が重なる集計は拒否する() -> None:
    cells = [
        {"spot_id": "OTM-001", "hour": 12, "weight": 1},
        {"spot_id": "OTM-001", "hour": 12, "weight": 2},
    ]
    with pytest.raises(ValueError):
        apply_demand(load_stage(SAMPLE_STAGE), _jitai_out(cells))


def test_区画と時刻ごとに判定_介入_段階別の見送りを数え_3人未満は出さない() -> None:
    from vmgs.jitai.engine import Outcome, Record, Stage, zone_hour_summary

    def rec(t: int, w: str, outcome: Outcome = "skipped", stage: Stage | None = "pods") -> Record:
        reason = {"pods": "no_pod_nearby", "calendar": "in_meeting", "pet": "pet_route"}
        return Record(t, w, outcome, stage, reason.get(stage or ""), None, None, None, None, None)

    zone_of = {"A": (0, 0), "B": (0, 0), "C": (3, 3)}
    home = {"W0": "A", "W1": "B", "W2": "A", "W9": "C"}
    recs = [
        rec(12 * 60, "W0", "intervened", None),
        rec(12 * 60, "W1", stage="calendar"),
        rec(12 * 60 + 15, "W1", stage="pet"),
        rec(12 * 60, "W2"),
        rec(12 * 60, "W9"),
    ]
    rows = zone_hour_summary(recs, home, zone_of, min_workers=3)
    assert rows == [
        {
            "zone": (0, 0),
            "hour": 12,
            "decisions": 4,
            "interventions": 1,
            "skipped_calendar": 1,
            "skipped_pods": 1,
            "skipped_pet": 1,
        }
    ]


# --- 介入しない場合の判定点（需要の元） ---


def _all_day(spot_ids: list[str], hours: range = range(24)) -> Dispatch:
    stage = load_stage(SAMPLE_STAGE)
    return Dispatch(
        schema_version="1",
        generator="test",
        stage_ref=StageRef(area_id=stage.area.id, date=stage.date),
        weather_scenario="normal",
        pods=[
            Pod(
                id=f"P{i}",
                timeline=[
                    PodStep(hour=h, spot_id=sid, action="stay", plant_health=1.0) for h in hours
                ],
            )
            for i, sid in enumerate(spot_ids)
        ],
        comfort=[],
    )


def _cool_pet(spot_id: str, hour: int, *, covered: bool) -> float:
    return 25.0  # どこでも閾値より低い（PET で見送らない）


def test_介入しない場合の判定点は_ポッドの無い配車で回した判定と一致する() -> None:
    from vmgs.jitai.engine import latent_decisions, run_jitai
    from vmgs.jitai.workers import make_workers

    stage = load_stage(SAMPLE_STAGE)
    world = StageWorld.from_stage(stage)
    workers = make_workers(stage, 40, seed=3)
    # Given 判定の時間帯（8〜19 時）にポッドがいない配車（全員が見送りになる） / When 回す
    # （契約ではポッドを 1 台以上置くので、0 時だけいるポッドにする）
    no_pods = _all_day(["OTM-001"], hours=range(1))
    result = run_jitai(workers, world, no_pods, _cool_pet, recovery=False)
    assert result.records and all(r.outcome == "skipped" for r in result.records)
    # Then 介入しない場合の判定点と、時刻・ワーカーが一致する
    expected = [(r.time_min, r.worker_id) for r in result.records]
    assert [(p.time_min, p.worker_id) for p in latent_decisions(workers)] == expected


def test_需要の集計は配車によらない() -> None:
    from vmgs.jitai.demand import DEMAND_ZONE_M, demand_zones
    from vmgs.jitai.engine import demand_cells, latent_decisions, run_jitai
    from vmgs.jitai.workers import make_workers

    stage = load_stage(SAMPLE_STAGE)
    world = StageWorld.from_stage(stage)
    workers = make_workers(stage, 60, seed=5)
    home = {w.id: w.home_spot_id for w in workers}
    zone_of = demand_zones(stage, DEMAND_ZONE_M)
    # Given 全地点に終日ポッドがいる配車では介入が起き、判定点が介入なしと変わる
    busy = run_jitai(workers, world, _all_day([s.id for s in stage.spots]), _cool_pet)
    assert any(r.outcome == "intervened" for r in busy.records)
    latent = latent_decisions(workers)
    assert [(r.time_min, r.worker_id) for r in busy.records] != [
        (p.time_min, p.worker_id) for p in latent
    ]
    # Then 需要（介入しない場合の判定点の集計）は、配車の結果を見ずに決まる
    cells = demand_cells(latent, home, zone_of, min_workers=1)
    assert cells
    assert cells == demand_cells(latent_decisions(workers), home, zone_of, min_workers=1)


def test_予定の空きを求めると_介入しない場合の判定点のうち予定で見送られないものだけが残る() -> (
    None
):
    from vmgs.jitai.calendar import MIN_FREE_MIN, free_minutes
    from vmgs.jitai.engine import latent_decisions, run_jitai
    from vmgs.jitai.workers import make_workers

    stage = load_stage(SAMPLE_STAGE)
    world = StageWorld.from_stage(stage)
    workers = make_workers(stage, 40, seed=3)
    by_id = {w.id: w for w in workers}
    # Given 判定の時間帯にポッドのいない配車で回すと、予定か、ポッドの段階で見送られる
    no_pods = _all_day(["OTM-001"], hours=range(1))
    records = run_jitai(workers, world, no_pods, _cool_pet, recovery=False).records
    assert {r.stage for r in records} == {"calendar", "pods"}
    # When 予定の空きを求めて判定点を数える
    points = latent_decisions(workers, require_free=True)
    # Then 予定で見送られなかった判定（ポッドの段階まで進んだもの）と一致する
    assert [(p.time_min, p.worker_id) for p in points] == [
        (r.time_min, r.worker_id) for r in records if r.stage == "pods"
    ]
    assert all(
        free_minutes(by_id[p.worker_id].meetings, p.time_min) >= MIN_FREE_MIN for p in points
    )
    # 予定の空きを求めない判定点の一部になっている
    assert set(points) < set(latent_decisions(workers))
