"""JITAI の需要の集計を stage.json の demand へ書き戻す（vmgs.jitai.demand）振る舞い。"""

import json
from pathlib import Path
from typing import Any

import pytest

from vmgs.contracts import DemandHour, Stage
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
    # Given 舞台のサンプルと、2 地点の集計 / When 書き戻す
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


def test_書き戻した舞台では配車が既定の需要を使わない() -> None:
    out = apply_demand(
        load_stage(SAMPLE_STAGE), _jitai_out([{"spot_id": "OTM-001", "hour": 12, "weight": 2}])
    )
    world = StageWorld.from_stage(out)
    assert not world.uses_default_demand
    assert world.demand("OTM-001", 12) == 2
    assert world.demand("OTM-002", 12) == 0


def test_書き戻した舞台は契約のスキーマで検証できる() -> None:
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
    ids=["別の区域", "別の日付", "舞台にない地点", "集計が空"],
)
def test_舞台と食い違う集計や空の集計は書き戻さない(jitai: dict[str, Any], needle: str) -> None:
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
