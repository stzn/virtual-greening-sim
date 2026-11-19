"""JITAI の需要の集計（`vmgs jitai` の出力の `demand.cells`）を、地点データの stage.json の demand へ書き戻す。

書き戻した地点データで `vmgs dispatch` を回すと、既定の需要（8〜19 時に一様）の代わりに、
過緊張の判定点を地点・時刻ごとに数えた集計値で配車する。集計値だけを扱い、個人の値は入らない。
"""

import math
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from vmgs.contracts import DemandHour, JitaiSummary, Stage
from vmgs.fleet.stage_world import EARTH_RADIUS_M
from vmgs.jitai.engine import DEMAND_MIN_WORKERS, JitaiResult, zone_hour_summary
from vmgs.jitai.pods import WALK_RADIUS_M

# 需要を数える区画の一辺。徒歩圏（片道 400 m）に合わせる。地点×時刻では細かすぎて、
# 少人数（合成 30 人、実際の研究の参加者も少ない）だと 3 人未満の規則でほぼすべて消えるため、
# 区画×時刻で数えてから消し、残った件数を区画内の地点に等分する。
DEMAND_ZONE_M = WALK_RADIUS_M

Zone = tuple[int, int]


def demand_zones(stage: Stage, zone_m: float = DEMAND_ZONE_M) -> dict[str, Zone]:
    """地点 ID → 区画（区域の南西の角から、東西・南北に zone_m ごとに切った升目の番号）。"""
    west, south, _east, north = stage.area.bbox
    # 区域は数 km なので、区域の中央の緯度で平面に近似する
    m_per_deg_lat = math.radians(1.0) * EARTH_RADIUS_M
    m_per_deg_lon = m_per_deg_lat * math.cos(math.radians((south + north) / 2))
    return {
        s.id: (
            math.floor((s.lon - west) * m_per_deg_lon / zone_m),
            math.floor((s.lat - south) * m_per_deg_lat / zone_m),
        )
        for s in stage.spots
    }


def apply_demand(stage: Stage, jitai_out: Mapping[str, Any]) -> Stage:
    """stage の全地点の demand を、jitai_out の集計で置き換えた新しい Stage を返す（足し合わせない）。

    地点データと区域・日付が違う、地点データにない地点を含む、集計が空のときは ValueError。
    集計が空のまま書き戻すと、配車が黙って既定の需要に戻るため拒否する。
    """
    ref = jitai_out["stage_ref"]
    cells = jitai_out["demand"]["cells"]
    errors: list[str] = []
    if ref["area_id"] != stage.area.id:
        errors.append(f"区域が違う: 集計 {ref['area_id']} / 地点データ {stage.area.id}")
    if ref["date"] != stage.date.isoformat():
        errors.append(f"日付が違う: 集計 {ref['date']} / 地点データ {stage.date.isoformat()}")
    known = {s.id for s in stage.spots}
    unknown = sorted({c["spot_id"] for c in cells} - known)
    if unknown:
        errors.append(f"地点データにない地点: {', '.join(unknown)}")
    if not cells:
        errors.append(
            "需要の集計が空（3 人未満の地点・時刻は出さないため、すべて消えた可能性がある）"
        )
    if errors:
        raise ValueError("; ".join(errors))

    by_spot: dict[str, list[DemandHour]] = {}
    for c in sorted(cells, key=lambda c: (c["spot_id"], c["hour"])):
        by_spot.setdefault(c["spot_id"], []).append(DemandHour(hour=c["hour"], weight=c["weight"]))
    # 元の地点データに無かった既定値（版 2 の屋根のフィールドなど）は書き足さない
    data = stage.model_dump(mode="json", exclude_unset=True)
    for spot in data["spots"]:
        spot["demand"] = [d.model_dump() for d in by_spot.get(spot["id"], [])]
    try:
        # 時刻の重複や範囲は契約のモデルで確かめる
        return Stage.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"書き戻した地点データが契約に合わない: {e}") from e


def jitai_summary(
    stage: Stage,
    result: JitaiResult,
    home_spot: Mapping[str, str],
    scenario: str,
    generator: str,
    zone_m: float = DEMAND_ZONE_M,
    min_workers: int = DEMAND_MIN_WORKERS,
) -> JitaiSummary:
    """地図アプリに渡す JITAI の集計（契約 v1）。区画・時刻ごとの件数だけで、個人の記録は入れない。"""
    west, south, _east, _north = stage.area.bbox
    zones = zone_hour_summary(result.records, home_spot, demand_zones(stage, zone_m), min_workers)
    return JitaiSummary.model_validate(
        {
            "schema_version": "1",
            "stage_ref": {"area_id": stage.area.id, "date": stage.date},
            "generator": generator,
            "weather_scenario": scenario,
            "workers": len(result.workers),
            "min_workers": min_workers,
            "grid": {"origin_lon": west, "origin_lat": south, "zone_m": zone_m},
            "totals": {
                "decisions": len(result.records),
                "interventions": sum(r.outcome == "intervened" for r in result.records),
            },
            "recovery": {"enabled": result.recovery, "carryover_minutes": result.carryover_min},
            "zones": zones,
        }
    )
