"""JITAI の需要の集計（`vmgs jitai` の出力の `demand.cells`）を、舞台の stage.json の demand へ書き戻す。

書き戻した舞台で `vmgs dispatch` を回すと、既定の需要（8〜19 時に一様）の代わりに、
過緊張の判定点を地点・時刻ごとに数えた集計値で配車する。集計値だけを扱い、個人の値は入らない。
"""

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from vmgs.contracts import DemandHour, Stage


def apply_demand(stage: Stage, jitai_out: Mapping[str, Any]) -> Stage:
    """stage の全地点の demand を、jitai_out の集計で置き換えた新しい Stage を返す（足し合わせない）。

    舞台と区域・日付が違う、舞台にない地点を含む、集計が空のときは ValueError。
    集計が空のまま書き戻すと、配車が黙って既定の需要に戻るため拒否する。
    """
    ref = jitai_out["stage_ref"]
    cells = jitai_out["demand"]["cells"]
    errors: list[str] = []
    if ref["area_id"] != stage.area.id:
        errors.append(f"区域が違う: 集計 {ref['area_id']} / 舞台 {stage.area.id}")
    if ref["date"] != stage.date.isoformat():
        errors.append(f"日付が違う: 集計 {ref['date']} / 舞台 {stage.date.isoformat()}")
    known = {s.id for s in stage.spots}
    unknown = sorted({c["spot_id"] for c in cells} - known)
    if unknown:
        errors.append(f"舞台にない地点: {', '.join(unknown)}")
    if not cells:
        errors.append(
            "需要の集計が空（3 人未満の地点・時刻は出さないため、すべて消えた可能性がある）"
        )
    if errors:
        raise ValueError("; ".join(errors))

    by_spot: dict[str, list[DemandHour]] = {}
    for c in sorted(cells, key=lambda c: (c["spot_id"], c["hour"])):
        by_spot.setdefault(c["spot_id"], []).append(DemandHour(hour=c["hour"], weight=c["weight"]))
    data = stage.model_dump(mode="json")
    for spot in data["spots"]:
        spot["demand"] = [d.model_dump() for d in by_spot.get(spot["id"], [])]
    try:
        # 時刻の重複や範囲は契約のモデルで確かめる
        return Stage.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"書き戻した舞台が契約に合わない: {e}") from e
