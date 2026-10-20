"""シミュレーションの記録（1ステップ分のログ・チェックポイント）。"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.prompts import TurnSummary
from vmgs.world.plant import LastMove, PlantUnit, UnitStatus


class StepRecord(BaseModel):
    """1時間分。state_after は「行動後に1時間経過した時点」で、次ステップの state_before と一致する。"""

    step: int  # 1始まり
    time: str  # 判断した仮想時刻（ISO）
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    tool_calls: list[ToolCallRecord]
    decision: TurnDecision


class Checkpoint(BaseModel):
    """ステップ終了時点の世界とターン間記憶。再開に必要な全て。"""

    step: int
    sim_time: datetime
    plant: dict[str, Any]
    history: list[TurnSummary]


def plant_to_dict(p: PlantUnit) -> dict[str, Any]:
    last = p.last_move
    return {
        "spot_id": p.spot_id,
        "moisture": p.moisture,
        "health": p.health,
        "water_tank_ml": p.water_tank_ml,
        "battery_pct": p.battery_pct,
        "status": p.status.value,
        "heat_mitigation_score": p.heat_mitigation_score,
        "last_move": (
            {"from_spot": last.from_spot, "to_spot": last.to_spot, "at": last.at.isoformat()}
            if last
            else None
        ),
    }


def plant_from_dict(d: dict[str, Any]) -> PlantUnit:
    last = d["last_move"]
    return PlantUnit(
        spot_id=d["spot_id"],
        moisture=d["moisture"],
        health=d["health"],
        water_tank_ml=d["water_tank_ml"],
        battery_pct=d["battery_pct"],
        status=UnitStatus(d["status"]),
        heat_mitigation_score=d["heat_mitigation_score"],
        last_move=(
            LastMove(last["from_spot"], last["to_spot"], datetime.fromisoformat(last["at"]))
            if last
            else None
        ),
    )
