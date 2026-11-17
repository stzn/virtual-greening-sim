"""配車方式の比較（人の快適度と植物の健康）。"""

from dataclasses import dataclass

from vmgs.fleet.comfort import PetTable
from vmgs.fleet.dispatch import FleetResult
from vmgs.fleet.stage_world import StageWorld
from vmgs.world.plant import UnitStatus


@dataclass(frozen=True)
class PolicySummary:
    name: str
    demand_pet_c: float  # 需要のある地点・時刻の PET（需要で加重した平均）
    pod_spot_pet_c: float | None  # そのうちポッドのある地点・時刻の PET（需要で加重した平均）
    relief: float  # Σ 需要 × ポッドによる PET の低下 [需要・℃]
    demand_covered_pct: float  # 需要のうち、ポッドのある地点・時刻の割合
    mean_health: float | None  # 終了時の健康度の平均
    min_health: float | None  # 1 日を通した健康度の最低（全ポッド）
    dead: int
    moves: int
    water_used_ml: int


def _demand_cells(world: StageWorld, hours: list[int]) -> list[tuple[str, int, float]]:
    return [
        (sid, h, world.demand(sid, h))
        for sid in world.spot_ids
        for h in hours
        if world.demand(sid, h) > 0
    ]


def summarize(result: FleetResult, world: StageWorld, pets: PetTable) -> PolicySummary:
    cells = _demand_cells(world, result.hours)
    total = sum(d for _, _, d in cells)
    pet_sum = covered_sum = covered_pet = relief = 0.0
    for sid, h, d in cells:
        covered = result.pods_at(sid, h) > 0
        p = pets.pet(sid, h, covered=covered)
        pet_sum += d * p
        if covered:
            covered_sum += d
            covered_pet += d * p
            relief += d * pets.relief(sid, h)
    healths = [p.health for p in result.plants.values()]
    return PolicySummary(
        name=result.policy,
        demand_pet_c=pet_sum / total if total else float("nan"),
        pod_spot_pet_c=covered_pet / covered_sum if covered_sum else None,
        relief=relief,
        demand_covered_pct=100.0 * covered_sum / total if total else 0.0,
        mean_health=sum(healths) / len(healths),
        min_health=min(s.plant_health for steps in result.timelines.values() for s in steps),
        dead=sum(1 for p in result.plants.values() if p.status is UnitStatus.DEAD),
        moves=result.moves,
        water_used_ml=result.water_used_ml,
    )


def no_pod_summary(world: StageWorld, pets: PetTable, hours: list[int]) -> PolicySummary:
    """ポッドを 1 台も置かないときの快適度（植物の欄は空）。"""
    cells = _demand_cells(world, hours)
    total = sum(d for _, _, d in cells)
    pet_sum = sum(d * pets.pet(sid, h, covered=False) for sid, h, d in cells)
    return PolicySummary(
        name="no_pods",
        demand_pet_c=pet_sum / total if total else float("nan"),
        pod_spot_pet_c=None,
        relief=0.0,
        demand_covered_pct=0.0,
        mean_health=None,
        min_health=None,
        dead=0,
        moves=0,
        water_used_ml=0,
    )


def _f(v: float | None, digits: int = 1) -> str:
    return "—" if v is None else f"{v:.{digits}f}"


def comparison_markdown(rows: list[PolicySummary]) -> str:
    lines = [
        (
            "| 方式 | 需要地点の PET [℃] | ポッドのある地点の PET [℃] | PET 低減 [需要・℃] "
            "| 需要の被覆 [%] | 健康度 終了時平均 | 健康度 日中最低 | 枯死 | 移動 | 散水 [ml] |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.name} | {_f(r.demand_pet_c, 2)} | {_f(r.pod_spot_pet_c, 2)} | {_f(r.relief)} "
            f"| {_f(r.demand_covered_pct)} | {_f(r.mean_health)} | {_f(r.min_health)} "
            f"| {r.dead} | {r.moves} | {r.water_used_ml} |"
        )
    return "\n".join(lines)
