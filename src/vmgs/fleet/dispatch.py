"""複数ポッドの配車（貪欲法）と、比較用の固定配置・何もしない方式。

毎時、次の順で決める（判断は計算で行い、LLM は使わない）:
1. 植物の世話（既存の規則 vmgs.agent.baseline と同じ条件）
   - 日向・猛暑（気温 33℃ 以上か日射 700W/m² 以上）・水分 40% 未満・電池 15% 以上 → 日影の地点へ退避
     （greedy のみ。fixed・none は動かない）
   - 水分 30% 未満・3 時間以内に雨なし・タンク 50ml 以上 → 散水（その時間は動かない）
2. 残りのポッドを（greedy のみ）、目的関数が最も増える移動から 1 台ずつ動かす（1 時間に 1 台 1 回まで）
   目的関数 = Σ 需要 × PET の改善 × 減衰^(その地点の何台目か) − 移動の罰 − 植物の健康の損失
   （需要 × PET の改善は、この先 2 時間分も割り引いて足す）
   制約: 地点の容量 capacity_pods、1 時間に動ける距離、移動に要る電池
3. 追加の規則（動かなかったポッドのみ。greedy・fixed）: 日向・猛暑で水分 50% 未満・
   タンクが予備（2000ml）以上・雨なし → 先回りして散水する。退避の閾値（40%）を割る前に水を足し、
   需要のある日向の地点に留まれるようにする。水が足りなければ 1. の規則どおり退避する
   （退避が快適度を下げた比較は docs/dispatch.md）
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

from vmgs.contracts import Dispatch, Pod, PodStep, SpotComfort, Stage, StageRef
from vmgs.fleet.comfort import PetTable
from vmgs.fleet.stage_world import StageWorld
from vmgs.weather.models import WeatherRecord
from vmgs.world.params import Params
from vmgs.world.plant import PlantUnit, UnitStatus, clamp, evapotranspiration, health_delta

Policy = Literal["greedy", "fixed", "none"]
POLICIES: tuple[Policy, ...] = ("greedy", "fixed", "none")

# 植物の世話の条件（vmgs.agent.baseline と同じ値）
RAIN_LOOKAHEAD_HOURS = 3
RAIN_PROB_PCT = 60.0
RAIN_MM = 1.0
HOT_TEMP_C = 33.0
HOT_SUN_W_M2 = 700.0
DRY_MOISTURE = 30.0
SUN_RETREAT_MOISTURE = 40.0
MIN_IRRIGATION_ML = 50
MAX_IRRIGATION_ML = 1000
# 日射係数がこれ以下の地点を「日影」とする（A/B/C では B が 0.2）
SHADE_MAX_SOLAR_FACTOR = 0.5


@dataclass(frozen=True)
class FleetParams:
    pods: int = 3
    # 1 時間に動ける距離 [m] と、移動 1 km あたりの電池 [%]（docs/dispatch.md）
    max_move_m: float = 1000.0
    move_cost_pct_per_km: float = 10.0
    min_battery_to_move: float = 15.0
    # 同じ地点の 2 台目以降の価値の減衰（座れる人が増える分だけ価値がある）
    pod_value_decay: float = 0.5
    # 先読み: 価値に、この先の何時間分を割り引いて足すか
    lookahead_hours: int = 2
    lookahead_discount: float = 0.5
    # 目的関数の重み（単位は「需要 × ℃」）
    move_penalty_per_km: float = 2.0
    plant_risk_weight: float = 1.0
    min_move_gain: float = 0.5  # これ以下の改善では動かない（行ったり来たりを防ぐ）
    initial_moisture: float = 60.0
    initial_tank_ml: int = 10_000
    # 先回りの散水（日向・猛暑で水分がこの値未満になったら、退避の閾値を割る前に散水する）。None で無効
    keep_moist_in_sun_below: float | None = 50.0
    preemptive_tank_reserve_ml: int = 2000


@dataclass
class FleetResult:
    policy: Policy
    pets: PetTable
    hours: list[int]
    timelines: dict[str, list[PodStep]]
    plants: dict[str, PlantUnit]
    water_used_ml: int = 0
    moves: int = 0
    occupancy: dict[tuple[str, int], int] = field(default_factory=dict)

    def pods_at(self, spot_id: str, hour: int) -> int:
        return self.occupancy.get((spot_id, hour), 0)


@dataclass
class _Pod:
    id: str
    plant: PlantUnit


class _Planner:
    def __init__(
        self,
        world: StageWorld,
        pets: PetTable,
        weather: dict[int, WeatherRecord],
        fleet: FleetParams,
        params: Params,
    ) -> None:
        self.world, self.pets, self.weather = world, pets, weather
        self.fleet, self.params = fleet, params

    # --- 価値 ---
    def slot_value(self, spot_id: str, hour: int, nth: int) -> float:
        """地点に nth 台目（0 始まり）を置いたときの価値。

        ポッドはすぐには動けないので、先の lookahead_hours 時間の価値も割り引いて足す。
        """
        total = 0.0
        for k in range(self.fleet.lookahead_hours + 1):
            t = hour + k
            if t not in self.weather:
                break
            demand = self.world.demand(spot_id, t)
            if demand > 0.0:
                total += demand * self.pets.relief(spot_id, t) * self.fleet.lookahead_discount**k
        return total * self.fleet.pod_value_decay**nth

    def plant_risk(self, plant: PlantUnit, spot_id: str, hour: int) -> float:
        """その地点で 1 時間過ごしたときの健康度の損失（の見込み）× 重み。"""
        w = self.weather[hour]
        spot = self.world.spot(spot_id, hour)
        temp_eff = spot.effective_temp(w.temperature_c, w.shortwave_w_m2, self.params)
        et = evapotranspiration(w, spot, temp_eff, self.params)
        moisture = clamp(plant.moisture - et + w.precipitation_mm * self.params.rain_gain_per_mm)
        delta = health_delta(moisture, temp_eff, self.params)
        return max(0.0, -delta) * self.fleet.plant_risk_weight

    def move_cost_pct(self, a: str, b: str) -> float:
        return self.world.distance_m(a, b) / 1000.0 * self.fleet.move_cost_pct_per_km

    def can_move(self, plant: PlantUnit, target: str) -> bool:
        if plant.status is not UnitStatus.ACTIVE or target == plant.spot_id:
            return False
        if self.world.distance_m(plant.spot_id, target) > self.fleet.max_move_m:
            return False
        need = max(self.fleet.min_battery_to_move, self.move_cost_pct(plant.spot_id, target))
        return plant.battery_pct >= need

    # --- 植物の世話の条件 ---
    def rain_soon(self, hour: int) -> bool:
        for h in range(hour + 1, hour + 1 + RAIN_LOOKAHEAD_HOURS):
            w = self.weather.get(h)
            if w and (
                (w.precipitation_prob_pct or 0.0) >= RAIN_PROB_PCT or w.precipitation_mm >= RAIN_MM
            ):
                return True
        return False

    def needs_retreat(self, plant: PlantUnit, hour: int) -> bool:
        return (
            plant.status is UnitStatus.ACTIVE
            and self._hot_sun(plant, hour)
            and plant.moisture < SUN_RETREAT_MOISTURE
            and plant.battery_pct >= self.fleet.min_battery_to_move
        )

    def _hot_sun(self, plant: PlantUnit, hour: int) -> bool:
        w = self.weather[hour]
        hot = w.temperature_c >= HOT_TEMP_C or w.shortwave_w_m2 >= HOT_SUN_W_M2
        return hot and self.world.spot(plant.spot_id, hour).solar_factor > SHADE_MAX_SOLAR_FACTOR

    def needs_preemptive_irrigation(self, plant: PlantUnit, hour: int) -> bool:
        limit = self.fleet.keep_moist_in_sun_below
        return (
            limit is not None
            and plant.status is UnitStatus.ACTIVE
            and self._hot_sun(plant, hour)
            and plant.moisture < limit
            and plant.water_tank_ml >= self.fleet.preemptive_tank_reserve_ml
            and not self.rain_soon(hour)
        )

    def needs_irrigation(self, plant: PlantUnit, hour: int) -> bool:
        return (
            plant.status is not UnitStatus.DEAD
            and plant.moisture < DRY_MOISTURE
            and not self.rain_soon(hour)
            and plant.water_tank_ml >= MIN_IRRIGATION_ML
        )


def place_initial(
    world: StageWorld,
    pets: PetTable,
    hours: Iterable[int],
    pods: int,
    *,
    decay: float = FleetParams.pod_value_decay,
) -> list[str]:
    """1 日の価値（Σ 需要 × PET の改善）が大きい地点から、容量を守って順に置く。"""
    hours = list(hours)
    day_value: dict[str, float] = {}
    for sid in world.spot_ids:
        day_value[sid] = sum(
            world.demand(sid, h) * pets.relief(sid, h) for h in hours if world.demand(sid, h) > 0
        )
    count: dict[str, int] = {}
    placed: list[str] = []
    for _ in range(pods):
        free = [s for s in world.spot_ids if count.get(s, 0) < world.capacity(s)]
        if not free:
            raise ValueError(f"ポッド {pods} 台を置く容量がありません")
        best = max(free, key=lambda s: day_value[s] * decay ** count.get(s, 0))
        count[best] = count.get(best, 0) + 1
        placed.append(best)
    return placed


def run_fleet(
    world: StageWorld,
    pets: PetTable,
    weather: dict[int, WeatherRecord],
    *,
    policy: Policy,
    fleet: FleetParams | None = None,
    params: Params | None = None,
    initial_spots: list[str] | None = None,
) -> FleetResult:
    fleet = fleet or FleetParams()
    params = params or Params()
    hours = sorted(set(weather) & world.hours)
    if not hours:
        raise ValueError("気象と舞台に共通の時刻がありません")
    planner = _Planner(world, pets, weather, fleet, params)
    starts = initial_spots or place_initial(
        world, pets, hours, fleet.pods, decay=fleet.pod_value_decay
    )
    if len(starts) != fleet.pods:
        raise ValueError(f"初期地点の数 {len(starts)} がポッドの数 {fleet.pods} と違います")
    pods = [
        _Pod(
            id=f"P{i + 1}",
            plant=PlantUnit(
                spot_id=s, moisture=fleet.initial_moisture, water_tank_ml=fleet.initial_tank_ml
            ),
        )
        for i, s in enumerate(starts)
    ]
    result = FleetResult(
        policy=policy,
        pets=pets,
        hours=hours,
        timelines={p.id: [] for p in pods},
        plants={p.id: p.plant for p in pods},
    )
    for hour in hours:
        actions = _decide(planner, pods, hour, policy)
        for pod in pods:
            action, target = actions[pod.id]
            _apply(planner, pod, action, target, result)
            result.occupancy[(pod.plant.spot_id, hour)] = (
                result.occupancy.get((pod.plant.spot_id, hour), 0) + 1
            )
            _step(planner, pod.plant, hour)
            result.timelines[pod.id].append(
                PodStep(
                    hour=hour,
                    spot_id=pod.plant.spot_id,
                    action=action,
                    plant_health=round(pod.plant.health, 2),
                )
            )
    return result


Action = tuple[Literal["stay", "move", "irrigate"], str | None]


def _decide(planner: _Planner, pods: list[_Pod], hour: int, policy: Policy) -> dict[str, Action]:
    actions: dict[str, Action] = {p.id: ("stay", None) for p in pods}
    if policy == "none":
        return actions
    count: dict[str, int] = {}
    for p in pods:
        count[p.plant.spot_id] = count.get(p.plant.spot_id, 0) + 1
    locked: set[str] = set()

    def free(spot_id: str) -> bool:
        return count.get(spot_id, 0) < planner.world.capacity(spot_id)

    def relocate(pod: _Pod, target: str) -> None:
        count[pod.plant.spot_id] -= 1
        count[target] = count.get(target, 0) + 1
        actions[pod.id] = ("move", target)
        locked.add(pod.id)

    # 1. 植物の世話（退避が先。退避先が無ければ散水の条件を見る）
    for pod in pods:
        plant = pod.plant
        if policy == "greedy" and planner.needs_retreat(plant, hour):
            shaded = [
                s
                for s in planner.world.spot_ids
                if planner.world.spot(s, hour).solar_factor <= SHADE_MAX_SOLAR_FACTOR
                and free(s)
                and planner.can_move(plant, s)
            ]
            if shaded:
                best = max(
                    shaded,
                    key=lambda s: (
                        planner.slot_value(s, hour, count.get(s, 0))
                        - planner.fleet.move_penalty_per_km
                        * planner.world.distance_m(plant.spot_id, s)
                        / 1000
                        - planner.plant_risk(plant, s, hour)
                    ),
                )
                relocate(pod, best)
                continue
        if planner.needs_irrigation(plant, hour):
            actions[pod.id] = ("irrigate", None)
            locked.add(pod.id)
        elif plant.status is not UnitStatus.ACTIVE:
            locked.add(pod.id)
    if policy == "greedy":
        _improve_by_moves(planner, pods, hour, actions, count, locked, free, relocate)

    # 3. 動かなかったポッドは、先回りの散水が要れば散水する（急がないので移動を優先する）
    for pod in pods:
        if actions[pod.id][0] == "stay" and planner.needs_preemptive_irrigation(pod.plant, hour):
            actions[pod.id] = ("irrigate", None)
    return actions


def _improve_by_moves(
    planner: _Planner,
    pods: list[_Pod],
    hour: int,
    actions: dict[str, Action],
    count: dict[str, int],
    locked: set[str],
    free: Callable[[str], bool],
    relocate: Callable[[_Pod, str], None],
) -> None:
    """目的関数が最も増える移動から順に適用する（各ポッドは 1 時間に 1 回まで）。"""
    while True:
        best_gain, best_move = planner.fleet.min_move_gain, None
        for pod in pods:
            if pod.id in locked:
                continue
            cur = pod.plant.spot_id
            keep = planner.slot_value(cur, hour, count[cur] - 1) - planner.plant_risk(
                pod.plant, cur, hour
            )
            for s in planner.world.spot_ids:
                if not free(s) or not planner.can_move(pod.plant, s):
                    continue
                gain = (
                    planner.slot_value(s, hour, count.get(s, 0))
                    - planner.fleet.move_penalty_per_km * planner.world.distance_m(cur, s) / 1000
                    - planner.plant_risk(pod.plant, s, hour)
                    - keep
                )
                if gain > best_gain:
                    best_gain, best_move = gain, (pod, s)
        if best_move is None:
            return
        relocate(*best_move)


def _apply(
    planner: _Planner, pod: _Pod, action: str, target: str | None, result: FleetResult
) -> None:
    plant = pod.plant
    if action == "move" and target is not None:
        plant.battery_pct -= planner.move_cost_pct(plant.spot_id, target)
        plant.spot_id = target
        result.moves += 1
    elif action == "irrigate":
        applied = min(MAX_IRRIGATION_ML, plant.water_tank_ml)
        plant.water_tank_ml -= applied
        plant.moisture = clamp(plant.moisture + applied * planner.params.irrigation_gain_per_ml)
        result.water_used_ml += applied


def _step(planner: _Planner, plant: PlantUnit, hour: int) -> None:
    """1 時間の生理（vmgs.world.world.World._step と同じ式）。"""
    if plant.status is UnitStatus.DEAD:
        return
    params = planner.params
    w = planner.weather[hour]
    spot = planner.world.spot(plant.spot_id, hour)
    temp_eff = spot.effective_temp(w.temperature_c, w.shortwave_w_m2, params)
    et = evapotranspiration(w, spot, temp_eff, params)
    plant.moisture = clamp(plant.moisture - et + w.precipitation_mm * params.rain_gain_per_mm)
    plant.health = clamp(plant.health + health_delta(plant.moisture, temp_eff, params))
    solar = params.solar_charge_per_kw * spot.effective_radiation(w.shortwave_w_m2) / 1000.0
    plant.battery_pct = clamp(plant.battery_pct + solar - params.idle_drain_pct)
    if plant.health <= 0.0:
        plant.status = UnitStatus.DEAD
    elif plant.battery_pct <= 0.0:
        plant.status = UnitStatus.STRANDED
    else:
        plant.status = UnitStatus.ACTIVE


def to_dispatch(result: FleetResult, stage: Stage, *, generator: str, scenario: str) -> Dispatch:
    """配車結果を契約 v1 の Dispatch にする。快適度は全地点・全時刻の PET（ポッドがあれば樹冠の下）。"""
    comfort = [
        SpotComfort(
            spot_id=sid,
            hour=h,
            pet_c=result.pets.pet(sid, h, covered=result.pods_at(sid, h) > 0),
            pods_present=result.pods_at(sid, h),
        )
        for sid in result.pets.world.spot_ids
        for h in result.hours
    ]
    return Dispatch(
        schema_version="1",
        stage_ref=StageRef(area_id=stage.area.id, date=stage.date),
        generator=generator,
        weather_scenario=scenario,
        pods=[Pod(id=pid, timeline=steps) for pid, steps in result.timelines.items()],
        comfort=comfort,
    )
