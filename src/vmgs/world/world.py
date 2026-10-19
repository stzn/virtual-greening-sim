"""仮想世界。LLMを一切知らない純粋なPython。ツールはこのクラスを呼ぶだけ。"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import WeatherProvider
from vmgs.world.clock import SimClock
from vmgs.world.params import Params
from vmgs.world.plant import PlantUnit, UnitStatus, clamp, evapotranspiration, health_delta
from vmgs.world.spots import SPOTS, Spot, move_cost_pct


class WorldError(Exception):
    """世界のルール違反。ツール層が構造化エラーに変換する。"""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class PlantState(BaseModel):
    sim_time: datetime
    spot_id: str
    spot_name: str
    soil_moisture_pct: float
    health: float
    water_tank_ml: int
    battery_pct: float
    status: str
    heat_mitigation_score: float


class MoveResult(BaseModel):
    new_spot_id: str
    battery_used_pct: float
    battery_pct: float


class IrrigateResult(BaseModel):
    soil_moisture_pct: float
    water_tank_ml: int
    cooling_c: float
    heat_mitigation_gain: float


class World:
    def __init__(
        self,
        weather: WeatherProvider,
        clock: SimClock,
        *,
        notes_path: Path,
        params: Params | None = None,
        start_spot: str = "A",
    ) -> None:
        self.weather = weather
        self.clock = clock
        self.params = params or Params()
        self.notes_path = notes_path
        self.plant = PlantUnit(
            spot_id=start_spot,
            moisture=self.params.initial_moisture,
            water_tank_ml=self.params.initial_tank_ml,
        )

    # --- 観測 ---
    @property
    def spot(self) -> Spot:
        return SPOTS[self.plant.spot_id]

    def current_weather(self) -> WeatherRecord:
        return self.weather.at(self.clock.now)

    def effective_temp(self) -> float:
        w = self.current_weather()
        return self.spot.effective_temp(w.temperature_c, w.shortwave_w_m2, self.params)

    def state(self) -> PlantState:
        p = self.plant
        return PlantState(
            sim_time=self.clock.now,
            spot_id=p.spot_id,
            spot_name=self.spot.name,
            soil_moisture_pct=p.moisture,
            health=p.health,
            water_tank_ml=p.water_tank_ml,
            battery_pct=p.battery_pct,
            status=p.status.value,
            heat_mitigation_score=p.heat_mitigation_score,
        )

    # --- 時間経過 ---
    def advance(self, hours: int = 1) -> None:
        for _ in range(hours):
            self._step()

    def _step(self) -> None:
        p, params = self.plant, self.params
        if p.status is not UnitStatus.DEAD:
            w = self.current_weather()
            spot = self.spot
            temp_eff = spot.effective_temp(w.temperature_c, w.shortwave_w_m2, params)

            et = evapotranspiration(w, spot, temp_eff, params)
            p.moisture = clamp(p.moisture - et + w.precipitation_mm * params.rain_gain_per_mm)
            p.health = clamp(p.health + health_delta(p.moisture, temp_eff, params))

            solar = params.solar_charge_per_kw * spot.effective_radiation(w.shortwave_w_m2) / 1000.0
            p.battery_pct = clamp(p.battery_pct + solar - params.idle_drain_pct)

            if p.health <= 0.0:
                p.status = UnitStatus.DEAD
            elif p.battery_pct <= 0.0:
                p.status = UnitStatus.STRANDED
            else:
                p.status = UnitStatus.ACTIVE
        self.clock.advance(1)

    # --- 行動 ---
    def _require_alive(self) -> None:
        if self.plant.status is UnitStatus.DEAD:
            raise WorldError("UNIT_DEAD", "植物が枯死しているため操作できません")

    def move_to(self, spot_id: str) -> MoveResult:
        self._require_alive()
        p = self.plant
        if spot_id not in SPOTS:
            raise WorldError("UNKNOWN_SPOT", f"スポット {spot_id!r} は存在しません")
        if spot_id == p.spot_id:
            raise WorldError("ALREADY_AT_SPOT", f"すでにスポット {spot_id} にいます")
        cost = move_cost_pct(p.spot_id, spot_id, self.params)
        if p.battery_pct < cost:
            raise WorldError(
                "INSUFFICIENT_BATTERY",
                f"バッテリー不足（必要 {cost:.1f}%・残り {p.battery_pct:.1f}%）",
                required_pct=cost,
                battery_pct=p.battery_pct,
            )
        p.battery_pct -= cost
        p.spot_id = spot_id
        return MoveResult(new_spot_id=spot_id, battery_used_pct=cost, battery_pct=p.battery_pct)

    def irrigate(self, amount_ml: int) -> IrrigateResult:
        self._require_alive()
        p, params = self.plant, self.params
        if amount_ml <= 0:
            raise WorldError("INVALID_AMOUNT", "散水量は正の値で指定してください")
        if amount_ml > p.water_tank_ml:
            raise WorldError(
                "INSUFFICIENT_WATER",
                f"タンク残量不足（要求 {amount_ml}ml・残り {p.water_tank_ml}ml）",
                requested_ml=amount_ml,
                water_tank_ml=p.water_tank_ml,
            )
        p.water_tank_ml -= amount_ml
        p.moisture = clamp(p.moisture + amount_ml * params.irrigation_gain_per_ml)

        # 気化冷却：冷却需要（実効気温が閾値を超えた分）が大きいほど暑熱緩和への貢献が大きい
        cooling = min(amount_ml * params.cooling_per_ml, params.cooling_cap_c)
        need = max(0.0, self.effective_temp() - params.heat_threshold_c)
        gain = cooling * need
        p.heat_mitigation_score += gain
        return IrrigateResult(
            soil_moisture_pct=p.moisture,
            water_tank_ml=p.water_tank_ml,
            cooling_c=cooling,
            heat_mitigation_gain=gain,
        )

    def log_note(self, text: str, tags: list[str]) -> Path:
        self.notes_path.parent.mkdir(parents=True, exist_ok=True)
        tag_str = " ".join(f"`{t}`" for t in tags)
        with self.notes_path.open("a", encoding="utf-8") as f:
            f.write(f"- **{self.clock.now:%Y-%m-%d %H:%M}** {tag_str} {text}\n")
        return self.notes_path
