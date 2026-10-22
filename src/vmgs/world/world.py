"""仮想世界。LLMを一切知らない純粋なPython。ツールはこのクラスを呼ぶだけ。"""

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import WeatherProvider
from vmgs.world.clock import SimClock
from vmgs.world.params import Params
from vmgs.world.plant import (
    LastMove,
    PlantUnit,
    UnitStatus,
    clamp,
    evapotranspiration,
    health_delta,
)
from vmgs.world.spots import SPOTS, Spot, move_cost_pct


class WorldError(Exception):
    """世界のルール違反。ツール層が構造化エラーに変換する。"""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


class LastMoveInfo(BaseModel):
    from_spot: str
    to_spot: str
    hours_ago: int


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
    last_move: LastMoveInfo | None = None
    moves_in_window: int = 0
    move_budget_remaining: int = 0


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
        horizon_end: datetime | None = None,
        planning_observations: bool = False,
    ) -> None:
        self.weather = weather
        self.clock = clock
        self.params = params or Params()
        self.notes_path = notes_path
        # 計画観測: シミュレーションの終了時刻までの残りから、水・電池の「1時間あたりに使える量」を返す
        self.horizon_end = horizon_end
        self.planning_observations_enabled = planning_observations
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
        last = p.last_move
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
            moves_in_window=self._moves_in_window(),
            move_budget_remaining=max(0, self.params.move_budget_max - self._moves_in_window()),
            last_move=(
                LastMoveInfo(
                    from_spot=last.from_spot,
                    to_spot=last.to_spot,
                    hours_ago=self._hours_since(last.at),
                )
                if last
                else None
            ),
        )

    def _hours_since(self, t: datetime) -> int:
        return int((self.clock.now - t).total_seconds() // 3600)

    def planning_observations(self) -> dict[str, float | int | None] | None:
        """計画観測（計算値のみ。判断基準は含まない）。無効なら None。

        hours_remaining は現在のターンを含む残りターン数。水・電池は残量を残り時間で割っただけの値。
        終了時刻が不明、または残り時間が 0 以下のときは、3つとも None。
        """
        if not self.planning_observations_enabled:
            return None
        remaining: int | None = None
        if self.horizon_end is not None:
            remaining = int((self.horizon_end - self.clock.now).total_seconds() // 3600)
        if remaining is None or remaining <= 0:
            return {
                "hours_remaining": None,
                "water_per_remaining_hour_ml": None,
                "battery_per_remaining_hour_pct": None,
            }
        p = self.plant
        return {
            "hours_remaining": remaining,
            "water_per_remaining_hour_ml": round(p.water_tank_ml / remaining, 1),
            "battery_per_remaining_hour_pct": round(p.battery_pct / remaining, 1),
        }

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
        self._guard_reversal(spot_id)
        self._guard_move_budget()
        cost = move_cost_pct(p.spot_id, spot_id, self.params)
        if p.battery_pct < cost:
            raise WorldError(
                "INSUFFICIENT_BATTERY",
                f"バッテリー不足（必要 {cost:.1f}%・残り {p.battery_pct:.1f}%）",
                required_pct=cost,
                battery_pct=p.battery_pct,
            )
        p.battery_pct -= cost
        p.last_move = LastMove(from_spot=p.spot_id, to_spot=spot_id, at=self.clock.now)
        p.move_times.append(self.clock.now)
        p.spot_id = spot_id
        return MoveResult(new_spot_id=spot_id, battery_used_pct=cost, battery_pct=p.battery_pct)

    def _moves_in_window(self) -> int:
        window = self.params.move_budget_window_hours
        return sum(1 for t in self.plant.move_times if self._hours_since(t) < window)

    def _guard_move_budget(self) -> None:
        """直近の移動回数が予算に達していれば拒否する（健康度が低い緊急時は免除）。"""
        p, params = self.plant, self.params
        if p.health < params.reversal_emergency_health:
            return
        recent = sorted(
            t for t in p.move_times if self._hours_since(t) < params.move_budget_window_hours
        )
        if len(recent) < params.move_budget_max:
            return
        blocking = recent[len(recent) - params.move_budget_max]  # これが窓から外れると1回分空く
        retry = max(1, params.move_budget_window_hours - self._hours_since(blocking))
        raise WorldError(
            "MOVE_BUDGET_EXCEEDED",
            f"直近 {params.move_budget_window_hours} 時間に {len(recent)} 回移動済みで、"
            f"移動予算（{params.move_budget_max} 回）を使い切っています。"
            f"あと {retry} 時間は移動できません",
            retry_after_hours=retry,
            moves_in_window=len(recent),
            move_budget_max=params.move_budget_max,
        )

    def _guard_reversal(self, target: str) -> None:
        """直前の移動元へ短時間で戻る往復を拒否する（健康度が低い緊急時は許可）。"""
        p, params = self.plant, self.params
        last = p.last_move
        if last is None or target != last.from_spot:
            return
        if p.health < params.reversal_emergency_health:
            return
        since = self._hours_since(last.at)
        if since < params.reversal_cooldown_hours:
            raise WorldError(
                "RECENT_REVERSAL",
                f"{since}時間前に {last.from_spot}→{last.to_spot} へ移動したばかりです。"
                f"往復はバッテリーの無駄なので、あと {params.reversal_cooldown_hours - since} 時間は"
                f" {target} へ戻れません",
                retry_after_hours=params.reversal_cooldown_hours - since,
                last_move_from=last.from_spot,
                last_move_to=last.to_spot,
            )

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
