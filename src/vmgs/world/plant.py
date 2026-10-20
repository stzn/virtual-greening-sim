"""植物ユニットの状態と生理モデル。"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from vmgs.weather.models import WeatherRecord
from vmgs.world.params import Params
from vmgs.world.spots import Spot


class UnitStatus(StrEnum):
    ACTIVE = "ACTIVE"
    STRANDED = "STRANDED"  # バッテリー切れで移動不能
    DEAD = "DEAD"  # 枯死


@dataclass(frozen=True)
class LastMove:
    """直前の移動（往復の検出と、LLMへの観測に使う）。"""

    from_spot: str
    to_spot: str
    at: datetime


@dataclass
class PlantUnit:
    spot_id: str = "A"
    moisture: float = 60.0
    health: float = 100.0
    water_tank_ml: int = 10_000
    battery_pct: float = 100.0
    status: UnitStatus = UnitStatus.ACTIVE
    heat_mitigation_score: float = 0.0
    last_move: LastMove | None = None
    move_times: list[datetime] = field(default_factory=list)  # 移動した時刻（予算の集計用）


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def evapotranspiration(w: WeatherRecord, spot: Spot, temp_eff: float, params: Params) -> float:
    """毎時の蒸散による水分減少 [%/h]（負にはならない）。"""
    radiation = spot.effective_radiation(w.shortwave_w_m2)
    wind = w.wind_ms * spot.wind_factor
    et = (
        params.et_base
        + params.et_rad * radiation / 1000.0
        + params.et_temp * max(0.0, temp_eff - 20.0)
        + params.et_wind * wind
        - params.et_hum * w.humidity_pct / 100.0
    )
    return max(0.0, et)


def health_delta(moisture: float, temp_eff: float, params: Params) -> float:
    delta = 0.0
    if params.moisture_ok_low <= moisture <= params.moisture_ok_high:
        delta += params.health_recovery
    elif moisture < params.moisture_ok_low:
        delta -= (params.moisture_ok_low - moisture) * params.health_dry_rate
    elif moisture > params.moisture_wet:
        delta -= (moisture - params.moisture_wet) * params.health_wet_rate
    if temp_eff > params.heat_stress_temp_c:
        delta -= (temp_eff - params.heat_stress_temp_c) * params.health_heat_rate
    return delta
