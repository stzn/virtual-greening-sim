"""数理モデルの係数。調整は全てここで行う（根拠は docs/model.md）。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    # 蒸散 [%/h] = base + rad*(R/1000) + temp*max(0,T-20) + wind*U - hum*(RH/100)
    et_base: float = 0.3
    et_rad: float = 1.5
    et_temp: float = 0.15
    et_wind: float = 0.2
    et_hum: float = 0.5
    rain_gain_per_mm: float = 3.0  # 降水1mmあたりの水分増加 [%]
    irrigation_gain_per_ml: float = 0.02  # 散水1mlあたりの水分増加 [%]

    # 健康度
    moisture_ok_low: float = 30.0
    moisture_ok_high: float = 70.0
    moisture_wet: float = 85.0
    health_recovery: float = 1.0
    health_dry_rate: float = 0.3
    health_wet_rate: float = 0.2
    heat_stress_temp_c: float = 35.0
    health_heat_rate: float = 0.5

    # バッテリー
    solar_charge_per_kw: float = 2.0  # 有効日射 1000W/m² で +2%/h
    idle_drain_pct: float = 0.5
    move_cost_min_pct: float = 5.0
    move_cost_max_pct: float = 10.0
    move_dist_min_m: float = 150.0
    move_dist_max_m: float = 300.0

    # 往復移動の抑制
    reversal_cooldown_hours: int = 2  # 直前の移動元へ戻れない時間
    reversal_emergency_health: float = 50.0  # 健康度がこれ未満なら緊急として往復を許可

    # 暑熱緩和
    cooling_per_ml: float = 0.004  # 散水1mlあたりの冷却 [℃]
    cooling_cap_c: float = 3.0
    heat_threshold_c: float = 30.0  # これを超えた分が「冷却需要」
    sun_full_w_m2: float = 600.0  # 地表加熱が最大になる日射量

    initial_moisture: float = 60.0
    initial_tank_ml: int = 10_000
