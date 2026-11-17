"""人の温熱快適度。PET（生理学的等価温度）と、その入力の平均放射温度（Tmrt）の近似。

- PET は pythermalcomfort の `pet_steady`（Höppe の MEMI に基づく定常計算）で求める。
- Tmrt は、毎時の全天日射（Open-Meteo の shortwave_radiation）・日射係数（1 − 建物の遮蔽率）・
  天空率・地表面の温度補正から、人体を上下 2 半球で見る簡易な放射収支で近似する。

式・係数の根拠と限界は docs/thermal.md。フェーズ 5（解析基盤）も共変量の計算に使うため、
世界（vmgs.world）や配車（vmgs.fleet）に依存しない独立したモジュールにしている。
"""

import datetime as dt
import math
from dataclasses import dataclass

from pythermalcomfort.models import pet_steady

STEFAN_BOLTZMANN = 5.67e-8  # [W/m²/K⁴]
SOLAR_CONSTANT_W_M2 = 1361.0
JST_MERIDIAN_DEG = 135.0
ABSORB_SHORTWAVE = 0.7  # 人体の短波吸収率
EMISSIVITY_BODY = 0.97  # 人体の長波吸収率（=放射率）
GROUND_ALBEDO = 0.15  # 地表面の反射率（アスファルト・舗装・芝の間の値）
MIN_SUN_ELEVATION_DEG = 2.0  # これ未満は直達日射なし（全て散乱として扱う）
MAX_DIRECT_NORMAL_W_M2 = 1000.0

# 気象の風速は高さ 10 m。PET は人体の重心の高さ（1.1 m）の風速を使う。
# 対数則 ln(1.1/z0)/ln(10/z0)、粗度 z0=0.1 m で 0.52。
WIND_10M_TO_BODY = 0.52
MIN_BODY_WIND_MS = 0.1

# PET の人体の条件（pythermalcomfort の例と同じ: 代謝 1.37 met、夏の軽装 0.5 clo、座位）
PET_MET = 1.37
PET_CLO = 0.5


@dataclass(frozen=True)
class RadiationInput:
    """1 時間・1 地点の放射環境。"""

    temp_c: float
    humidity_pct: float
    shortwave_w_m2: float  # 全天日射（水平面）
    solar_elevation_deg: float
    sun_factor: float  # 日射係数（1 − 建物の遮蔽率）。0=日影、1=日向
    sky_view_factor: float
    ground_offset_c: float = 0.0  # 地表面の温度 − 気温（アスファルトの加熱など）


def solar_elevation_deg(lat: float, lon: float, date: dt.date, hour_jst: float) -> float:
    """太陽高度 [°]（簡易式。誤差は 1° 程度）。hour_jst は日本標準時の時刻（例: 12.5）。"""
    n = date.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(math.radians(360.0 / 365.0 * (284 + n)))
    b = math.radians(360.0 / 365.0 * (n - 81))
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    solar_time = hour_jst + (4.0 * (lon - JST_MERIDIAN_DEG) + eot_min) / 60.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))
    phi = math.radians(lat)
    sin_h = math.sin(phi) * math.sin(decl) + math.cos(phi) * math.cos(decl) * math.cos(hour_angle)
    return math.degrees(math.asin(max(-1.0, min(1.0, sin_h))))


def diffuse_fraction(shortwave_w_m2: float, solar_elevation_deg: float) -> float:
    """全天日射のうち散乱の割合（Erbs のモデル）。太陽が低いときは全て散乱。"""
    if solar_elevation_deg < MIN_SUN_ELEVATION_DEG or shortwave_w_m2 <= 0.0:
        return 1.0
    kt = shortwave_w_m2 / (SOLAR_CONSTANT_W_M2 * math.sin(math.radians(solar_elevation_deg)))
    kt = min(kt, 1.0)
    if kt <= 0.22:
        return 1.0 - 0.09 * kt
    if kt <= 0.8:
        return 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    return 0.165


def _projected_area_factor(solar_elevation_deg: float) -> float:
    """直達日射を受ける人体の投影面積率（立位の円柱近似。Fanger）。"""
    h = solar_elevation_deg
    return 0.308 * math.cos(math.radians(h * (1.0 - h * h / 48402.0)))


def _sky_emissivity(temp_c: float, humidity_pct: float) -> float:
    """晴天の空の実効放射率（Brutsaert）。"""
    vapor_hpa = humidity_pct / 100.0 * 6.112 * math.exp(17.67 * temp_c / (temp_c + 243.5))
    return min(1.0, 1.24 * float((vapor_hpa / (temp_c + 273.15)) ** (1.0 / 7.0)))


def mean_radiant_temp(r: RadiationInput) -> float:
    """平均放射温度 [℃]。人体が上半球（空と建物の壁）と下半球（地面）から受ける放射の和から求める。

    - 短波: 直達（太陽が見える割合 sun_factor）＋空の散乱（天空率）＋地面の反射
    - 長波: 空（天空率 × 空の放射率）、建物の壁（1 − 天空率。気温と同じ温度）、地面（気温 + 補正）
    """
    kd = diffuse_fraction(r.shortwave_w_m2, r.solar_elevation_deg)
    diffuse = r.shortwave_w_m2 * kd
    direct_h = r.shortwave_w_m2 - diffuse
    direct_normal = 0.0
    if direct_h > 0.0:
        sin_h = math.sin(math.radians(r.solar_elevation_deg))
        direct_normal = min(MAX_DIRECT_NORMAL_W_M2, direct_h / sin_h)
    svf, s = r.sky_view_factor, r.sun_factor

    shortwave = (
        _projected_area_factor(r.solar_elevation_deg) * direct_normal * s
        + 0.5 * diffuse * svf
        + 0.5 * GROUND_ALBEDO * (direct_h * s + diffuse * svf)
    )
    ta_k = r.temp_c + 273.15
    tg_k = ta_k + r.ground_offset_c
    black_air = STEFAN_BOLTZMANN * ta_k**4
    longwave = (
        0.5
        * (svf * _sky_emissivity(r.temp_c, r.humidity_pct) * black_air + (1.0 - svf) * black_air)
        + 0.5 * STEFAN_BOLTZMANN * tg_k**4
    )
    absorbed = ABSORB_SHORTWAVE * shortwave + EMISSIVITY_BODY * longwave
    return float((absorbed / (EMISSIVITY_BODY * STEFAN_BOLTZMANN)) ** 0.25 - 273.15)


def body_wind_ms(wind_10m_ms: float) -> float:
    """高さ 10 m の風速を、人体の高さ（1.1 m）の風速に換算する。"""
    return max(MIN_BODY_WIND_MS, wind_10m_ms * WIND_10M_TO_BODY)


def pet(*, temp_c: float, mrt_c: float, wind_ms: float, humidity_pct: float) -> float:
    """PET [℃]。wind_ms は人体の高さの風速（body_wind_ms で換算した値）。"""
    result = pet_steady(tdb=temp_c, tr=mrt_c, v=wind_ms, rh=humidity_pct, met=PET_MET, clo=PET_CLO)
    return float(result.pet)
