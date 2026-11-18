"""PET と平均放射温度の近似（vmgs.thermal）。"""

import datetime as dt

import pytest

from vmgs.thermal import (
    RadiationInput,
    globe_mrt_c,
    mean_radiant_temp,
    pet,
    solar_elevation_deg,
)

LAT, LON = 35.6866, 139.7639
DAY = dt.date(2025, 8, 30)


def _rad(**kw: float) -> RadiationInput:
    base: dict[str, float] = {
        "temp_c": 34.0,
        "humidity_pct": 50.0,
        "shortwave_w_m2": 800.0,
        "solar_elevation_deg": 60.0,
        "sun_factor": 1.0,
        "sky_view_factor": 0.5,
        "ground_offset_c": 0.0,
    }
    base.update(kw)
    return RadiationInput(**base)


def test_solar_elevation_is_high_at_noon_and_negative_at_midnight() -> None:
    noon = solar_elevation_deg(LAT, LON, DAY, 11.75)
    assert 55.0 < noon < 70.0  # 8 月末の東京の南中高度はおよそ 63°
    assert solar_elevation_deg(LAT, LON, DAY, 0.5) < 0.0


def test_mrt_without_sun_is_close_to_air_temp() -> None:
    night = _rad(shortwave_w_m2=0.0, solar_elevation_deg=-30.0, temp_c=28.0)
    assert abs(mean_radiant_temp(night) - 28.0) < 5.0


def test_mrt_in_sun_is_far_above_air_temp() -> None:
    assert mean_radiant_temp(_rad()) > 34.0 + 15.0


def test_shade_and_low_sky_view_lower_mrt() -> None:
    sun = mean_radiant_temp(_rad(sun_factor=1.0))
    shade = mean_radiant_temp(_rad(sun_factor=0.0))
    assert shade < sun
    # 天空率が下がると空からの散乱日射が減る（建物の壁は気温として数える）
    assert mean_radiant_temp(_rad(sun_factor=0.0, sky_view_factor=0.1)) < shade


def test_hot_ground_raises_mrt() -> None:
    assert mean_radiant_temp(_rad(ground_offset_c=5.0)) > mean_radiant_temp(_rad())


def test_pet_equals_air_temp_reference_room() -> None:
    # PET の定義: 室内基準（Tmrt=Ta、風 0.1 m/s、湿度 50%）では PET はほぼ気温に等しい
    assert pet(temp_c=25.0, mrt_c=25.0, wind_ms=0.1, humidity_pct=50.0) == pytest.approx(
        25.0, abs=2.0
    )


def test_pet_rises_with_mrt() -> None:
    low = pet(temp_c=34.0, mrt_c=34.0, wind_ms=1.0, humidity_pct=50.0)
    high = pet(temp_c=34.0, mrt_c=60.0, wind_ms=1.0, humidity_pct=50.0)
    assert high > low + 5.0


def test_黒球温度が気温と同じなら_MRTは気温と同じ() -> None:
    assert globe_mrt_c(ta_c=30.0, globe_c=30.0, wind_ms=1.0) == pytest.approx(30.0)


def test_黒球温度が気温より高ければ_MRTは黒球温度より高い() -> None:
    # 風で黒球が冷やされる分、放射は黒球温度より強い
    assert globe_mrt_c(ta_c=30.0, globe_c=35.0, wind_ms=1.0) > 35.0


def test_黒球からのMRTは_ISO7726の強制対流の式の手計算と合う() -> None:
    # 直径 0.15 m・放射率 0.95、気温 30・黒球 35 ℃・風 1 m/s:
    # [(308.15)^4 + 1.1e8 × 1^0.6 / (0.95 × 0.15^0.4) × 5]^0.25 − 273.15 ≈ 45.1 ℃
    assert globe_mrt_c(ta_c=30.0, globe_c=35.0, wind_ms=1.0) == pytest.approx(45.1, abs=0.2)


def test_風が弱すぎるときは_下限の風速で計算する() -> None:
    assert globe_mrt_c(ta_c=30.0, globe_c=35.0, wind_ms=0.0) == pytest.approx(
        globe_mrt_c(ta_c=30.0, globe_c=35.0, wind_ms=0.1)
    )
