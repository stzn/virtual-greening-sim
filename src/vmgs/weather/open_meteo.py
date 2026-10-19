"""Open-Meteo API クライアント（APIキー不要）。"""

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import httpx

from vmgs.weather.models import WeatherRecord

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = "temperature_2m,relative_humidity_2m,shortwave_radiation,wind_speed_10m,precipitation"


def synthetic_precip_prob(precip_mm: float) -> float:
    """過去データには降水確率が無いため、実降水量から「ほぼ当たる予報」を合成する。"""
    if precip_mm >= 1.0:
        return 90.0
    if precip_mm > 0.0:
        return 60.0
    return 5.0


class OpenMeteoClient:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=30.0)

    def fetch(
        self, lat: float, lon: float, start: date, end: date, *, historical: bool = True
    ) -> list[WeatherRecord]:
        params: dict[str, str | float] = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": HOURLY_VARS + ("" if historical else ",precipitation_probability"),
            "timezone": "Asia/Tokyo",
            "wind_speed_unit": "ms",
        }
        resp = self._http.get(ARCHIVE_URL if historical else FORECAST_URL, params=params)
        resp.raise_for_status()
        return parse_hourly(resp.json())


def parse_hourly(payload: Mapping[str, Any]) -> list[WeatherRecord]:
    hourly: Mapping[str, Any] = payload["hourly"]
    times: list[str] = hourly["time"]
    probs: list[float | None] = hourly.get("precipitation_probability") or [None] * len(times)
    records: list[WeatherRecord] = []
    for i, t in enumerate(times):
        precip = float(hourly["precipitation"][i] or 0.0)
        prob = probs[i]
        records.append(
            WeatherRecord(
                time=datetime.fromisoformat(t),
                temperature_c=hourly["temperature_2m"][i],
                humidity_pct=hourly["relative_humidity_2m"][i],
                shortwave_w_m2=hourly["shortwave_radiation"][i] or 0.0,
                wind_ms=hourly["wind_speed_10m"][i],
                precipitation_mm=precip,
                precipitation_prob_pct=float(prob)
                if prob is not None
                else synthetic_precip_prob(precip),
            )
        )
    return records
