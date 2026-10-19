"""気象データのJSONキャッシュ。オフライン再生と再現性のために使う。"""

import json
from pathlib import Path

from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import SeriesWeather


def save_records(path: Path, records: list[WeatherRecord], *, lat: float, lon: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "latitude": lat,
        "longitude": lon,
        "records": [r.model_dump(mode="json") for r in records],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_records(path: Path) -> list[WeatherRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [WeatherRecord.model_validate(r) for r in payload["records"]]


def load_series(path: Path) -> SeriesWeather:
    return SeriesWeather(load_records(path))
