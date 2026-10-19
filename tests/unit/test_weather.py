from datetime import date, datetime
from pathlib import Path

import httpx

from vmgs.weather.cache import load_series, save_records
from vmgs.weather.open_meteo import OpenMeteoClient, parse_hourly
from vmgs.weather.providers import SeriesWeather, WeatherUnavailableError

PAYLOAD = {
    "hourly": {
        "time": ["2025-08-01T00:00", "2025-08-01T01:00", "2025-08-01T02:00"],
        "temperature_2m": [28.0, 27.5, 27.0],
        "relative_humidity_2m": [80, 82, 85],
        "shortwave_radiation": [0.0, 0.0, None],
        "wind_speed_10m": [1.2, 1.0, 0.8],
        "precipitation": [0.0, 0.5, 2.0],
    }
}


def test_parse_hourly_は降水確率を合成し欠損日射を0にする() -> None:
    recs = parse_hourly(PAYLOAD)
    assert [r.precipitation_prob_pct for r in recs] == [5.0, 60.0, 90.0]
    assert recs[2].shortwave_w_m2 == 0.0
    assert recs[0].time == datetime(2025, 8, 1, 0, 0)


def test_クライアントはAsiaTokyoと風速m_sを指定する() -> None:
    seen: dict[str, str] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(dict(req.url.params))
        return httpx.Response(200, json=PAYLOAD)

    client = OpenMeteoClient(httpx.Client(transport=httpx.MockTransport(handler)))
    recs = client.fetch(35.6866, 139.7639, date(2025, 8, 1), date(2025, 8, 1))
    assert len(recs) == 3
    assert seen["timezone"] == "Asia/Tokyo"
    assert seen["wind_speed_unit"] == "ms"


def test_キャッシュは保存と再生で同じ内容になる(tmp_path: Path) -> None:
    recs = parse_hourly(PAYLOAD)
    path = tmp_path / "w.json"
    save_records(path, recs, lat=35.6866, lon=139.7639)
    series = load_series(path)
    assert series.at(datetime(2025, 8, 1, 1, 30)) == recs[1]


def test_forecast_は次の時間から始まりデータ末尾で止まる() -> None:
    series = SeriesWeather(parse_hourly(PAYLOAD))
    fc = series.forecast(datetime(2025, 8, 1, 0, 0), 24)
    assert [f.time.hour for f in fc] == [1, 2]


def test_データ外の時刻は例外() -> None:
    series = SeriesWeather(parse_hourly(PAYLOAD))
    try:
        series.at(datetime(2025, 8, 2, 0, 0))
    except WeatherUnavailableError:
        return
    raise AssertionError("例外が出るはず")
