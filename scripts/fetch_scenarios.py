"""Open-Meteoの過去データから猛暑日・雨天日・平常日を選び、data/weather/ にキャッシュする。

使い方: uv run python scripts/fetch_scenarios.py
"""

from collections import defaultdict
from datetime import date
from pathlib import Path

from vmgs.weather.cache import save_records
from vmgs.weather.models import WeatherRecord
from vmgs.weather.open_meteo import OpenMeteoClient
from vmgs.world.spots import OTEMACHI_LAT, OTEMACHI_LON

OUT = Path("data/weather")


def main() -> None:
    records = OpenMeteoClient().fetch(
        OTEMACHI_LAT, OTEMACHI_LON, date(2025, 6, 1), date(2025, 9, 30), historical=True
    )
    by_day: dict[date, list[WeatherRecord]] = defaultdict(list)
    for r in records:
        by_day[r.time.date()].append(r)
    days = {d: rs for d, rs in by_day.items() if len(rs) == 24}

    def tmax(d: date) -> float:
        return max(r.temperature_c for r in days[d])

    def rain(d: date) -> float:
        return sum(r.precipitation_mm for r in days[d])

    def sun(d: date) -> float:
        return sum(r.shortwave_w_m2 for r in days[d])

    heat = max(days, key=tmax)
    rainy = max((d for d in days if tmax(d) < 30), key=rain)
    dry = [d for d in days if rain(d) == 0.0 and sun(d) > 5000]
    median_tmax = sorted(tmax(d) for d in dry)[len(dry) // 2]
    normal = min(dry, key=lambda d: abs(tmax(d) - median_tmax))

    for name, d in (("heatwave", heat), ("rainy", rainy), ("normal", normal)):
        save_records(OUT / f"{name}.json", days[d], lat=OTEMACHI_LAT, lon=OTEMACHI_LON)
        print(f"{name}: {d}  最高気温 {tmax(d):.1f}℃  降水 {rain(d):.1f}mm  日射積算 {sun(d):.0f}")


if __name__ == "__main__":
    main()
