"""World が参照する気象プロバイダ。実装を差し替えてもWorldは変わらない。"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from vmgs.weather.models import WeatherRecord


class WeatherUnavailableError(Exception):
    """要求された時刻の気象データが無い。"""


def floor_hour(t: datetime) -> datetime:
    return t.replace(minute=0, second=0, microsecond=0)


class WeatherProvider(Protocol):
    def at(self, t: datetime) -> WeatherRecord: ...

    def forecast(self, t: datetime, hours: int) -> list[WeatherRecord]:
        """t の次の時間から最大 hours 件。データ末尾を超える分は含めない。"""
        ...


class SeriesWeather:
    """時系列データ（キャッシュJSONや取得結果）を再生するプロバイダ。"""

    def __init__(self, records: list[WeatherRecord]) -> None:
        self._by_hour = {floor_hour(r.time): r for r in records}

    def at(self, t: datetime) -> WeatherRecord:
        try:
            return self._by_hour[floor_hour(t)]
        except KeyError:
            raise WeatherUnavailableError(f"{t.isoformat()} の気象データがありません") from None

    def forecast(self, t: datetime, hours: int) -> list[WeatherRecord]:
        base = floor_hour(t)
        out: list[WeatherRecord] = []
        for i in range(1, hours + 1):
            rec = self._by_hour.get(base + timedelta(hours=i))
            if rec is None:
                break
            out.append(rec)
        return out


@dataclass
class ConstantWeather:
    """常に同じ気象を返す（テスト・単純シナリオ用）。値は後から書き換えられる。"""

    temperature_c: float = 25.0
    humidity_pct: float = 50.0
    shortwave_w_m2: float = 500.0
    wind_ms: float = 1.0
    precipitation_mm: float = 0.0
    precipitation_prob_pct: float | None = None

    def at(self, t: datetime) -> WeatherRecord:
        return WeatherRecord(
            time=floor_hour(t),
            temperature_c=self.temperature_c,
            humidity_pct=self.humidity_pct,
            shortwave_w_m2=self.shortwave_w_m2,
            wind_ms=self.wind_ms,
            precipitation_mm=self.precipitation_mm,
            precipitation_prob_pct=self.precipitation_prob_pct,
        )

    def forecast(self, t: datetime, hours: int) -> list[WeatherRecord]:
        return [self.at(floor_hour(t) + timedelta(hours=i)) for i in range(1, hours + 1)]
