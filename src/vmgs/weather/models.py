"""気象データのモデル。時刻は Open-Meteo の timezone=Asia/Tokyo に合わせた naive な現地時刻。"""

from datetime import datetime

from pydantic import BaseModel


class WeatherRecord(BaseModel):
    """1時間分の気象。"""

    time: datetime
    temperature_c: float
    humidity_pct: float
    shortwave_w_m2: float
    wind_ms: float
    precipitation_mm: float
    # 降水確率。過去データには無いため、取得時に実降水から合成する（open_meteo.py 参照）
    precipitation_prob_pct: float | None = None
