"""地点・時刻ごとの PET。ポッドがあるとき（その木陰に座る人）と、ないときの 2 通り。"""

import datetime as dt

from vmgs.fleet.stage_world import StageWorld
from vmgs.thermal import RadiationInput, body_wind_ms, mean_radiant_temp, pet, solar_elevation_deg
from vmgs.weather.models import WeatherRecord
from vmgs.world.params import Params

# ポッドの樹冠の下: 直達日射の透過率と、空が見える割合の減り方（docs/thermal.md）
POD_SUN_TRANSMITTANCE = 0.2
POD_SKY_VIEW_KEEP = 0.3


class PetTable:
    """PET を必要になったときに計算して覚えておく（255 地点 × 24 時 × 2 通りで数秒）。"""

    def __init__(
        self,
        world: StageWorld,
        weather_by_hour: dict[int, WeatherRecord],
        *,
        params: Params | None = None,
    ) -> None:
        self.world = world
        self.weather = weather_by_hour
        self.params = params or Params()
        self.date: dt.date = world.stage.date
        self._cache: dict[tuple[str, int, bool], float] = {}

    def pet(self, spot_id: str, hour: int, *, covered: bool) -> float:
        key = (spot_id, hour, covered)
        if key not in self._cache:
            self._cache[key] = self._compute(spot_id, hour, covered)
        return self._cache[key]

    def relief(self, spot_id: str, hour: int) -> float:
        """ポッドを置くと下がる PET [℃]（0 未満にはしない）。"""
        return max(
            0.0, self.pet(spot_id, hour, covered=False) - self.pet(spot_id, hour, covered=True)
        )

    def _compute(self, spot_id: str, hour: int, covered: bool) -> float:
        w = self.weather[hour]
        spot = self.world.spot(spot_id, hour)
        ss = self.world.stage_spot(spot_id)
        sun = spot.solar_factor * (POD_SUN_TRANSMITTANCE if covered else 1.0)
        svf = self.world.sky_view_factor(spot_id) * (POD_SKY_VIEW_KEEP if covered else 1.0)
        # 地表面の補正は、その場の日射係数で決まる（樹冠の下の地面は日影になる）
        ground = spot.surface_offset_c
        if spot.offset_scales_with_sun:
            ground *= min(1.0, w.shortwave_w_m2 * sun / self.params.sun_full_w_m2)
        # 時刻 h は h:00〜h:59 の 1 時間を表す。太陽高度はその中央で見る
        elevation = solar_elevation_deg(ss.lat, ss.lon, self.date, hour + 0.5)
        mrt = mean_radiant_temp(
            RadiationInput(
                temp_c=w.temperature_c,
                humidity_pct=w.humidity_pct,
                shortwave_w_m2=w.shortwave_w_m2,
                solar_elevation_deg=elevation,
                sun_factor=sun,
                sky_view_factor=svf,
                ground_offset_c=ground,
            )
        )
        return round(
            pet(
                temp_c=w.temperature_c,
                mrt_c=mrt,
                wind_ms=body_wind_ms(w.wind_ms * spot.wind_factor),
                humidity_pct=w.humidity_pct,
            ),
            2,
        )
