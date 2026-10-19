"""仮想スポット定義（基準座標: 東京都大手町）。"""

from dataclasses import dataclass

from vmgs.world.params import Params

OTEMACHI_LAT = 35.6866
OTEMACHI_LON = 139.7639


@dataclass(frozen=True)
class Spot:
    id: str
    name: str
    solar_factor: float
    wind_factor: float
    surface_offset_c: float
    offset_scales_with_sun: bool = False  # True: 日射に応じて地表温度補正が増える

    def effective_radiation(self, shortwave: float) -> float:
        return shortwave * self.solar_factor

    def effective_temp(self, temp_c: float, shortwave: float, params: Params) -> float:
        offset = self.surface_offset_c
        if self.offset_scales_with_sun:
            offset *= min(1.0, self.effective_radiation(shortwave) / params.sun_full_w_m2)
        return temp_c + offset


SPOTS: dict[str, Spot] = {
    "A": Spot("A", "芝生広場", solar_factor=1.0, wind_factor=1.0, surface_offset_c=0.0),
    "B": Spot("B", "ビルの谷間", solar_factor=0.2, wind_factor=1.5, surface_offset_c=-3.0),
    "C": Spot("C", "アスファルト歩道", 1.0, 1.0, surface_offset_c=5.0, offset_scales_with_sun=True),
}

# スポット間の仮想距離 [m]
_DISTANCES_M: dict[frozenset[str], float] = {
    frozenset("AB"): 200.0,
    frozenset("AC"): 150.0,
    frozenset("BC"): 300.0,
}


def distance_m(a: str, b: str) -> float:
    return _DISTANCES_M[frozenset((a, b))]


def move_cost_pct(a: str, b: str, params: Params) -> float:
    """距離に比例して 5〜10% にマップ。"""
    d = distance_m(a, b)
    span = params.move_dist_max_m - params.move_dist_min_m
    ratio = (d - params.move_dist_min_m) / span
    return params.move_cost_min_pct + ratio * (params.move_cost_max_pct - params.move_cost_min_pct)
