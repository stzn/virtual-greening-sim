"""stage.json（契約 v1 の Stage）から、地点・時刻ごとの Spot と地点間の距離を作る。

既存の A/B/C の世界（vmgs.world.spots.SPOTS）には触れず、別の入口として使う。
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from vmgs.contracts import SpotKind, Stage, StageSpot
from vmgs.world.spots import Spot

EARTH_RADIUS_M = 6_371_000.0

# stage.json に需要（demand）が 1 件も無いとき（地図アプリの v1 の出力）に使う既定の需要。
# 勤務時間帯（8〜19 時）に全地点で 1.0。集計した需要が来たら使わない。
DEFAULT_DEMAND_HOURS = tuple(range(8, 20))
DEFAULT_DEMAND_WEIGHT = 1.0


@dataclass(frozen=True)
class KindDefaults:
    """地点の種類ごとの、表面温度の補正などの既定値（A/B/C の値に合わせる。docs/thermal.md）。"""

    surface_offset_c: float
    offset_scales_with_sun: bool
    wind_factor: float = 1.0


KIND_DEFAULTS: dict[SpotKind, KindDefaults] = {
    # 歩道: アスファルト（既存の C と同じ。日射に応じて最大 +5℃）
    "sidewalk": KindDefaults(5.0, offset_scales_with_sun=True),
    # 広場: 石・タイル舗装。アスファルトよりやや低い
    "plaza": KindDefaults(3.0, offset_scales_with_sun=True),
    # 公園の縁: 芝・土（既存の A と同じ。補正なし）
    "park_edge": KindDefaults(0.0, offset_scales_with_sun=False),
    # 屋上: 防水シートの加熱が大きく、風も強い
    "rooftop": KindDefaults(6.0, offset_scales_with_sun=True, wind_factor=1.3),
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2 点間の大円距離 [m]。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def load_stage(path: Path) -> Stage:
    return Stage.model_validate(json.loads(path.read_text(encoding="utf-8")))


class StageWorld:
    """地点データの地点の集まり。時刻ごとの日射係数は 1 − shade_fraction。"""

    def __init__(self, stage: Stage) -> None:
        self.stage = stage
        self._spots: dict[str, StageSpot] = {s.id: s for s in stage.spots}
        self._shade = {s.id: {h.hour: h.shade_fraction for h in s.hourly} for s in stage.spots}
        self.uses_default_demand = all(not s.demand for s in stage.spots)
        self._demand = {s.id: {d.hour: d.weight for d in s.demand} for s in stage.spots}
        self._dist: dict[tuple[str, str], float] = {}

    @classmethod
    def from_stage(cls, stage: Stage) -> "StageWorld":
        return cls(stage)

    @property
    def spot_ids(self) -> list[str]:
        return list(self._spots)

    @property
    def hours(self) -> set[int]:
        """全地点に日射遮蔽率がある時刻。"""
        sets = [set(v) for v in self._shade.values()]
        return set.intersection(*sets)

    def stage_spot(self, spot_id: str) -> StageSpot:
        return self._spots[spot_id]

    def same_shelter(self, a: str, b: str) -> bool:
        """地点 a と b が同じ屋根付きの網の上にあり、濡れずに行き来できるか（版 1 の地点データでは常に偽）。"""
        net = self._spots[a].shelter_network
        return net is not None and net == self._spots[b].shelter_network

    def capacity(self, spot_id: str) -> int:
        return self._spots[spot_id].capacity_pods

    def sky_view_factor(self, spot_id: str) -> float:
        return self._spots[spot_id].sky_view_factor

    def spot(self, spot_id: str, hour: int) -> Spot:
        """その時刻の Spot（既存の植物モデル・表面温度の計算をそのまま使えるようにする）。"""
        s = self._spots[spot_id]
        try:
            shade = self._shade[spot_id][hour]
        except KeyError:
            raise ValueError(f"地点 {spot_id} に {hour} 時の日射遮蔽率がありません") from None
        d = KIND_DEFAULTS[s.kind]
        return Spot(
            id=s.id,
            name=s.name,
            solar_factor=1.0 - shade,
            wind_factor=d.wind_factor,
            surface_offset_c=d.surface_offset_c,
            offset_scales_with_sun=d.offset_scales_with_sun,
        )

    def demand(self, spot_id: str, hour: int) -> float:
        if self.uses_default_demand:
            return DEFAULT_DEMAND_WEIGHT if hour in DEFAULT_DEMAND_HOURS else 0.0
        return self._demand[spot_id].get(hour, 0.0)

    def distance_m(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b) if a < b else (b, a)
        if key not in self._dist:
            sa, sb = self._spots[a], self._spots[b]
            self._dist[key] = haversine_m(sa.lat, sa.lon, sb.lat, sb.lon)
        return self._dist[key]
