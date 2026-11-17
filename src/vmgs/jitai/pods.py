"""段階 3: 徒歩圏のポッドの空き（dispatch.json の時刻別の位置と、席の予約）。"""

import math
from dataclasses import dataclass, field

from vmgs.contracts import Dispatch
from vmgs.fleet.stage_world import StageWorld, haversine_m

# 徒歩: 80 m/分（不動産の表示に関する公正競争規約の徒歩所要時間の基準）。
# 徒歩圏は片道 5 分 = 400 m（休憩の往復が 10 分に収まる距離）
WALK_SPEED_M_PER_MIN = 80.0
MAX_WALK_MIN = 5
WALK_RADIUS_M = WALK_SPEED_M_PER_MIN * MAX_WALK_MIN
# ポッドでの滞在。合成の R(t)（フェーズ 5）の時定数 120 秒で 95% に達する約 6 分に余裕を足す
DWELL_MIN = 10
# 1 台のポッドの樹冠の下に同時に座れる人数（仮定）
POD_SEATS = 2


def walk_minutes(distance_m: float) -> int:
    return math.ceil(distance_m / WALK_SPEED_M_PER_MIN)


@dataclass(frozen=True)
class Candidate:
    pod_id: str
    spot_id: str
    walk_m: float


class PodSchedule:
    """配車の時刻表（ポッド → 時 → 地点）。時刻表に無い時は、そのポッドは使えない。"""

    def __init__(self, dispatch: Dispatch) -> None:
        self._at = {p.id: {s.hour: s.spot_id for s in p.timeline} for p in dispatch.pods}

    def spot_at(self, pod_id: str, hour: int) -> str | None:
        return self._at[pod_id].get(hour)

    def nearby(self, world: StageWorld, lat: float, lon: float, hour: int) -> list[Candidate]:
        """その時に徒歩圏（WALK_RADIUS_M 以内）にいるポッド。近い順。"""
        found = []
        for pod_id in self._at:
            spot_id = self.spot_at(pod_id, hour)
            if spot_id is None:
                continue
            s = world.stage_spot(spot_id)
            d = haversine_m(lat, lon, s.lat, s.lon)
            if d <= WALK_RADIUS_M:
                found.append(Candidate(pod_id, spot_id, round(d, 1)))
        return sorted(found, key=lambda c: (c.walk_m, c.pod_id))

    def stays(self, pod_id: str, spot_id: str, start_min: int, end_min: int) -> bool:
        """[start_min, end_min) の間ずっと spot_id にいるか（時刻表は 1 時間単位）。"""
        last_hour = (end_min - 1) // 60
        return all(
            self.spot_at(pod_id, h) == spot_id for h in range(start_min // 60, last_hour + 1)
        )


@dataclass
class Occupancy:
    """ポッドの席の予約（[start, end) 分）。"""

    seats: int = POD_SEATS
    _bookings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def free(self, pod_id: str, start_min: int, end_min: int) -> bool:
        overlap = sum(1 for s, e in self._bookings.get(pod_id, []) if s < end_min and start_min < e)
        return overlap < self.seats

    def book(self, pod_id: str, start_min: int, end_min: int) -> None:
        self._bookings.setdefault(pod_id, []).append((start_min, end_min))
