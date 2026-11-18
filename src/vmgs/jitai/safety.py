"""段階 4: 天候の安全判定（雨 → 目的地の樹冠の下の PET → 日向の道中の PET）。

閾値は、高温多湿の地域向けの PET の温熱感の区分（Lin & Matzarakis 2008、台湾）に合わせる。
中欧の区分（Matzarakis & Mayer 1996）より約 4℃高く、東京の夏の暑熱順化に近いと考えた。

| PET [℃] | 温熱感 | 生理的な負荷 |
| --- | --- | --- |
| 34〜38 | warm | 中程度の暑熱ストレス |
| 38〜42 | hot | 強い暑熱ストレス |
| 42 以上 | very hot | 極度の暑熱ストレス |

- 目的地（樹冠の下で 10 分座る）: 回復が目的なので、強い暑熱ストレス（38℃）に入る前で止める。
  フェーズ 5 の合成モデルでは PET 1℃あたり ln(HF) の到達値が 0.04 下がる（暑さは回復を妨げる）。
- 道中（日向を片道 5 分以内歩く）: 短時間なので強い暑熱ストレスまでは許し、極度（42℃）で止める。
- 雨: 行きから帰りまでのどの時刻も 1 mm/h 未満（傘が要る雨では見送る）。屋外のポッドで 10 分座って
  休むので、弱い雨でも勧めない。降水確率ではなく、その時刻の降水量で見る。
  ポッドに雨よけ（屋根）があるときは、座る間は雨を問わず、行き帰りを傘で歩ける 5 mm/h 未満なら勧める
  （仮の値。「やや強い雨」10 mm/h の手前とした）。
根拠と限界は docs/jitai.md。
"""

from collections.abc import Iterable, Mapping
from typing import Literal, Protocol

DEST_PET_MAX_C = 38.0
ROUTE_PET_MAX_C = 42.0
RAIN_MAX_MM_H = 1.0
RAIN_COVER_MAX_MM_H = 5.0

PetReason = Literal["pet_destination", "pet_route"]


class PetFn(Protocol):
    """地点・時の PET [℃]。covered はポッドの樹冠の下か（vmgs.fleet.comfort.PetTable.pet）。"""

    def __call__(self, spot_id: str, hour: int, *, covered: bool) -> float: ...


def pet_check(
    pet: PetFn, dest_spot: str, route_spots: Iterable[str], hours: Iterable[int]
) -> tuple[PetReason | None, float, float]:
    """(見送る理由 / None, 目的地の PET の最大, 道中の PET の最大) を返す。目的地を先に見る。"""
    hours = list(hours)
    dest = max(pet(dest_spot, h, covered=True) for h in hours)
    route = max(pet(s, h, covered=False) for s in route_spots for h in hours)
    if dest >= DEST_PET_MAX_C:
        return "pet_destination", dest, route
    if route >= ROUTE_PET_MAX_C:
        return "pet_route", dest, route
    return None, dest, route


def rain_limit(*, rain_cover: bool) -> float:
    """見送る降水量の基準 [mm/h]。ポッドに雨よけがあれば、行き帰りを傘で歩ける強さまで許す。"""
    return RAIN_COVER_MAX_MM_H if rain_cover else RAIN_MAX_MM_H


def rain_check(rain_mm: Mapping[int, float], hours: Iterable[int]) -> float:
    """行きから帰りまでの各時の降水量 [mm/h] の最大。気象に無い時刻は雨なしとみなす。"""
    return max((rain_mm.get(h, 0.0) for h in hours), default=0.0)
