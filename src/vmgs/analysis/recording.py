"""1 回の計測（セッション）の RR 間隔と、その条件・環境。"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# 2×2 の条件: 名前 → (歩行の有無, オフィスからの離脱の有無)
CONDITIONS: dict[str, tuple[bool, bool]] = {
    "sit-stay": (False, False),
    "walk-stay": (True, False),
    "sit-leave": (False, True),
    "walk-leave": (True, True),
}

# 追跡の区間に記録する出来事（その窓は解析から除く）
EVENTS = ("talk", "phone", "food", "away")


@dataclass(frozen=True, eq=False)
class Recording:
    """1 セッション分の計測。時刻は接触開始（条件の切り替え）を 0 秒とする。

    beat_times_s[k] は拍 k の時刻で、rr_ms[k] はその拍で終わる RR 間隔。
    """

    session: str
    subject: str
    condition: str
    green: float  # 緑の量（0〜1。緑視率などの割合）
    pet_c: float  # 体感温度 PET [°C]（シミュレーター側のフェーズ 2 で計算したものを入力として受け取る）
    noise_db: float  # 騒音 [dB(A)]
    beat_times_s: FloatArray
    rr_ms: FloatArray
    # 追跡（接触の後、pilot-protocol.md 4.2）。無ければ None（接触中の解析だけに使える記録）
    contact_end_s: float | None = None
    track_start_s: float | None = None
    track_end_s: float | None = None
    events: tuple[tuple[float, str], ...] = ()  # (時刻 [秒], 出来事)。時刻の順

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"知らない条件: {self.condition}（{sorted(CONDITIONS)} のどれか）")
        if self.beat_times_s.shape != self.rr_ms.shape:
            raise ValueError("beat_times_s と rr_ms の長さが違う")
        track = (self.contact_end_s, self.track_start_s, self.track_end_s)
        if any(t is None for t in track) and any(t is not None for t in track):
            raise ValueError("contact_end・track_start・track_end はそろって書くか、どれも書かない")
        if self.contact_end_s is not None:
            assert self.track_start_s is not None and self.track_end_s is not None
            if not 0.0 <= self.contact_end_s <= self.track_start_s <= self.track_end_s:
                raise ValueError(
                    "時刻の順が違う（接触の始め ≦ 接触の終わり ≦ 追跡の始め ≦ 追跡の終わり）"
                )
        unknown = sorted({e for _, e in self.events} - set(EVENTS))
        if unknown:
            raise ValueError(
                f"知らない出来事: {', '.join(unknown)}（{', '.join(EVENTS)} のどれか）"
            )

    @property
    def tracked(self) -> bool:
        """追跡の区間を持つか。"""
        return self.contact_end_s is not None

    @property
    def walk(self) -> bool:
        return CONDITIONS[self.condition][0]

    @property
    def leave(self) -> bool:
        return CONDITIONS[self.condition][1]
