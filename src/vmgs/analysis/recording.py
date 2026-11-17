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


@dataclass(frozen=True, eq=False)
class Recording:
    """1 セッション分の計測。時刻は接触開始（条件の切り替え）を 0 秒とする。

    beat_times_s[k] は拍 k の時刻で、rr_ms[k] はその拍で終わる RR 間隔。
    """

    session: str
    subject: str
    condition: str
    green: float  # 緑の量（0〜1。緑視率などの割合）
    pet_c: float  # 体感温度 PET [°C]（役者側のフェーズ 2 で計算したものを入力として受け取る）
    noise_db: float  # 騒音 [dB(A)]
    beat_times_s: FloatArray
    rr_ms: FloatArray

    def __post_init__(self) -> None:
        if self.condition not in CONDITIONS:
            raise ValueError(f"知らない条件: {self.condition}（{sorted(CONDITIONS)} のどれか）")
        if self.beat_times_s.shape != self.rr_ms.shape:
            raise ValueError("beat_times_s と rr_ms の長さが違う")

    @property
    def walk(self) -> bool:
        return CONDITIONS[self.condition][0]

    @property
    def leave(self) -> bool:
        return CONDITIONS[self.condition][1]
