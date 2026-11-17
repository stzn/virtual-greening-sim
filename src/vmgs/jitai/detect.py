"""段階 1: 日内変動を補正した過緊張の検知。

個人の過去の勤務日の ln(HF) に、24 時間周期の余弦（cosinor）を当てはめて基準の日内変動を求め、
当日の値との差を個人の揺らぎ（残差の頑健な SD）で割った z が、直近 SUSTAIN_WINDOWS 窓
（15 分）続けて Z_THRESHOLD 以下なら過緊張とみなす。過去の値だけを使う（その場で判定できる）。

比較のため、日内変動を補正しない基準（個人の中央値と SD だけ）も持つ。根拠は docs/jitai.md。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vmgs.analysis.recording import FloatArray
from vmgs.jitai.workers import Worker, time_grid_min

BoolArray = npt.NDArray[np.bool_]
TimeArray = npt.NDArray[np.int64]

Z_THRESHOLD = -2.0
SUSTAIN_WINDOWS = 3  # 5 分 × 3 = 15 分
EPISODE_GRACE_MIN = 15  # エピソードの終わりからこの分までの検知は、そのエピソードの検知に数える
MAD_TO_SD = 1.4826
OMEGA = 2 * np.pi / (24 * 60)  # [rad/分]


@dataclass(frozen=True)
class DayProfile:
    """個人の基準: mean + a·cos(ωt) + b·sin(ωt)、揺らぎの SD。"""

    mean: float
    a: float
    b: float
    sd: float

    def predict(self, t_min: TimeArray) -> FloatArray:
        out: FloatArray = (
            self.mean + self.a * np.cos(OMEGA * t_min) + self.b * np.sin(OMEGA * t_min)
        )
        return out


def _robust_sd(resid: FloatArray) -> float:
    # エピソード（片側の外れ）に引っ張られないように MAD から求める
    return float(MAD_TO_SD * np.median(np.abs(resid - np.median(resid))))


def _flatten(t_min: TimeArray, days: FloatArray) -> tuple[TimeArray, FloatArray]:
    tt = np.tile(t_min, days.shape[0])
    x = days.reshape(-1)
    ok = ~np.isnan(x)
    return tt[ok], x[ok]


def fit_circadian(t_min: TimeArray, days: FloatArray) -> DayProfile:
    """過去の日（days: 日数 × 窓）に 24 時間周期の余弦を最小二乗で当てはめる。NaN は除く。"""
    tt, x = _flatten(t_min, days)
    design = np.column_stack([np.ones(tt.size), np.cos(OMEGA * tt), np.sin(OMEGA * tt)])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    resid = x - design @ coef
    return DayProfile(float(coef[0]), float(coef[1]), float(coef[2]), _robust_sd(resid))


def fit_flat(t_min: TimeArray, days: FloatArray) -> DayProfile:
    """日内変動を補正しない基準（個人の中央値と、その周りの頑健な SD）。"""
    _, x = _flatten(t_min, days)
    center = float(np.median(x))
    return DayProfile(center, 0.0, 0.0, _robust_sd(x - center))


def detect(t_min: TimeArray, ln_hf: FloatArray, profile: DayProfile) -> BoolArray:
    """各窓で過緊張か。直近 SUSTAIN_WINDOWS 窓の z がすべて閾値以下のとき True（NaN の窓は根拠にしない）。"""
    z = (ln_hf - profile.predict(t_min)) / profile.sd
    low = np.where(np.isnan(z), False, z <= Z_THRESHOLD)
    flags = np.zeros(z.size, dtype=bool)
    for i in range(SUSTAIN_WINDOWS - 1, z.size):
        flags[i] = bool(low[i - SUSTAIN_WINDOWS + 1 : i + 1].all())
    return flags


def worker_flags(worker: Worker, *, corrected: bool = True) -> BoolArray:
    t = time_grid_min()
    fit = fit_circadian if corrected else fit_flat
    return detect(t, worker.ln_hf, fit(t, worker.calibration_ln_hf))


@dataclass(frozen=True)
class DetectionScores:
    """合成の真のエピソードに対する検知の成績（集計値のみ）。"""

    episodes: int
    detected_episodes: int
    false_alarm_windows: int  # エピソードの外で過緊張とした窓
    false_alarm_windows_afternoon: int  # そのうち 13 時以降
    windows: int

    def to_json(self) -> dict[str, float | int]:
        return {
            "episodes": self.episodes,
            "detected_episodes": self.detected_episodes,
            "sensitivity": self.detected_episodes / self.episodes if self.episodes else 0.0,
            "false_alarm_windows": self.false_alarm_windows,
            "false_alarm_windows_afternoon": self.false_alarm_windows_afternoon,
            "windows": self.windows,
        }


def detection_scores(workers: Sequence[Worker], *, corrected: bool = True) -> DetectionScores:
    t = time_grid_min()
    episodes = detected = false_alarms = false_pm = windows = 0
    for w in workers:
        flags = worker_flags(w, corrected=corrected)
        inside = np.zeros(t.size, dtype=bool)
        for e in w.episodes:
            span = (t >= e.start_min) & (t < e.end_min + EPISODE_GRACE_MIN)
            inside |= span
            episodes += 1
            detected += int(flags[span].any())
        outside = flags & ~inside
        false_alarms += int(outside.sum())
        false_pm += int(outside[t >= 13 * 60].sum())
        windows += t.size
    return DetectionScores(episodes, detected, false_alarms, false_pm, windows)
