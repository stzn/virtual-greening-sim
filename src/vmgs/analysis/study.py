"""1 セッションの RR から、接触開始からの ln(HF) の変化の曲線を作る。"""

from dataclasses import dataclass

import numpy as np

from vmgs.analysis.hrv import WindowedHrv, clean_rr, windowed_hrv
from vmgs.analysis.recording import FloatArray, Recording

BASELINE_S = 180.0  # 接触前のオフィスでの安静（基準）
CONTACT_S = 600.0  # 接触の長さ（10 分）
WINDOW_S = 60.0
STEP_S = 10.0


@dataclass(frozen=True, eq=False)
class SessionCurve:
    """接触開始後の窓（窓全体が t>=0 にあるもの）での、基準からの ln(HF) の変化。"""

    recording: Recording
    t_s: FloatArray
    delta_lnhf: FloatArray
    baseline_lnhf: float
    hrv: WindowedHrv  # 基準から接触の終わりまでの全窓
    n_removed: int  # clean_rr で除いた RR の数


def session_curve(
    rec: Recording,
    baseline_s: float = BASELINE_S,
    contact_s: float = CONTACT_S,
    window_s: float = WINDOW_S,
    step_s: float = STEP_S,
) -> SessionCurve:
    """基準 = 窓全体が接触前にある窓の ln(HF) の平均。接触をまたぐ窓は使わない。"""
    times, rr, n_removed = clean_rr(rec.beat_times_s, rec.rr_ms)
    half = window_s / 2.0
    centers = np.arange(-baseline_s + half, contact_s - half + 1e-9, step_s)
    hrv = windowed_hrv(times, rr, centers, window_s=window_s)
    before = hrv.t_s <= -half
    after = hrv.t_s >= half
    baseline = float(np.nanmean(hrv.ln_hf[before]))
    return SessionCurve(
        recording=rec,
        t_s=hrv.t_s[after],
        delta_lnhf=hrv.ln_hf[after] - baseline,
        baseline_lnhf=baseline,
        hrv=hrv,
        n_removed=n_removed,
    )
