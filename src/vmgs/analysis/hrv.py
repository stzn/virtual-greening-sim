"""RR 間隔から、移動窓ごとの ln(HF)・SDNN・RMSSD を求める。

HF は、RR を等間隔（既定 4 Hz）に 3 次スプラインで補間し、線形トレンドを除いて Welch 法で
パワースペクトル密度を求め、0.15〜0.40 Hz を積分したもの [ms²]。窓は既定で 60 秒
（HF の短時間解析の最短として一般に使われる長さ）。
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.signal import detrend, welch

from vmgs.analysis.recording import FloatArray

HF_BAND_HZ = (0.15, 0.40)
RR_RANGE_MS = (300.0, 2000.0)


def _detrend_linear(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    # scipy-stubs の welch の detrend は Literal が "literal" になっていて "linear" を受けないため関数で渡す
    out: npt.NDArray[Any] = detrend(x, type="linear")
    return out


@dataclass(frozen=True, eq=False)
class WindowedHrv:
    """窓ごとの指標。t_s は窓の中心。計算できない窓は NaN。"""

    t_s: FloatArray
    ln_hf: FloatArray  # ln(HF [ms²])
    sdnn_ms: FloatArray
    rmssd_ms: FloatArray
    mean_rr_ms: FloatArray


def clean_rr(
    beat_times_s: FloatArray, rr_ms: FloatArray, max_jump: float = 0.2, half_width: int = 5
) -> tuple[FloatArray, FloatArray, int]:
    """範囲外（300〜2000 ms の外）と、前後の拍の中央値から max_jump 以上離れた RR を除く。

    期外収縮のような孤立した跳びは除き、歩行開始のような持続する変化は中央値が追従して残す。
    中央値は、範囲内の RR だけで前後 half_width 拍（自身を含む）から求める。
    除いた件数も返す（隠さずに報告するため）。
    """
    in_range = (rr_ms >= RR_RANGE_MS[0]) & (rr_ms <= RR_RANGE_MS[1])
    valid = rr_ms[in_range]
    local = np.empty(valid.size)
    for j in range(valid.size):
        local[j] = np.median(valid[max(0, j - half_width) : j + half_width + 1])
    ok = np.abs(valid - local) / local <= max_jump
    keep = np.zeros(rr_ms.size, dtype=bool)
    keep[np.flatnonzero(in_range)[ok]] = True
    return beat_times_s[keep], rr_ms[keep], int(rr_ms.size - keep.sum())


def windowed_hrv(
    beat_times_s: FloatArray,
    rr_ms: FloatArray,
    centers_s: FloatArray,
    window_s: float = 60.0,
    fs_hz: float = 4.0,
    min_coverage: float = 0.9,
) -> WindowedHrv:
    """centers_s を中心とする window_s 秒の窓ごとに指標を求める。

    窓の中の拍が窓の長さの min_coverage 未満しか覆っていない窓は NaN にする。
    """
    n = centers_s.size
    ln_hf = np.full(n, np.nan)
    sdnn = np.full(n, np.nan)
    rmssd = np.full(n, np.nan)
    mean_rr = np.full(n, np.nan)
    if beat_times_s.size < 4:
        return WindowedHrv(centers_s, ln_hf, sdnn, rmssd, mean_rr)

    # 補間は系列全体で一度だけ行い、窓ごとに切り出す
    grid = np.arange(beat_times_s[0], beat_times_s[-1], 1.0 / fs_hz)
    resampled: FloatArray = CubicSpline(beat_times_s, rr_ms)(grid)
    half = window_s / 2.0
    for i, c in enumerate(centers_s):
        lo, hi = c - half, c + half
        in_win = (beat_times_s >= lo) & (beat_times_s < hi)
        rr = rr_ms[in_win]
        if rr.size < 3 or rr.sum() / 1000.0 < window_s * min_coverage:
            continue
        mean_rr[i] = rr.mean()
        sdnn[i] = rr.std(ddof=1)
        rmssd[i] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
        seg = resampled[(grid >= lo) & (grid < hi)]
        if seg.size < window_s * fs_hz * min_coverage:
            continue
        nperseg = min(seg.size, int(32 * fs_hz))
        freqs, psd = welch(seg, fs=fs_hz, nperseg=nperseg, detrend=_detrend_linear)
        band = (freqs >= HF_BAND_HZ[0]) & (freqs <= HF_BAND_HZ[1])
        hf = float(np.trapezoid(psd[band], freqs[band]))
        if hf > 0:
            ln_hf[i] = np.log(hf)
    return WindowedHrv(centers_s, ln_hf, sdnn, rmssd, mean_rr)
