"""追跡の解析（docs/pilot-protocol.md 4.2）: 接触の後に ln(HF) の回復がどれだけ残るか。

- 接触の終わりからの 1 分窓の Δln(HF)（基準は接触前の安静、`study.session_curve` と同じ）。
  帰りの 5 分と、出来事（会話・電話・飲食・離席）のある窓は除く。
- 持ち越しがあるか: 接触の終わりから 5〜25 分（帰りの 5 分の直後から）の Δ の平均を `carry ~ walk + leave + (1|subject)`。
  `leave` の係数が正なら、離脱（緑）の後に回復が残っている。stay の条件が日内変動の対照になる。
- τc: leave の条件のセッションの平均曲線に K·e^{−t/τc} を当てはめ、セッション単位のブートストラップ
  （窓どうしの誤差が相関するため）で信頼区間を求める。決めた τc は `vmgs jitai --carryover-min` に入れる。
指数で減るという形は仮定。実測の曲線が別の形なら、JITAI の回復のモデルも見直す。
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from vmgs.analysis.hrv import clean_rr, windowed_hrv
from vmgs.analysis.recording import FloatArray, Recording
from vmgs.analysis.study import BASELINE_S, STEP_S

RETURN_MIN = 5  # 接触の終わりからオフィスに戻るまで（帰りの歩行。解析から除く）
TRACK_END_MIN = 60  # 接触の終わりから追跡の終わりまで
ENDPOINT_MIN = (
    5,
    25,
)  # 持ち越しの検出に使う区間（接触の終わりから。見直しは docs/pilot-protocol.md 4.2）
TRACK_WINDOW_S = 60.0
TAU_C_BOUNDS_MIN = (1.0, 600.0)


@dataclass(frozen=True, eq=False)
class TrackingCurve:
    """接触の終わりからの分（窓の中心）ごとの、基準からの ln(HF) の変化。除いた窓は NaN。"""

    recording: Recording
    t_min: FloatArray
    delta_lnhf: FloatArray
    baseline_lnhf: float


@dataclass(frozen=True)
class CarryEffect:
    coef: float  # leave の係数（ENDPOINT_MIN の区間の Δ の平均の差）
    ci: tuple[float, float]  # 95% CI
    n_sessions: int


@dataclass(frozen=True)
class TauCEstimate:
    tau_c_min: float
    k: float  # 接触の終わり（t=0）に外挿した Δ
    ci: tuple[float, float]  # τc の 95% CI（セッション単位のブートストラップ）
    n_sessions: int


def tracking_curve(rec: Recording, baseline_s: float = BASELINE_S) -> TrackingCurve:
    """1 セッションの追跡の曲線。窓は接触の終わりから RETURN_MIN〜TRACK_END_MIN 分の 1 分刻み。

    窓が追跡の区間（track_start〜track_end）からはみ出すもの、出来事を含むものは NaN にする。
    """
    if not rec.tracked:
        raise ValueError(f"session {rec.session} に追跡の時刻（contact_end など）が無い")
    assert rec.contact_end_s is not None and rec.track_start_s is not None
    assert rec.track_end_s is not None
    times, rr, _ = clean_rr(rec.beat_times_s, rec.rr_ms)
    half = TRACK_WINDOW_S / 2.0
    base_centers = np.arange(-baseline_s + half, -half + 1e-9, STEP_S)
    baseline = float(np.nanmean(windowed_hrv(times, rr, base_centers).ln_hf))
    t_min: FloatArray = np.arange(RETURN_MIN, TRACK_END_MIN, dtype=np.float64) + 0.5
    centers = rec.contact_end_s + t_min * 60.0
    ln_hf = windowed_hrv(times, rr, centers, window_s=TRACK_WINDOW_S).ln_hf
    inside = (centers - half >= rec.track_start_s) & (centers + half <= rec.track_end_s)
    event_t = np.array([t for t, _ in rec.events])
    clean = np.array([not np.any((event_t >= c - half) & (event_t < c + half)) for c in centers])
    delta = np.where(inside & clean, ln_hf - baseline, np.nan)
    return TrackingCurve(rec, t_min, delta, baseline)


def carry_table(
    curves: Sequence[TrackingCurve], endpoint_min: tuple[float, float] = ENDPOINT_MIN
) -> pd.DataFrame:
    """セッションごとの、接触の終わりから endpoint_min（既定 5〜25 分）の Δ の平均（列 carry）。"""
    rows = []
    for c in curves:
        in_ep = (c.t_min >= endpoint_min[0]) & (c.t_min < endpoint_min[1])
        values = c.delta_lnhf[in_ep]
        carry = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
        rec = c.recording
        rows.append(
            {
                "session": rec.session,
                "subject": rec.subject,
                "walk": int(rec.walk),
                "leave": int(rec.leave),
                "carry": carry,
            }
        )
    return pd.DataFrame(rows)


def fit_carry_effect(table: pd.DataFrame) -> CarryEffect:
    """`carry ~ walk + leave + (1|subject)` の leave の係数と 95% CI。carry が無いセッションは除く。"""
    data = table.dropna(subset=["carry"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = smf.mixedlm("carry ~ walk + leave", data, groups=data["subject"]).fit(
            reml=True, method=["bfgs", "lbfgs", "powell"]
        )
    lo, hi = result.conf_int().loc["leave"]
    return CarryEffect(float(result.params["leave"]), (float(lo), float(hi)), len(data))


def decay(t: FloatArray, k: float, tau_c: float) -> FloatArray:
    y: FloatArray = k * np.exp(-t / tau_c)
    return y


def fit_decay(t_min: FloatArray, mean: FloatArray) -> tuple[float, float]:
    """平均の曲線に K·e^{−t/τc} を当てはめた (K, τc [分])。当てはまらなければ NaN。"""
    use = np.isfinite(mean)
    try:
        params, _ = curve_fit(
            decay,
            t_min[use],
            mean[use],
            p0=(0.3, 30.0),
            bounds=([-5, TAU_C_BOUNDS_MIN[0]], [5, TAU_C_BOUNDS_MIN[1]]),
            maxfev=10000,
        )
    except (RuntimeError, ValueError):
        return float("nan"), float("nan")
    return float(params[0]), float(params[1])


def _mean_curve(curves: Sequence[TrackingCurve]) -> FloatArray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # どのセッションも除いた窓は NaN のまま
        mean: FloatArray = np.nanmean(np.vstack([c.delta_lnhf for c in curves]), axis=0)
    return mean


def fit_tau_c(curves: Sequence[TrackingCurve], n_boot: int = 1000, seed: int = 0) -> TauCEstimate:
    """leave の条件のセッションから τc を推定する（stay の条件は使わない）。"""
    leave = [c for c in curves if c.recording.leave]
    if not leave:
        raise ValueError("leave の条件の追跡が 1 つもない")
    t_min = leave[0].t_min
    k, tau_c = fit_decay(t_min, _mean_curve(leave))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(leave), len(leave))
        boots[b] = fit_decay(t_min, _mean_curve([leave[i] for i in idx]))[1]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return TauCEstimate(tau_c, k, (float(lo), float(hi)), len(leave))
