"""過渡応答 R(t) の当てはめ（問い①）。

接触開始（t=0）からの ln(HF) の変化を、即時の段差と 1 次の飽和曲線の和で表す:

    R(t) = offset + amplitude * (1 - exp(-t / tau))

offset は歩行の開始のような即時の変化、amplitude と tau は緑などによるゆっくりした立ち上がり。
ピーク到達時間は、立ち上がりが到達値の 95% に届く時刻 tau * ln 20 とする。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit

from vmgs.analysis.recording import FloatArray

if TYPE_CHECKING:
    from vmgs.analysis.study import SessionCurve

PARAM_NAMES = ("offset", "amplitude", "tau_s")
TAU_BOUNDS_S = (5.0, 3600.0)


def saturating(t: FloatArray, offset: float, amplitude: float, tau_s: float) -> FloatArray:
    out: FloatArray = offset + amplitude * (1.0 - np.exp(-t / tau_s))
    return out


@dataclass(frozen=True)
class ResponseFit:
    offset: float
    amplitude: float
    tau_s: float

    @property
    def t95_s(self) -> float:
        """立ち上がりが到達値の 95% に届く時刻（接触開始から）。"""
        return self.tau_s * math.log(20.0)


@dataclass(frozen=True)
class ResponseEstimate:
    """当てはめた値と、セッションを単位にしたブートストラップの 95% 信頼区間。"""

    fit: ResponseFit
    ci: dict[str, tuple[float, float]]
    n_sessions: int


def fit_response(t: FloatArray, y: FloatArray) -> ResponseFit:
    """1 本の曲線（NaN を除く）に R(t) を最小二乗で当てはめる。"""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    span = float(t.max() - t.min())
    p0 = (float(y[: max(1, y.size // 10)].mean()), float(y[-y.size // 4 :].mean() - y[0]), span / 4)
    params, _ = curve_fit(
        saturating,
        t,
        y,
        p0=p0,
        bounds=([-np.inf, -np.inf, TAU_BOUNDS_S[0]], [np.inf, np.inf, TAU_BOUNDS_S[1]]),
        maxfev=10000,
    )
    return ResponseFit(float(params[0]), float(params[1]), float(params[2]))


def session_amplitude(t: FloatArray, y: FloatArray, tau_s: float) -> tuple[float, float]:
    """時定数を固定し、段差と到達値を線形の最小二乗で求める（セッションごとの推定に使う）。"""
    ok = np.isfinite(y)
    g = 1.0 - np.exp(-t[ok] / tau_s)
    design = np.column_stack([np.ones_like(g), g])
    coef, *_ = np.linalg.lstsq(design, y[ok], rcond=None)
    return float(coef[0]), float(coef[1])


def _mean_curve(curves: Sequence[SessionCurve]) -> tuple[FloatArray, FloatArray]:
    t = curves[0].t_s
    for c in curves:
        if c.t_s.shape != t.shape or not np.allclose(c.t_s, t):
            raise ValueError("平均する曲線は同じ時刻の格子で求める")
    mean: FloatArray = np.nanmean(np.vstack([c.delta_lnhf for c in curves]), axis=0)
    return t, mean


def fit_group_response(
    curves: Sequence[SessionCurve], n_boot: int = 1000, seed: int = 0
) -> ResponseEstimate:
    """セッションの曲線を平均して R(t) を当てはめ、セッションを再標本化して信頼区間を求める。

    窓は重なっていて隣の窓どうしの誤差が相関するため、窓ではなくセッションを単位に再標本化する。
    """
    if not curves:
        raise ValueError("曲線が 1 本もない")
    fit = fit_response(*_mean_curve(curves))
    rng = np.random.default_rng(seed)
    boots = np.empty((n_boot, 3))
    for b in range(n_boot):
        idx = rng.integers(0, len(curves), len(curves))
        f = fit_response(*_mean_curve([curves[i] for i in idx]))
        boots[b] = (f.offset, f.amplitude, f.tau_s)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ci = {name: (float(lo[i]), float(hi[i])) for i, name in enumerate(PARAM_NAMES)}
    return ResponseEstimate(fit, ci, len(curves))
