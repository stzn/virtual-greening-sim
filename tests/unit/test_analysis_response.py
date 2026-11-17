"""過渡応答 R(t)（飽和曲線）の当てはめの振る舞い。"""

import math

import numpy as np

from vmgs.analysis.response import fit_response, saturating, session_amplitude


def test_雑音のない飽和曲線からパラメータを取り戻す() -> None:
    t = np.arange(30.0, 571.0, 10.0)
    y = saturating(t, offset=-0.3, amplitude=0.8, tau_s=120.0)
    fit = fit_response(t, y)
    assert math.isclose(fit.offset, -0.3, abs_tol=1e-4)
    assert math.isclose(fit.amplitude, 0.8, abs_tol=1e-4)
    assert math.isclose(fit.tau_s, 120.0, rel_tol=1e-3)


def test_ピーク到達時間は到達値の95パーセントに届く時刻() -> None:
    t = np.arange(30.0, 571.0, 10.0)
    fit = fit_response(t, saturating(t, 0.0, 0.5, 100.0))
    # 1 - exp(-t/τ) = 0.95 → t = τ ln 20
    assert math.isclose(fit.t95_s, 100.0 * math.log(20.0), rel_tol=1e-3)


def test_時定数を固定すると段差と到達値を線形に求める() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(30.0, 571.0, 10.0)
    y = saturating(t, -0.5, 0.6, 120.0) + rng.normal(0.0, 0.02, t.size)
    offset, amplitude = session_amplitude(t, y, tau_s=120.0)
    assert abs(offset + 0.5) < 0.05
    assert abs(amplitude - 0.6) < 0.05
