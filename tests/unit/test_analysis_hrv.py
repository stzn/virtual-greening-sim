"""RR 間隔から窓ごとの ln(HF)・SDNN・RMSSD を求める振る舞い。"""

import numpy as np

from vmgs.analysis.hrv import clean_rr, windowed_hrv


def _stationary_rr(ln_hf: float, duration_s: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """呼吸性の揺らぎ（0.25 Hz の正弦）だけを持つ定常の RR 系列。HF のパワーは exp(ln_hf)。"""
    rng = np.random.default_rng(seed)
    amp = np.sqrt(2.0 * np.exp(ln_hf))
    t = 0.0
    times, rrs = [], []
    while t < duration_s:
        rr = 850.0 + amp * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0.0, 3.0)
        t += rr / 1000.0
        times.append(t)
        rrs.append(rr)
    return np.array(times), np.array(rrs)


def test_定常のRRから求めたlnHFの差は真の差に近い() -> None:
    # Given HF のパワーが e^5 と e^6 の 2 つの系列
    t5, rr5 = _stationary_rr(5.0, 300.0, seed=1)
    t6, rr6 = _stationary_rr(6.0, 300.0, seed=2)
    centers = np.arange(60.0, 241.0, 10.0)
    # When 60 秒窓で ln(HF) を求める
    h5 = windowed_hrv(t5, rr5, centers)
    h6 = windowed_hrv(t6, rr6, centers)
    # Then 差は 1 に近く、絶対値も真の値から大きく外れない
    assert abs(float(np.mean(h6.ln_hf - h5.ln_hf)) - 1.0) < 0.1
    assert abs(float(np.mean(h6.ln_hf)) - 6.0) < 0.3


def test_SDNNとRMSSDを窓ごとに返す() -> None:
    # Given 800 と 820 を交互に繰り返す RR（平均 810、隣との差は常に 20）
    rr = np.tile([800.0, 820.0], 200)
    times = np.cumsum(rr) / 1000.0
    # When 窓ごとに求める
    h = windowed_hrv(times, rr, np.array([60.0, 120.0]))
    # Then SDNN は約 10、RMSSD は 20、平均 RR は 810
    assert np.allclose(h.sdnn_ms, 10.0, atol=0.2)
    assert np.allclose(h.rmssd_ms, 20.0)
    assert np.allclose(h.mean_rr_ms, 810.0, atol=1.0)


def test_RRが足りない窓は欠損にする() -> None:
    rr = np.full(50, 800.0)
    times = np.cumsum(rr) / 1000.0  # 40 秒しかない
    h = windowed_hrv(times, rr, np.array([30.0, 200.0]))
    assert np.isnan(h.ln_hf[1])
    assert np.isnan(h.sdnn_ms[1])


def test_範囲外と急な変化のRRを除く() -> None:
    # Given 正常な RR に、範囲外（150・2500 ms）と直前から 40% の跳び（期外収縮）を混ぜる
    rr = np.array([800.0, 810.0, 150.0, 805.0, 2500.0, 800.0, 1150.0, 810.0])
    times = np.cumsum(rr) / 1000.0
    # When 除く
    kept_t, kept_rr, n_removed = clean_rr(times, rr)
    # Then 3 件が除かれ、時刻は対応したまま残る
    assert n_removed == 3
    assert list(kept_rr) == [800.0, 810.0, 805.0, 800.0, 810.0]
    assert len(kept_t) == len(kept_rr)


def test_持続する心拍の変化は外れ値として除かない() -> None:
    # Given 歩き始めで RR が 850 から 650 に下がり（24% の変化）、そのまま続く
    rr = np.concatenate([np.full(100, 850.0), np.full(100, 650.0)])
    times = np.cumsum(rr) / 1000.0
    # When 除く / Then 切り替わりの数拍を除いても、下がった後の拍は残る
    _, kept_rr, n_removed = clean_rr(times, rr)
    assert n_removed <= 2
    assert np.sum(kept_rr == 650.0) >= 98
