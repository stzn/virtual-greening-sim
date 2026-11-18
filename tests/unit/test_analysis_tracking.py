"""追跡の解析（pilot-protocol.md 4.2）: 接触の後の Δln(HF)、持ち越しの効果、τc。すべて合成データ。"""

import dataclasses

import numpy as np
import pytest

from vmgs.analysis.synthetic import Design, SyntheticStudy, TrueParams, simulate_study
from vmgs.analysis.tracking import (
    ENDPOINT_MIN,
    RETURN_MIN,
    TRACK_END_MIN,
    TrackingCurve,
    carry_table,
    fit_carry_effect,
    fit_tau_c,
    tracking_curve,
)

TRUE_TAU_C_S = 1800.0  # 30 分


@pytest.fixture(scope="module")
def study() -> SyntheticStudy:
    return simulate_study(
        Design(n_subjects=16, repeats=2, track_s=TRACK_END_MIN * 60.0),
        TrueParams(tau_c_s=TRUE_TAU_C_S),
        seed=3,
    )


@pytest.fixture(scope="module")
def curves(study: SyntheticStudy) -> list[TrackingCurve]:
    return [tracking_curve(r) for r in study.recordings]


def test_追跡ありの合成データは_接触の終わり_追跡の始めと終わりを持つ(
    study: SyntheticStudy,
) -> None:
    rec = study.recordings[0]
    assert rec.contact_end_s == 600.0
    assert rec.track_start_s == 600.0 + RETURN_MIN * 60
    assert rec.track_end_s == 600.0 + TRACK_END_MIN * 60
    assert rec.beat_times_s[-1] >= rec.track_end_s - 2.0


def test_追跡の無い合成データは_今までと同じ拍を作る() -> None:
    a = simulate_study(Design(n_subjects=1, repeats=1), seed=0).recordings[0]
    b = simulate_study(Design(n_subjects=1, repeats=1, track_s=0.0), seed=0).recordings[0]
    assert np.array_equal(a.rr_ms, b.rr_ms) and a.track_end_s is None


def test_追跡の曲線は_帰りの5分を除いた1分窓で_接触の終わりからの分を中心に持つ(
    curves: list[TrackingCurve],
) -> None:
    c = curves[0]
    assert c.t_min[0] == pytest.approx(RETURN_MIN + 0.5)
    assert c.t_min[-1] == pytest.approx(TRACK_END_MIN - 0.5)
    assert np.isfinite(c.delta_lnhf).all()


def test_出来事のある窓は除く(study: SyntheticStudy) -> None:
    rec = dataclasses.replace(study.recordings[0], events=((600.0 + 20.3 * 60, "phone"),))
    c = tracking_curve(rec)
    removed = np.isnan(c.delta_lnhf)
    assert c.t_min[removed].tolist() == [20.5]


def test_追跡の時刻が無い記録は_理由を挙げて止める() -> None:
    rec = simulate_study(Design(n_subjects=1, repeats=1), seed=0).recordings[0]
    with pytest.raises(ValueError, match="追跡"):
        tracking_curve(rec)


def test_離脱した条件の持ち越しは_stay_より大きいと推定する(curves: list[TrackingCurve]) -> None:
    table = carry_table(curves)
    assert set(table.columns) >= {"session", "subject", "walk", "leave", "carry"}
    effect = fit_carry_effect(table)
    # 真の leave の効果（到達値 0.2 + 緑の分）は正で、帰った後の 5〜25 分にも残る
    assert effect.coef > 0
    assert effect.ci[0] < effect.coef < effect.ci[1]
    assert ENDPOINT_MIN == (5, 25)


def test_持ち越しの区間を渡すと_その区間の窓だけで平均する(curves: list[TrackingCurve]) -> None:
    # Given 既定（5〜25 分）と、10〜30 分の区間
    default = carry_table(curves)
    early = carry_table(curves, endpoint_min=(10, 30))
    # Then 渡した区間の窓（中心が 10 分以上 30 分未満）だけで平均する
    c = curves[0]
    in_ep = (c.t_min >= 10) & (c.t_min < 30)
    assert early["carry"].iloc[0] == pytest.approx(float(np.nanmean(c.delta_lnhf[in_ep])))
    assert not np.allclose(early["carry"], default["carry"])


def test_真のτc_30分の合成データから推定したτcの信頼区間は_真の値を含む(
    curves: list[TrackingCurve],
) -> None:
    est = fit_tau_c(curves, n_boot=100, seed=0)
    assert est.n_sessions == sum(c.recording.leave for c in curves)
    assert est.ci[0] <= TRUE_TAU_C_S / 60 <= est.ci[1]
    assert est.k > 0
