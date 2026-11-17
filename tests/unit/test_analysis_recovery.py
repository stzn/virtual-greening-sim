"""合成データで、真のパラメータを解析で取り戻せること（問い①: R(t)、問い②: 緑の効果）。"""

import numpy as np
import pytest

from vmgs.analysis.mixed import fit_green_effect, session_table
from vmgs.analysis.response import ResponseEstimate, fit_group_response
from vmgs.analysis.study import SessionCurve, session_curve
from vmgs.analysis.synthetic import Design, SyntheticStudy, simulate_study


@pytest.fixture(scope="module")
def study() -> SyntheticStudy:
    return simulate_study(Design(n_subjects=24, repeats=2), seed=7)


@pytest.fixture(scope="module")
def curves(study: SyntheticStudy) -> list[SessionCurve]:
    return [session_curve(r) for r in study.recordings]


def _group(
    curves: list[SessionCurve], condition: str
) -> tuple[list[SessionCurve], ResponseEstimate]:
    members = [c for c in curves if c.recording.condition == condition]
    return members, fit_group_response(members, n_boot=200, seed=0)


def test_合成データは2x2の条件をそろえ真の値を持つ(study: SyntheticStudy) -> None:
    conditions = {r.condition for r in study.recordings}
    assert conditions == {"sit-stay", "walk-stay", "sit-leave", "walk-leave"}
    assert len(study.recordings) == 24 * 4 * 2
    assert set(study.session_truth) == {r.session for r in study.recordings}
    # 離脱した（緑のある）場所では、緑が多いほど PET・騒音も高い（交絡を埋め込んでいる）
    leave = [r for r in study.recordings if r.leave]
    green = np.array([r.green for r in leave])
    assert np.corrcoef(green, [r.pet_c for r in leave])[0, 1] > 0.5
    assert np.corrcoef(green, [r.noise_db for r in leave])[0, 1] > 0.5


def test_緑の中で座る条件の時定数と到達値を取り戻す(
    study: SyntheticStudy, curves: list[SessionCurve]
) -> None:
    members, est = _group(curves, "sit-leave")
    true_tau = study.truth.tau_s
    true_amp = float(np.mean([study.session_truth[c.recording.session].amplitude for c in members]))
    # 誤差は時定数 15% 以内・到達値 0.1 以内で、95% 信頼区間は真の値を含む
    assert abs(est.fit.tau_s - true_tau) / true_tau < 0.15
    assert abs(est.fit.amplitude - true_amp) < 0.1
    assert est.ci["tau_s"][0] <= true_tau <= est.ci["tau_s"][1]
    assert est.ci["amplitude"][0] <= true_amp <= est.ci["amplitude"][1]
    # ピーク（95% 到達）は 5〜10 分
    assert 300.0 <= est.fit.t95_s <= 600.0


def test_歩行の段差と緑の立ち上がりを分けて取り戻す(
    study: SyntheticStudy, curves: list[SessionCurve]
) -> None:
    # Given 歩きながら離脱する条件（歩行の即時の段差 + 緑のゆっくりした立ち上がり）
    members, est = _group(curves, "walk-leave")
    true_amp = float(np.mean([study.session_truth[c.recording.session].amplitude for c in members]))
    # Then 段差は歩行の効果、到達値は緑などによる立ち上がりとして分かれる
    assert abs(est.fit.offset - study.truth.beta_walk_step) < 0.1
    assert abs(est.fit.amplitude - true_amp) < 0.15
    assert abs(est.fit.tau_s - study.truth.tau_s) / study.truth.tau_s < 0.3


def test_PETと騒音を共変量に入れると緑の効果が真の値に近づく(
    study: SyntheticStudy, curves: list[SessionCurve]
) -> None:
    # Given 時定数は真の値ではなく、問い①で推定した値を使う（2 段階の解析）
    _, est = _group(curves, "sit-leave")
    table = session_table(curves, tau_s=est.fit.tau_s)
    adjusted = fit_green_effect(table, covariates=True)
    naive = fit_green_effect(table, covariates=False)
    true_green = study.truth.beta_green
    assert adjusted.converged and naive.converged
    # 共変量あり: 真の値の近くで、95% 信頼区間が真の値を含む
    assert abs(adjusted.estimate - true_green) < 0.25
    assert adjusted.ci_low <= true_green <= adjusted.ci_high
    # 共変量なし: 交絡で大きく下に歪み、信頼区間は真の値を含まない
    assert naive.estimate < true_green - 0.4
    assert naive.ci_high < true_green


def test_表には被験者と条件と推定した到達値が並ぶ(
    study: SyntheticStudy, curves: list[SessionCurve]
) -> None:
    table = session_table(curves, tau_s=study.truth.tau_s)
    assert len(table) == len(study.recordings)
    for col in ("subject", "walk", "leave", "green", "pet_c", "noise_db", "offset", "amplitude"):
        assert col in table.columns
