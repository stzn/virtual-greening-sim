"""緑の純粋な効果を取り出す階層モデル（問い②）。

セッションごとに、時定数を問い①の値に固定して段差（offset）と到達値（amplitude）を求め、
到達値を目的変数にした線形混合効果モデル（被験者のランダム切片）を当てはめる:

    amplitude ~ walk + leave + green [+ pet_c + noise_db] + (1 | subject)

PET・騒音を入れないと、緑の多い場所ほど暑く騒がしい交絡により、緑の効果が歪む。
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from vmgs.analysis.response import session_amplitude
from vmgs.analysis.study import SessionCurve

FORMULA_ADJUSTED = "amplitude ~ walk + leave + green + pet_c + noise_db"
FORMULA_NAIVE = "amplitude ~ walk + leave + green"


def session_table(curves: Sequence[SessionCurve], tau_s: float) -> pd.DataFrame:
    """セッションごとに 1 行: 被験者・条件・環境と、推定した段差・到達値。"""
    rows = []
    for c in curves:
        r = c.recording
        offset, amplitude = session_amplitude(c.t_s, c.delta_lnhf, tau_s)
        rows.append(
            {
                "session": r.session,
                "subject": r.subject,
                "condition": r.condition,
                "walk": int(r.walk),
                "leave": int(r.leave),
                "green": r.green,
                "pet_c": r.pet_c,
                "noise_db": r.noise_db,
                "offset": offset,
                "amplitude": amplitude,
                "baseline_lnhf": c.baseline_lnhf,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class EffectEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    params: dict[str, float]  # 全固定効果の推定値（PET・騒音は標本平均で中心化した切片）
    formula: str
    converged: bool  # 最適化が収束したか（収束しなかった推定は信頼しない）
    subject_var: float  # 被験者のランダム切片の分散
    on_boundary: bool  # 被験者の分散が 0 の境界に近い（固定効果は使えるが、分散の推定は不安定）


def fit_green_effect(table: pd.DataFrame, covariates: bool = True) -> EffectEstimate:
    """緑の量の固定効果と 95% 信頼区間。covariates=False は PET・騒音を入れない比較用。"""
    formula = FORMULA_ADJUSTED if covariates else FORMULA_NAIVE
    # 中心化して数値の条件を良くする（傾きは変わらない）
    data = table.assign(
        pet_c=table["pet_c"] - table["pet_c"].mean(),
        noise_db=table["noise_db"] - table["noise_db"].mean(),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        # 既定の BFGS で収束しなければ L-BFGS、Powell の順に試す（statsmodels が順に再試行する）
        result = smf.mixedlm(formula, data, groups=data["subject"]).fit(
            reml=True, method=["bfgs", "lbfgs", "powell"]
        )
    on_boundary = any("boundary" in str(w.message) for w in caught)
    ci = result.conf_int().loc["green"]
    fe = result.fe_params
    return EffectEstimate(
        estimate=float(fe["green"]),
        ci_low=float(ci.iloc[0]),
        ci_high=float(ci.iloc[1]),
        params={str(k): float(v) for k, v in fe.items()},
        formula=formula,
        converged=bool(result.converged),
        subject_var=float(result.cov_re.iloc[0, 0]),
        on_boundary=on_boundary,
    )
