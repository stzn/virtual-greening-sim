"""パイロット実験のサンプルサイズの目安: 被験者数を変えて、緑の効果と回復の持ち越しの検出力を試算する。

    uv run python scripts/pilot_power.py [--reps 30] [--subjects 8 12 16 24 32]

- 緑の効果: `simulate_study`（接触 10 分）で被験者数・緑の真の効果を変え、フェーズ 5 と同じ解析
  （τ を sit-leave の群の曲線で推定 → セッションごとの到達値 → 共変量ありの混合効果モデル）で、
  緑の効果の 95% CI が 0 を含まなかった割合を検出力とする。
- 持ち越し: `simulate_study` は接触の後を作らないため、この試算の中で追跡の区間を窓の値で作る
  （下の `_tracking`）。セッションの真の到達値 A が、接触の終わりから時定数 τc で減ると置く。
  窓の推定誤差は、拍から作った合成データに `windowed_hrv` をかけて測った値を使う（`_window_noise_sd`）。

API は使わない。乱数の種は固定（rep k は種 1000+k）。src のコードは変えない。
"""

import argparse
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from vmgs.analysis.hrv import clean_rr, windowed_hrv
from vmgs.analysis.mixed import fit_green_effect, session_table
from vmgs.analysis.recording import FloatArray
from vmgs.analysis.response import fit_response
from vmgs.analysis.study import session_curve
from vmgs.analysis.synthetic import Design, SyntheticStudy, TrueParams, ou_process, simulate_study

# 追跡の区間の前提（仮定。docs/pilot-protocol.md に書いたもの）
RETURN_MIN = 5  # 接触の終わりからオフィスに戻るまで（帰りの歩行。解析から除く）
TRACK_END_MIN = 60  # 接触の終わりから追跡の終わりまで
ENDPOINT_MIN = (20, 40)  # 持ち越しの検出に使う区間（接触の終わりから）
SLOW_OU_SD = 0.2  # 1 時間の単位のゆっくりした揺らぎ（JITAI の合成のワーカーと同じ仮定）
SLOW_OU_MIN = 30.0
BASELINE_MIN = 3


def _window_noise_sd(seed: int = 0) -> float:
    """60 秒窓の ln(HF) の推定誤差の SD を、合成の拍で測る。

    ゆっくりした揺らぎ・被験者とセッションの揺らぎを 0 にし、立ち上がりが終わった後
    （接触 10 分以降）の重ならない窓の、セッション内の SD をまとめる。
    """
    p = TrueParams(ou_sd=0.0, subject_response_sd=0.0, session_response_sd=0.0)
    study = simulate_study(Design(n_subjects=6, repeats=1, contact_s=1500.0), p, seed=seed)
    variances = []
    for rec in study.recordings:
        times, rr, _ = clean_rr(rec.beat_times_s, rec.rr_ms)
        centers = np.arange(630.0, 1471.0, 60.0)
        ln_hf = windowed_hrv(times, rr, centers).ln_hf
        variances.append(np.nanvar(ln_hf, ddof=1))
    return float(np.sqrt(np.mean(variances)))


def _tracking(
    study: SyntheticStudy,
    tau_c_min: float,
    noise_sd: float,
    slow_sd: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """セッションごとの、接触の終わりからの 1 分窓の Δln(HF)（基準 = 接触前 3 分の窓の平均）。

    Δ(t) = A·(1 − e^{−600/τ})·e^{−t/τc} + slow(t) − 基準の slow + 窓の誤差。
    tau_c_min = 0 は持ち越しなし（偽陽性の確認用）。
    """
    t_min: FloatArray = np.arange(RETURN_MIN, TRACK_END_MIN, dtype=np.float64) + 0.5  # 窓の中心
    contact_min = 10
    # ゆっくりした揺らぎは、基準の始まりから追跡の終わりまでを 1 分刻みで続けて作る
    n_total = BASELINE_MIN + contact_min + TRACK_END_MIN
    out: dict[str, FloatArray] = {}
    p = study.truth
    for rec in study.recordings:
        amp = study.session_truth[rec.session].amplitude
        end_level = amp * (1.0 - math.exp(-600.0 / p.tau_s))
        carry = end_level * np.exp(-t_min / tau_c_min) if tau_c_min > 0 else np.zeros_like(t_min)
        slow = ou_process(n_total, 1.0, slow_sd, SLOW_OU_MIN, rng)
        base = slow[:BASELINE_MIN] + rng.normal(0.0, noise_sd, BASELINE_MIN)
        track = slow[BASELINE_MIN + contact_min + RETURN_MIN :] + rng.normal(
            0.0, noise_sd, t_min.size
        )
        out[rec.session] = carry + track - float(base.mean())
    return t_min, out


def _decay(t: FloatArray, k: float, tau_c: float) -> FloatArray:
    y: FloatArray = k * np.exp(-t / tau_c)
    return y


def _fit_tau_c(t_min: FloatArray, curves: Sequence[FloatArray], track_end: int) -> float:
    """離脱した条件のセッションの平均曲線に K·e^{−t/τc} を当てはめた τc [分]。上限 600 分。"""
    use = t_min < track_end
    mean = np.mean(np.vstack(curves), axis=0)[use]
    try:
        params, _ = curve_fit(
            _decay, t_min[use], mean, p0=(0.3, 30.0), bounds=([-5, 1], [5, 600]), maxfev=10000
        )
    except RuntimeError:
        return float("nan")
    return float(params[1])


def _carry_effect(table: pd.DataFrame, delta: dict[str, FloatArray], t_min: FloatArray) -> float:
    """持ち越しの区間の Δ の平均を、walk + leave + (1|subject) で比べた leave の 95% CI の下限。"""
    in_ep = (t_min >= ENDPOINT_MIN[0]) & (t_min < ENDPOINT_MIN[1])
    data = table.assign(carry=[float(delta[s][in_ep].mean()) for s in table["session"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        result = smf.mixedlm("carry ~ walk + leave", data, groups=data["subject"]).fit(
            reml=True, method=["bfgs", "lbfgs", "powell"]
        )
    return float(result.conf_int().loc["leave"].iloc[0])


@dataclass
class Tally:
    green_hits: dict[float, int]
    carry_hits: dict[float, int]
    tau_c_hat: dict[tuple[float, int], list[float]]


def run(
    subjects: Sequence[int],
    reps: int,
    greens: Sequence[float],
    taus: Sequence[float],
    noise_sd: float | None,
    slow_sd: float,
) -> None:
    measured = _window_noise_sd()
    print(f"60 秒窓の ln(HF) の推定誤差の SD（合成の拍で測定）: {measured:.3f}")
    noise_sd = measured if noise_sd is None else noise_sd
    print(
        f"追跡の試算に使う値: 窓の誤差 SD {noise_sd:.3f}、ゆっくりした揺らぎの SD {slow_sd:.2f}\n"
    )
    rows = []
    for n in subjects:
        tally = Tally({g: 0 for g in greens}, {t: 0 for t in taus}, {})
        for g in greens:
            for k in range(reps):
                study = simulate_study(
                    Design(n_subjects=n), TrueParams(beta_green=g), seed=1000 + k
                )
                curves = [session_curve(r) for r in study.recordings]
                sit_leave = [c for c in curves if c.recording.condition == "sit-leave"]
                mean = np.nanmean(np.vstack([c.delta_lnhf for c in sit_leave]), axis=0)
                tau = fit_response(sit_leave[0].t_s, mean).tau_s
                table = session_table(curves, tau_s=tau)
                tally.green_hits[g] += fit_green_effect(table, covariates=True).ci_low > 0
                if g != greens[0]:
                    continue  # 持ち越しは緑の真の効果が既定（最初の値）のときだけ試算する
                rng = np.random.default_rng(2000 + k)
                for tau_c in taus:
                    t_min, delta = _tracking(study, tau_c, noise_sd, slow_sd, rng)
                    tally.carry_hits[tau_c] += _carry_effect(table, delta, t_min) > 0
                    if tau_c == 0:
                        continue
                    leave = [delta[r.session] for r in study.recordings if r.leave]
                    for end in (30, 60):
                        tally.tau_c_hat.setdefault((tau_c, end), []).append(
                            _fit_tau_c(t_min, leave, end)
                        )
        rows.append((n, tally))
        print(f"被験者 {n} 人: 完了", flush=True)

    print(f"\n## 検出力（{reps} 回の繰り返し、被験者 × 4 条件 × 2 回）\n")
    head = ["被験者", "セッション"]
    head += [f"緑 {g:+.1f}" for g in greens]
    head += [("持ち越しなし（偽陽性）" if t == 0 else f"持ち越し τc={t:.0f} 分") for t in taus]
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for n, tally in rows:
        cells = [str(n), str(n * 8)]
        cells += [f"{tally.green_hits[g] / reps:.0%}" for g in greens]
        cells += [f"{tally.carry_hits[t] / reps:.0%}" for t in taus]
        print("| " + " | ".join(cells) + " |")

    print("\n## τc の推定（中央値と、繰り返しの 2.5〜97.5 パーセンタイル。上限 600 分）\n")
    keys = sorted({key for _, tally in rows for key in tally.tau_c_hat})
    head = ["被験者"] + [f"真 {t:.0f} 分・追跡 {e} 分" for t, e in keys]
    print("| " + " | ".join(head) + " |")
    print("|" + " --- |" * len(head))
    for n, tally in rows:
        cells = [str(n)]
        for key in keys:
            v = np.array(tally.tau_c_hat[key])
            v = v[np.isfinite(v)]
            lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
            cells.append(f"{med:.0f}（{lo:.0f}〜{hi:.0f}）")
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--subjects", type=int, nargs="+", default=[8, 12, 16, 24, 32])
    parser.add_argument("--greens", type=float, nargs="+", default=[0.6, 0.3])
    parser.add_argument("--tau-c", type=float, nargs="+", default=[0.0, 15.0, 30.0, 60.0])
    parser.add_argument("--noise-sd", type=float, default=None, help="窓の誤差 SD（既定は測定値）")
    parser.add_argument("--slow-sd", type=float, default=SLOW_OU_SD)
    args = parser.parse_args()
    run(args.subjects, args.reps, args.greens, args.tau_c, args.noise_sd, args.slow_sd)


if __name__ == "__main__":
    main()
