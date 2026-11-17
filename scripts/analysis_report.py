"""合成データで、解析基盤が真のパラメータを取り戻せるかを確かめ、Markdown の表を出す。

    uv run python scripts/analysis_report.py [--reps 20]

API は使わない。乱数の種は固定（詳細は種 7、繰り返しは種 100〜100+reps-1）。
"""

import argparse

import numpy as np

from vmgs.analysis.mixed import fit_green_effect, session_table
from vmgs.analysis.response import ResponseEstimate, fit_group_response
from vmgs.analysis.study import SessionCurve, session_curve
from vmgs.analysis.synthetic import Design, SyntheticStudy, simulate_study

CONDITIONS = ("sit-stay", "walk-stay", "sit-leave", "walk-leave")


def _analyze(
    study: SyntheticStudy, n_boot: int
) -> tuple[dict[str, tuple[ResponseEstimate, float]], list[SessionCurve]]:
    curves = [session_curve(r) for r in study.recordings]
    groups = {}
    for cond in CONDITIONS:
        members = [c for c in curves if c.recording.condition == cond]
        true_amp = float(
            np.mean([study.session_truth[c.recording.session].amplitude for c in members])
        )
        groups[cond] = (fit_group_response(members, n_boot=n_boot, seed=0), true_amp)
    return groups, curves


def detail(seed: int) -> None:
    study = simulate_study(Design(), seed=seed)
    t = study.truth
    groups, curves = _analyze(study, n_boot=1000)
    print(f"## 1 回の詳細（種 {seed}、{len(study.recordings)} セッション）\n")
    print("### 問い①: R(t) = offset + A(1 - exp(-t/τ))\n")
    print(
        f"| 条件 | τ [s]（真 {t.tau_s:.0f}） | 95% CI | A（真） | 95% CI | offset（真） | t95 [s] |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for cond, (est, true_amp) in groups.items():
        f, ci = est.fit, est.ci
        true_off = t.beta_walk_step if cond.startswith("walk") else 0.0
        print(
            f"| {cond} | {f.tau_s:.0f} | {ci['tau_s'][0]:.0f}〜{ci['tau_s'][1]:.0f} "
            f"| {f.amplitude:.2f}（{true_amp:.2f}） "
            f"| {ci['amplitude'][0]:.2f}〜{ci['amplitude'][1]:.2f} "
            f"| {f.offset:.2f}（{true_off:.2f}） | {f.t95_s:.0f} |"
        )
    tau_hat = groups["sit-leave"][0].fit.tau_s
    table = session_table(curves, tau_s=tau_hat)
    print(f"\n### 問い②: 到達値の混合効果モデル（τ は sit-leave の推定 {tau_hat:.0f} s に固定）\n")
    print("| 項 | 真の値 | 共変量あり | 共変量なし |")
    print("| --- | --- | --- | --- |")
    adj = fit_green_effect(table, covariates=True)
    naive = fit_green_effect(table, covariates=False)
    truth = {
        "walk": 0.0,
        "leave": t.beta_leave,
        "green": t.beta_green,
        "pet_c": t.beta_pet,
        "noise_db": t.beta_noise,
    }
    for term, true in truth.items():
        n = naive.params.get(term)
        naive_cell = f"{n:+.3f}" if n is not None else "—"
        print(f"| {term} | {true:+.3f} | {adj.params[term]:+.3f} | {naive_cell} |")
    print(
        f"\n緑の効果の 95% CI: 共変量あり {adj.ci_low:+.2f}〜{adj.ci_high:+.2f}、"
        f"共変量なし {naive.ci_low:+.2f}〜{naive.ci_high:+.2f}"
        f"（収束: {adj.converged} / {naive.converged}、"
        f"被験者の分散: {adj.subject_var:.4f} / {naive.subject_var:.4f}）\n"
    )


def replicate(reps: int) -> None:
    taus, amps_err, tau_cover, amp_cover = [], [], 0, 0
    adj_g, naive_g, adj_cover, naive_cover, not_converged, boundary = [], [], 0, 0, 0, 0
    truth = None
    for k in range(reps):
        study = simulate_study(Design(), seed=100 + k)
        truth = study.truth
        groups, curves = _analyze(study, n_boot=300)
        est, true_amp = groups["sit-leave"]
        taus.append(est.fit.tau_s)
        amps_err.append(est.fit.amplitude - true_amp)
        tau_cover += est.ci["tau_s"][0] <= truth.tau_s <= est.ci["tau_s"][1]
        amp_cover += est.ci["amplitude"][0] <= true_amp <= est.ci["amplitude"][1]
        table = session_table(curves, tau_s=est.fit.tau_s)
        adj = fit_green_effect(table, covariates=True)
        naive = fit_green_effect(table, covariates=False)
        not_converged += (not adj.converged) + (not naive.converged)
        boundary += adj.on_boundary + naive.on_boundary
        adj_g.append(adj.estimate)
        naive_g.append(naive.estimate)
        adj_cover += adj.ci_low <= truth.beta_green <= adj.ci_high
        naive_cover += naive.ci_low <= truth.beta_green <= naive.ci_high
    assert truth is not None
    print(f"## 種を変えた繰り返し（{reps} 回、種 100〜{100 + reps - 1}）\n")
    print("| 量 | 真の値 | 推定の平均 | 推定の SD | 95% CI が真の値を含んだ回数 |")
    print("| --- | --- | --- | --- | --- |")
    print(
        f"| τ（sit-leave） | {truth.tau_s:.0f} s | {np.mean(taus):.0f} s | {np.std(taus):.0f} s "
        f"| {tau_cover}/{reps} |"
    )
    print(
        f"| A − 真の A（sit-leave） | 0 | {np.mean(amps_err):+.3f} | {np.std(amps_err):.3f} "
        f"| {amp_cover}/{reps} |"
    )
    print(
        f"| 緑の効果（共変量あり） | {truth.beta_green:+.2f} | {np.mean(adj_g):+.3f} "
        f"| {np.std(adj_g):.3f} | {adj_cover}/{reps} |"
    )
    print(
        f"| 緑の効果（共変量なし） | {truth.beta_green:+.2f} | {np.mean(naive_g):+.3f} "
        f"| {np.std(naive_g):.3f} | {naive_cover}/{reps} |"
    )
    print(
        f"\n混合効果モデル: 収束しなかった当てはめ {not_converged}/{2 * reps}、"
        f"被験者の分散が 0 の境界に近かった当てはめ {boundary}/{2 * reps}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    detail(args.seed)
    replicate(args.reps)


if __name__ == "__main__":
    main()
