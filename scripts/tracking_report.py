"""追跡の解析（docs/pilot-protocol.md 4.2）を、拍から作った合成データで確かめ、Markdown の表を出す。

    uv run python scripts/tracking_report.py [--subjects 16] [--reps 10]
    uv run python scripts/tracking_report.py --beats B.csv --sessions S.csv [--events E.csv]

- 合成データ: 真の τc（15・30・60 分）ごとに `simulate_study(..., track_s=60 分)` を作り、
  τc の推定（セッション単位のブートストラップの 95% CI）と、持ち越しの leave の効果を出す。
  繰り返しでは、τc の CI が真の値を含んだ割合と、leave の効果の CI の下限が 0 より上だった割合を数える。
- 実データ: CSV のパスを渡すと同じ解析をする（個人の健康データはリポジトリに置かない。出すのは集計だけ）。

API は使わない。乱数の種は固定（真の τc ごとに、繰り返し k は種 500+k）。
"""

import argparse
from pathlib import Path

from vmgs.analysis.io import read_recordings
from vmgs.analysis.recording import Recording
from vmgs.analysis.synthetic import Design, TrueParams, simulate_study
from vmgs.analysis.tracking import (
    ENDPOINT_MIN,
    TRACK_END_MIN,
    CarryEffect,
    TauCEstimate,
    carry_table,
    fit_carry_effect,
    fit_tau_c,
    tracking_curve,
)

TRUE_TAU_C_MIN = (15.0, 30.0, 60.0)


def analyze(recordings: list[Recording], n_boot: int) -> tuple[TauCEstimate, CarryEffect, int]:
    curves = [tracking_curve(r) for r in recordings if r.tracked]
    removed = sum(int((c.delta_lnhf != c.delta_lnhf).sum()) for c in curves)  # NaN の窓
    return fit_tau_c(curves, n_boot=n_boot, seed=0), fit_carry_effect(carry_table(curves)), removed


def _synthetic(subjects: int, tau_c_min: float, seed: int) -> list[Recording]:
    design = Design(n_subjects=subjects, repeats=2, track_s=TRACK_END_MIN * 60.0)
    return simulate_study(design, TrueParams(tau_c_s=tau_c_min * 60.0), seed=seed).recordings


def synthetic_report(subjects: int, reps: int, n_boot: int) -> None:
    print(f"## 合成データ（被験者 {subjects} 人 × 4 条件 × 2 回、追跡 {TRACK_END_MIN} 分）\n")
    print(f"持ち越しの区間は接触の終わりから {ENDPOINT_MIN[0]}〜{ENDPOINT_MIN[1]} 分。\n")
    print("### 1 回（種 500）\n")
    print("| 真の τc [分] | 推定 τc | 95% CI | K | leave の効果 | 95% CI | 除いた窓 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for tau_c in TRUE_TAU_C_MIN:
        est, carry, removed = analyze(_synthetic(subjects, tau_c, 500), n_boot)
        print(
            f"| {tau_c:.0f} | {est.tau_c_min:.1f} | {est.ci[0]:.1f}〜{est.ci[1]:.1f} | {est.k:.2f} "
            f"| {carry.coef:+.3f} | {carry.ci[0]:+.3f}〜{carry.ci[1]:+.3f} | {removed} |"
        )
    if reps <= 0:
        return
    print(f"\n### 繰り返し（{reps} 回、種 500〜{500 + reps - 1}）\n")
    print(
        "| 真の τc [分] | τc の CI が真の値を含む | 推定 τc の中央値 | leave の効果の CI 下限 > 0 |"
    )
    print("| --- | --- | --- | --- |")
    for tau_c in TRUE_TAU_C_MIN:
        hits, detect, taus = 0, 0, []
        for k in range(reps):
            est, carry, _ = analyze(_synthetic(subjects, tau_c, 500 + k), n_boot)
            hits += est.ci[0] <= tau_c <= est.ci[1]
            detect += carry.ci[0] > 0
            taus.append(est.tau_c_min)
        taus.sort()
        print(f"| {tau_c:.0f} | {hits}/{reps} | {taus[len(taus) // 2]:.1f} | {detect}/{reps} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=int, default=16)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=300)
    parser.add_argument("--beats", type=Path, help="実データの拍の CSV")
    parser.add_argument("--sessions", type=Path, help="実データのセッションの CSV")
    parser.add_argument("--events", type=Path, help="実データの出来事の CSV（任意）")
    args = parser.parse_args()
    if args.beats or args.sessions:
        if not (args.beats and args.sessions):
            parser.error("--beats と --sessions は両方渡す")
        est, carry, removed = analyze(
            read_recordings(args.beats, args.sessions, events_path=args.events), args.n_boot
        )
        print(
            f"τc: {est.tau_c_min:.1f} 分（95% CI {est.ci[0]:.1f}〜{est.ci[1]:.1f}、{est.n_sessions} セッション）"
        )
        print(
            f"leave の効果（{ENDPOINT_MIN[0]}〜{ENDPOINT_MIN[1]} 分）: {carry.coef:+.3f}（95% CI {carry.ci[0]:+.3f}〜{carry.ci[1]:+.3f}）"
        )
        print(f"除いた窓: {removed}")
        return
    synthetic_report(args.subjects, args.reps, args.n_boot)


if __name__ == "__main__":
    main()
