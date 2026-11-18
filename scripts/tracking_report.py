"""追跡の解析（docs/pilot-protocol.md 4.2）を、拍から作った合成データで確かめ、Markdown の表を出す。

    uv run python scripts/tracking_report.py [--subjects 16] [--reps 10]
    uv run python scripts/tracking_report.py --review --subjects 16 24 32 \\
        --endpoints 5-25 10-30 20-40 [--reps 10] [--jobs 8]
    uv run python scripts/tracking_report.py --beats B.csv --sessions S.csv [--events E.csv]

- 合成データ: 真の τc（15・30・60 分）ごとに `simulate_study(..., track_s=60 分)` を作り、
  τc の推定（セッション単位のブートストラップの 95% CI）と、持ち越しの leave の効果を出す。
  繰り返しでは、τc の CI が真の値を含んだ割合と、leave の効果の CI の下限が 0 より上だった割合を数える。
- 見直し（--review）: 被験者の人数 × 持ち越しの区間ごとに、leave の効果の検出の割合を出す。
  同じ合成データから τc の当てはめは 1 回だけ行い、区間ごとに持ち越しの効果を出す。
  離れたことの効果が無い合成データ（応答の到達値を 0 にする）で、誤って検出する割合も出す。
- 実データ: CSV のパスを渡すと同じ解析をする（個人の健康データはリポジトリに置かない。出すのは集計だけ）。

API は使わない。乱数の種は固定（真の τc ごとに、繰り返し k は種 500+k）。
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
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
# 離れたことの効果が無い場合（応答の到達値が全条件で 0。持ち越しも 0）
NULL_PARAMS = {"beta_leave": 0.0, "beta_green": 0.0, "beta_pet": 0.0, "beta_noise": 0.0}


def analyze(recordings: list[Recording], n_boot: int) -> tuple[TauCEstimate, CarryEffect, int]:
    curves = [tracking_curve(r) for r in recordings if r.tracked]
    removed = sum(int((c.delta_lnhf != c.delta_lnhf).sum()) for c in curves)  # NaN の窓
    return fit_tau_c(curves, n_boot=n_boot, seed=0), fit_carry_effect(carry_table(curves)), removed


def _synthetic(
    subjects: int, tau_c_min: float, seed: int, *, null: bool = False
) -> list[Recording]:
    design = Design(n_subjects=subjects, repeats=2, track_s=TRACK_END_MIN * 60.0)
    extra = NULL_PARAMS if null else {}
    params = TrueParams(tau_c_s=tau_c_min * 60.0, **extra)
    return simulate_study(design, params, seed=seed).recordings


Endpoint = tuple[float, float]


def _review_one(
    unit: tuple[int, float | None, int, tuple[Endpoint, ...], int],
) -> tuple[int, float | None, bool | None, list[bool]]:
    """1 回分: (被験者, 真の τc（None は効果なし）, 種, 区間, ブートストラップ) →
    (被験者, 真の τc, τc の CI が真の値を含むか, 区間ごとの検出)。"""
    subjects, tau_c, seed, endpoints, n_boot = unit
    null = tau_c is None
    recs = _synthetic(subjects, 30.0 if null else tau_c, seed, null=null)  # type: ignore[arg-type]
    curves = [tracking_curve(r) for r in recs if r.tracked]
    hit: bool | None = None
    if not null:
        est = fit_tau_c(curves, n_boot=n_boot, seed=0)
        hit = bool(est.ci[0] <= tau_c <= est.ci[1])  # type: ignore[operator]
    detects = [fit_carry_effect(carry_table(curves, ep)).ci[0] > 0 for ep in endpoints]
    return subjects, tau_c, hit, detects


def review_report(
    subjects_list: list[int], endpoints: list[Endpoint], reps: int, n_boot: int, jobs: int
) -> None:
    taus: list[float | None] = [*TRUE_TAU_C_MIN, None]
    units = [
        (n, tau, 500 + k, tuple(endpoints), n_boot)
        for n in subjects_list
        for tau in taus
        for k in range(reps)
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_review_one, units))
    else:
        results = [_review_one(u) for u in units]
    print(
        f"## 持ち越しの区間と被験者数の見直し（合成データ、各 {reps} 回、種 500〜{500 + reps - 1}）\n"
    )
    print("leave の効果の 95% CI の下限が 0 より上だった回数。効果なしの列は誤検出。\n")
    head = " | ".join(f"τc {t:.0f} 分" for t in TRUE_TAU_C_MIN)
    print(f"| 被験者 | 区間 [分] | {head} | 効果なし（誤検出） |")
    print("| --- | --- |" + " --- |" * (len(TRUE_TAU_C_MIN) + 1))
    for n in subjects_list:
        for i, ep in enumerate(endpoints):
            cells = []
            for tau in taus:
                d = [r[3][i] for r in results if r[0] == n and r[1] == tau]
                cells.append(f"{sum(d)}/{len(d)}")
            print(f"| {n} | {ep[0]:.0f}〜{ep[1]:.0f} | " + " | ".join(cells) + " |")
    print("\n### τc の CI が真の値を含んだ回数（区間に依らない）\n")
    print(f"| 被験者 | {head} |")
    print("| --- |" + " --- |" * len(TRUE_TAU_C_MIN))
    for n in subjects_list:
        cells = []
        for tau in TRUE_TAU_C_MIN:
            h = [r[2] for r in results if r[0] == n and r[1] == tau]
            cells.append(f"{sum(bool(x) for x in h)}/{len(h)}")
        print(f"| {n} | " + " | ".join(cells) + " |")


def _endpoint(spec: str) -> Endpoint:
    lo, hi = (float(x) for x in spec.split("-"))
    if not lo < hi:
        raise argparse.ArgumentTypeError(f"区間は 開始-終了 [分]（開始 < 終了）: {spec}")
    return lo, hi


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
    parser.add_argument("--subjects", type=int, nargs="+", default=[16])
    parser.add_argument("--review", action="store_true", help="被験者数 × 持ち越しの区間の見直し")
    parser.add_argument(
        "--endpoints",
        type=_endpoint,
        nargs="+",
        default=[ENDPOINT_MIN],
        help="持ち越しの区間（例 10-30）。--review で使う",
    )
    parser.add_argument("--jobs", type=int, default=1)
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
    if args.review:
        review_report(args.subjects, args.endpoints, args.reps, args.n_boot, args.jobs)
        return
    for n in args.subjects:
        synthetic_report(n, args.reps, args.n_boot)


if __name__ == "__main__":
    main()
