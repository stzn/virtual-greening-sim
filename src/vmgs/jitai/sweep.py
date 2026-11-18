"""`vmgs sweep`: 条件（気象 × 配車の方式 × ポッドの台数）× seed で `vmgs loop` を回し、対にした差の信頼区間で比べる。

合成のワーカーの組（seed）による違いが条件の違いより大きいことがあるので、良し悪しは
同じ seed どうしの差（対）の平均と 95% 信頼区間（t 分布）で見る。
- 周の差: 各条件で、周 r − 周 0（一様の需要 → 集計の需要）
- 方式・台数の差: 同じ気象・seed・周で、最初の（方式, 台数）との差

信頼区間はワーカーの組の違いだけを表し、舞台・気象・前提の値の不確かさは含まない。

例: uv run vmgs sweep --stage stage.json --scenarios normal heatwave --policies greedy coverage \\
        --pods 3 5 \\
        --seeds 10 --workers 300 --rounds 1 --out-dir runs/sweep --jobs 4
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 比べる指標と向き（True: 大きい方が良い）
METRICS: list[tuple[str, bool]] = [
    ("intervention_rate", True),
    ("hyperstress_windows", False),
    ("episode_burden", False),
]
METRIC_LABELS = {
    "intervention_rate": "介入率",
    "hyperstress_windows": "過緊張の窓",
    "episode_burden": "エピソードの負担",
}
ROW_FIELDS = [
    "scenario",
    "policy",
    "pods",
    "seed",
    "round",
    "demand",
    "demand_spots",
    "decisions",
    "interventions",
    "intervention_rate",
    "hyperstress_windows",
    "episode_burden",
]


@dataclass(frozen=True)
class Stat:
    n: int
    mean: float
    lo: float | None  # 95% CI の下限（2 件未満なら None）
    hi: float | None


def mean_ci(values: Sequence[float], level: float = 0.95) -> Stat:
    """平均と、t 分布による信頼区間。"""
    from scipy import stats

    n = len(values)
    if n == 0:
        return Stat(0, math.nan, None, None)
    mean = statistics.fmean(values)
    if n < 2:
        return Stat(n, mean, None, None)
    half = float(stats.t.ppf((1 + level) / 2, n - 1)) * statistics.stdev(values) / math.sqrt(n)
    return Stat(n, mean, mean - half, mean + half)


def paired_diff(
    base: Mapping[int, float], other: Mapping[int, float], *, higher_is_better: bool = True
) -> tuple[Stat, int]:
    """seed で対にした差（other − base）の平均と CI、良くなった seed の数。片方に無い seed は除く。"""
    seeds = sorted(set(base) & set(other))
    diffs = [other[s] - base[s] for s in seeds]
    better = sum(1 for d in diffs if (d > 0 if higher_is_better else d < 0))
    return mean_ci(diffs), better


@dataclass(frozen=True)
class Job:
    stage: Path
    scenario: str
    policy: str
    pods: int
    seed: int
    workers: int
    rounds: int
    out_dir: Path


def _matches(meta: Mapping[str, Any], job: Job) -> bool:
    return (
        meta.get("stage") == str(job.stage)
        and meta.get("scenario") == job.scenario
        and meta.get("workers") == job.workers
        and meta.get("seed") == job.seed
        and meta.get("pods") == job.pods
        and meta.get("policy", "greedy") == job.policy
        and len(meta.get("rounds", [])) == job.rounds + 1
    )


def run_job(job: Job, weather_dir: Path) -> list[dict[str, Any]]:
    """1 つの条件 × seed で vmgs loop を回し、周ごとの行を返す。同じ引数の結果があれば回し直さない。"""
    from vmgs.jitai import loop

    summary = job.out_dir / "summary.json"
    if summary.exists():
        meta = json.loads(summary.read_text(encoding="utf-8"))
        if _matches(meta, job):
            return list(meta["rounds"])
    code = loop.cmd_loop(
        argparse.Namespace(
            stage=job.stage,
            scenario=job.scenario,
            workers=job.workers,
            seed=job.seed,
            pods=job.pods,
            policy=job.policy,
            rounds=job.rounds,
            no_recovery=False,
            carryover_min=None,
            out_dir=job.out_dir,
        ),
        weather_dir,
    )
    if code != 0:
        raise RuntimeError(
            f"vmgs loop が失敗しました（{job.scenario}・{job.policy}・ポッド {job.pods}・seed {job.seed}）"
        )
    return list(json.loads(summary.read_text(encoding="utf-8"))["rounds"])


def _run_quiet(job: Job, weather_dir: Path) -> list[dict[str, Any]]:
    # 並列のときは各実行の表示を捨てる（まとめだけを出す）
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return run_job(job, weather_dir)


def summarize_sweep(
    rows: Sequence[Mapping[str, Any]], metrics: Sequence[tuple[str, bool]] = METRICS
) -> dict[str, Any]:
    """条件 × 周の平均と CI、周の差（周 r − 周 0）、方式・台数の差（最初の条件との差）を、seed で対にして出す。

    条件（arm）は（方式, 台数）。最初の条件は、行に現れた最初の方式と最小の台数。
    """

    def policy_of(r: Mapping[str, Any]) -> str:
        return str(r.get("policy", "greedy"))

    def series(scenario: str, arm: tuple[str, int], rnd: int, metric: str) -> dict[int, float]:
        return {
            int(r["seed"]): float(r[metric])
            for r in rows
            if r["scenario"] == scenario
            and (policy_of(r), int(r["pods"])) == arm
            and int(r["round"]) == rnd
        }

    scenarios = list(dict.fromkeys(str(r["scenario"]) for r in rows))
    policies = list(dict.fromkeys(policy_of(r) for r in rows))
    pods_list = sorted({int(r["pods"]) for r in rows})
    present = {(policy_of(r), int(r["pods"])) for r in rows}
    arms = [(pol, p) for pol in policies for p in pods_list if (pol, p) in present]
    rounds = sorted({int(r["round"]) for r in rows})
    cells, round_diffs, arm_diffs = [], [], []
    for metric, higher in metrics:
        for sc in scenarios:
            for arm in arms:
                keys = {"scenario": sc, "policy": arm[0], "pods": arm[1], "metric": metric}
                for rnd in rounds:
                    s = mean_ci(list(series(sc, arm, rnd, metric).values()))
                    cells.append(_stat_row(s, round=rnd, **keys))
                for rnd in rounds[1:]:
                    d, better = paired_diff(
                        series(sc, arm, rounds[0], metric),
                        series(sc, arm, rnd, metric),
                        higher_is_better=higher,
                    )
                    round_diffs.append(_stat_row(d, round=rnd, better=better, **keys))
            base = arms[0]
            for arm in arms[1:]:
                for rnd in rounds:
                    d, better = paired_diff(
                        series(sc, base, rnd, metric),
                        series(sc, arm, rnd, metric),
                        higher_is_better=higher,
                    )
                    arm_diffs.append(
                        _stat_row(
                            d,
                            scenario=sc,
                            policy=arm[0],
                            pods=arm[1],
                            base_policy=base[0],
                            base_pods=base[1],
                            round=rnd,
                            metric=metric,
                            better=better,
                        )
                    )
    return {"cells": cells, "round_diffs": round_diffs, "arm_diffs": arm_diffs}


def _stat_row(s: Stat, **keys: Any) -> dict[str, Any]:
    return {**keys, "n": s.n, "mean": s.mean, "lo": s.lo, "hi": s.hi}


def _fmt(metric: str, v: float | None) -> str:
    if v is None or math.isnan(v):
        return "—"
    return f"{v * 100:+.1f}pt" if metric == "intervention_rate" else f"{v:+.0f}"


def _fmt_level(metric: str, v: float | None) -> str:
    if v is None or math.isnan(v):
        return "—"
    return f"{v:.1%}" if metric == "intervention_rate" else f"{v:.0f}"


def _ci(metric: str, d: Mapping[str, Any], fmt: Any) -> str:
    if d["lo"] is None:
        return "—"
    return f"{fmt(metric, d['lo'])}〜{fmt(metric, d['hi'])}"


def _markdown(summary: Mapping[str, Any]) -> str:
    direction = "介入率は大きい方が良く、過緊張の窓・エピソードの負担は小さい方が良い。"
    lines = [
        "## 条件 × 周の平均（95% CI、t 分布）",
        "",
        "| 指標 | 気象 | 方式 | ポッド | 周 | seed | 平均 | 95% CI |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in summary["cells"]:
        m = c["metric"]
        lines.append(
            f"| {METRIC_LABELS.get(m, m)} | {c['scenario']} | {c['policy']} | {c['pods']} "
            f"| {c['round']} | {c['n']} | {_fmt_level(m, c['mean'])} | {_ci(m, c, _fmt_level)} |"
        )
    lines += [
        "",
        "## 周の差（周 r − 周 0、同じ seed で対）",
        "",
        direction + "「良くなった」は良い向きに動いた seed の数。",
        "",
        "| 指標 | 気象 | 方式 | ポッド | 周 | seed | 差の平均 | 95% CI | 良くなった |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for d in summary["round_diffs"]:
        m = d["metric"]
        lines.append(
            f"| {METRIC_LABELS.get(m, m)} | {d['scenario']} | {d['policy']} | {d['pods']} "
            f"| {d['round']} | {d['n']} | {_fmt(m, d['mean'])} | {_ci(m, d, _fmt)} "
            f"| {d['better']}/{d['n']} |"
        )
    lines += [
        "",
        "## 方式・台数の差（最初の条件との差、同じ seed・周で対）",
        "",
    ]
    if summary["arm_diffs"]:
        lines += [
            "| 指標 | 気象 | 条件 | 周 | seed | 差の平均 | 95% CI | 良くなった |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for d in summary["arm_diffs"]:
            m = d["metric"]
            arm = f"{d['base_policy']}・{d['base_pods']} 台 → {d['policy']}・{d['pods']} 台"
            lines.append(
                f"| {METRIC_LABELS.get(m, m)} | {d['scenario']} | {arm} "
                f"| {d['round']} | {d['n']} | {_fmt(m, d['mean'])} | {_ci(m, d, _fmt)} "
                f"| {d['better']}/{d['n']} |"
            )
    else:
        lines.append("（条件（方式・台数）が 1 通りなので、比べる相手がありません）")
    return "\n".join(lines)


def add_sweep_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    sp = sub.add_parser(
        "sweep",
        help="気象 × 配車の方式 × ポッドの台数 × seed で vmgs loop を回し、対にした差の CI で比べる",
    )
    sp.add_argument("--stage", type=Path, required=True, help="舞台の stage.json（上書きしない）")
    sp.add_argument(
        "--scenarios", nargs="+", default=["normal"], choices=["heatwave", "normal", "rainy"]
    )
    sp.add_argument(
        "--policies",
        nargs="+",
        default=["coverage"],
        choices=["greedy", "coverage"],
        help="配車の方式（複数可。最初の方式が比べる基準）",
    )
    sp.add_argument("--pods", nargs="+", type=int, default=[3], help="ポッドの台数（複数可）")
    sp.add_argument("--seeds", type=int, default=10, help="seed 0〜N−1 を回す")
    sp.add_argument("--workers", type=int, default=300)
    sp.add_argument("--rounds", type=int, default=1, help="vmgs loop の周回数")
    sp.add_argument("--jobs", type=int, default=1, help="並列に回す数")
    sp.add_argument("--out-dir", type=Path, required=True)


def cmd_sweep(args: argparse.Namespace, weather_dir: Path) -> int:
    if args.seeds < 1 or args.rounds < 0 or args.jobs < 1 or min(args.pods) < 1:
        print(
            "--seeds・--jobs・--pods は 1 以上、--rounds は 0 以上にしてください", file=sys.stderr
        )
        return 1
    pods_list = list(dict.fromkeys(args.pods))
    policies = list(dict.fromkeys(args.policies))
    jobs = [
        Job(
            stage=args.stage,
            scenario=sc,
            policy=pol,
            pods=p,
            seed=s,
            workers=args.workers,
            rounds=args.rounds,
            out_dir=args.out_dir / f"{sc}-{pol}-p{p}" / f"seed{s}",
        )
        for sc in args.scenarios
        for pol in policies
        for p in pods_list
        for s in range(args.seeds)
    ]
    try:
        if args.jobs == 1:
            results = [_run_quiet(j, weather_dir) for j in jobs]
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as ex:
                results = list(ex.map(_run_quiet, jobs, [weather_dir] * len(jobs)))
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    rows = [
        {"scenario": j.scenario, "policy": j.policy, "pods": j.pods, "seed": j.seed, **r}
        for j, res in zip(jobs, results, strict=True)
        for r in res
    ]
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize_sweep(rows)
    meta = {
        "stage": str(args.stage),
        "scenarios": args.scenarios,
        "policies": policies,
        "pods": pods_list,
        "seeds": args.seeds,
        "workers": args.workers,
        "rounds": args.rounds,
        **summary,
    }
    (out / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    table = _markdown(summary)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    return 0
