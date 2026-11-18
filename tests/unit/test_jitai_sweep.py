"""vmgs sweep（条件 × seed を回し、対にした差の信頼区間で比べる）の集計と再開の振る舞い。"""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest
from scipy import stats

from vmgs.jitai import loop, sweep


def test_平均と95パーセント信頼区間は_t分布で求める() -> None:
    s = sweep.mean_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    half = stats.t.ppf(0.975, 4) * (2.5**0.5) / 5**0.5
    assert s.n == 5
    assert s.mean == pytest.approx(3.0)
    assert s.lo == pytest.approx(3.0 - half)
    assert s.hi == pytest.approx(3.0 + half)


def test_1件だけなら信頼区間は出さない() -> None:
    s = sweep.mean_ci([2.0])
    assert (s.n, s.mean, s.lo, s.hi) == (1, 2.0, None, None)


def test_対の差はseedで対応させ_片方に無いseedは除く() -> None:
    base = {0: 1.0, 1: 2.0, 2: 3.0, 9: 100.0}
    other = {2: 5.0, 0: 2.0, 1: 2.5}  # 順が違い、seed 9 が無い
    s, better = sweep.paired_diff(base, other)
    assert s.n == 3
    assert s.mean == pytest.approx((1.0 + 0.5 + 2.0) / 3)
    assert better == 3  # 差が正（大きい方が良い）の seed の数


def test_小さい方が良い指標では_差が負のseedを良くなったと数える() -> None:
    _, better = sweep.paired_diff({0: 10.0, 1: 10.0}, {0: 8.0, 1: 12.0}, higher_is_better=False)
    assert better == 1


def _fake_loop(calls: list[argparse.Namespace]):  # type: ignore[no-untyped-def]
    def run(args: argparse.Namespace, weather_dir: Path) -> int:
        calls.append(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "round": r,
                "demand": "一様（既定）" if r == 0 else "集計",
                "demand_spots": 0,
                "demand_l1_from_prev": None,
                "decisions": 10,
                "interventions": 1 + r + args.seed,
                "intervention_rate": (1 + r + args.seed) / 10,
                "hyperstress_windows": 100 - r,
                "episode_burden": 1000.0 - r,
            }
            for r in range(args.rounds + 1)
        ]
        meta = {
            "stage": str(args.stage),
            "scenario": args.scenario,
            "workers": args.workers,
            "seed": args.seed,
            "pods": args.pods,
            "policy": args.policy,
            "rain_cover": args.rain_cover,
            "rounds": rows,
        }
        (args.out_dir / "summary.json").write_text(json.dumps(meta), encoding="utf-8")
        return 0

    return run


def _job(tmp_path: Path, **kw: Any) -> sweep.Job:
    base: dict[str, Any] = {
        "stage": tmp_path / "stage.json",
        "scenario": "normal",
        "policy": "greedy",
        "pods": 3,
        "seed": 0,
        "workers": 30,
        "rounds": 1,
        "out_dir": tmp_path / "out",
    }
    base.update(kw)
    return sweep.Job(**base)


def test_同じ引数の実行は回し直さず_違えば回し直す(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(loop, "cmd_loop", _fake_loop(calls))
    rows = sweep.run_job(_job(tmp_path), tmp_path)
    assert len(calls) == 1 and [r["round"] for r in rows] == [0, 1]
    # Given 同じ引数 / When もう一度 / Then 回し直さず、前の結果を読む
    assert sweep.run_job(_job(tmp_path), tmp_path) == rows
    assert len(calls) == 1
    # 引数（ワーカー数）が違えば回し直す
    sweep.run_job(_job(tmp_path, workers=40), tmp_path)
    assert len(calls) == 2
    # 方式が違っても回し直す
    sweep.run_job(_job(tmp_path, workers=40, policy="coverage"), tmp_path)
    assert len(calls) == 3 and calls[-1].policy == "coverage"
    # 雨よけの有無が違っても回し直し、loop に渡す
    sweep.run_job(_job(tmp_path, workers=40, policy="coverage", rain_cover=False), tmp_path)
    assert len(calls) == 4 and calls[-1].rain_cover is False


def test_雨よけの記録が無い古い結果は雨よけ無しとみなし_既定の雨よけ有りでは回し直す(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(loop, "cmd_loop", _fake_loop(calls))
    # Given 雨よけ無しの古い結果（summary.json に rain_cover が無い）
    sweep.run_job(_job(tmp_path, rain_cover=False), tmp_path)
    meta_path = tmp_path / "out" / "summary.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    del meta["rain_cover"]
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    # When 既定（雨よけ有り）の Job で回す / Then 回し直し、雨よけ有りを loop に渡す
    assert _job(tmp_path).rain_cover is True
    sweep.run_job(_job(tmp_path), tmp_path)
    assert len(calls) == 2 and calls[-1].rain_cover is True


def test_集計は周の差と_方式と台数の差を_seedで対にして出す() -> None:
    rows = [
        {
            "scenario": "normal",
            "policy": pol,
            "pods": p,
            "seed": s,
            "round": r,
            "intervention_rate": v,
        }
        for pol, p, s, r, v in [
            ("greedy", 3, 0, 0, 0.10),
            ("greedy", 3, 0, 1, 0.12),
            ("greedy", 3, 1, 0, 0.20),
            ("greedy", 3, 1, 1, 0.21),
            ("greedy", 5, 0, 0, 0.15),
            ("greedy", 5, 0, 1, 0.18),
            ("greedy", 5, 1, 0, 0.25),
            ("greedy", 5, 1, 1, 0.24),
            ("coverage", 3, 0, 0, 0.11),
            ("coverage", 3, 1, 0, 0.23),
            ("coverage", 3, 0, 1, 0.14),
            ("coverage", 3, 1, 1, 0.26),
        ]
    ]
    out = sweep.summarize_sweep(rows, metrics=[("intervention_rate", True)])
    by_round = {(d["policy"], d["pods"], d["metric"]): d for d in out["round_diffs"]}
    d = by_round[("greedy", 3, "intervention_rate")]
    assert d["n"] == 2 and d["mean"] == pytest.approx(0.015) and d["better"] == 2
    # 最初の条件（引数の順の最初の方式・最小の台数 = greedy・3 台）との差
    arms = {(d["policy"], d["pods"], d["round"]): d for d in out["arm_diffs"]}
    g5 = arms[("greedy", 5, 0)]
    assert g5["mean"] == pytest.approx(0.05)
    assert (g5["base_policy"], g5["base_pods"]) == ("greedy", 3)
    assert arms[("greedy", 5, 1)]["mean"] == pytest.approx((0.06 + 0.03) / 2)
    assert arms[("coverage", 3, 1)]["mean"] == pytest.approx((0.02 + 0.05) / 2)
    assert ("greedy", 3, 0) not in arms
    cell = {(c["policy"], c["pods"], c["round"]): c for c in out["cells"]}
    assert cell[("greedy", 3, 0)]["mean"] == pytest.approx(0.15)
