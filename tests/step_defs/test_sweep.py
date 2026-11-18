"""vmgs sweep（条件 × seed の比較）。"""

import csv
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli

scenarios("jitai/sweep.feature")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx() -> dict[str, Any]:
    return {}


@given(parsers.parse('サンプルの舞台 "{stage}" がある'))
def _g_stage(ctx: dict[str, Any], stage: str) -> None:
    ctx["stage"] = ROOT / stage


def _run(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
) -> None:
    monkeypatch.setattr(cli, "WEATHER_DIR", ROOT / "data" / "weather")
    ctx["out"] = tmp_path / "sweep"
    ctx["code"] = cli.main(
        ["sweep", "--stage", str(ctx["stage"]), "--out-dir", str(ctx["out"]), *args]
    )
    ctx["stderr"] = capsys.readouterr().err


@when(
    parsers.parse(
        '気象 "{scenario}"・方式 "{policy}"・ポッド {p1:d} 台と {p2:d} 台・{seeds:d} seed・'
        "ワーカー {n:d} 人・{rounds:d} 周で比べる"
    )
)
def _w_sweep_pods(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    policy: str,
    p1: int,
    p2: int,
    seeds: int,
    n: int,
    rounds: int,
) -> None:
    _run(
        ctx,
        tmp_path,
        capsys,
        monkeypatch,
        [
            "--scenarios",
            scenario,
            "--policies",
            policy,
            "--pods",
            str(p1),
            str(p2),
            "--seeds",
            str(seeds),
            "--workers",
            str(n),
            "--rounds",
            str(rounds),
        ],
    )


@when(
    parsers.parse(
        '気象 "{scenario}"・方式 "{a}" と "{b}"・ポッド {p:d} 台・{seeds:d} seed・'
        "ワーカー {n:d} 人・{rounds:d} 周で比べる"
    )
)
def _w_sweep_policies(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    a: str,
    b: str,
    p: int,
    seeds: int,
    n: int,
    rounds: int,
) -> None:
    _run(
        ctx,
        tmp_path,
        capsys,
        monkeypatch,
        [
            "--scenarios",
            scenario,
            "--policies",
            a,
            b,
            "--pods",
            str(p),
            "--seeds",
            str(seeds),
            "--workers",
            str(n),
            "--rounds",
            str(rounds),
        ],
    )


@then(parsers.parse("終了コードは {code:d} になる"))
def _t_code(ctx: dict[str, Any], code: int) -> None:
    assert ctx["code"] == code, ctx["stderr"]


@then("実行ごとの表には 2 条件 × 2 seed × 2 周 の 8 行がある")
def _t_runs(ctx: dict[str, Any]) -> None:
    with (ctx["out"] / "runs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert {(r["pods"], r["seed"], r["round"]) for r in rows} == {
        (p, s, r) for p in ("1", "2") for s in ("0", "1") for r in ("0", "1")
    }


@then("実行ごとの表には方式 2 通り × 2 seed × 2 周 の 8 行がある")
def _t_runs_policies(ctx: dict[str, Any]) -> None:
    with (ctx["out"] / "runs.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {(r["policy"], r["seed"], r["round"]) for r in rows} == {
        (p, s, r) for p in ("greedy", "coverage") for s in ("0", "1") for r in ("0", "1")
    }
    assert len(rows) == 8


@then("まとめには周の差と方式・台数の差の表がある")
def _t_summary(ctx: dict[str, Any]) -> None:
    md = (ctx["out"] / "summary.md").read_text(encoding="utf-8")
    assert "周の差" in md and "方式・台数の差" in md and "95%" in md
    assert (ctx["out"] / "summary.json").exists()
