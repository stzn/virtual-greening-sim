"""vmgs loop（配車 → JITAI → 需要の書き戻し の繰り返し）。"""

import json
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli
from vmgs.contracts import JitaiSummary, Stage, jitai_summary_errors_against_stage

scenarios("jitai/loop.feature")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx() -> dict[str, Any]:
    return {}


@given(parsers.parse('サンプルの地点データ "{stage}" がある'))
def _g_stage(ctx: dict[str, Any], stage: str) -> None:
    ctx["stage"] = ROOT / stage


@when(parsers.parse('気象 "{scenario}"・ワーカー {n:d} 人で {rounds:d} 周回す'))
def _w_loop(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
    rounds: int,
) -> None:
    monkeypatch.setattr(cli, "WEATHER_DIR", ROOT / "data" / "weather")
    ctx["out"] = tmp_path / "loop"
    ctx["rounds"] = rounds
    ctx["code"] = cli.main(
        [
            "loop",
            "--stage",
            str(ctx["stage"]),
            "--scenario",
            scenario,
            "--workers",
            str(n),
            "--rounds",
            str(rounds),
            "--out-dir",
            str(ctx["out"]),
        ]
    )
    ctx["stderr"] = capsys.readouterr().err


@then(parsers.parse("終了コードは {code:d} になる"))
def _t_code(ctx: dict[str, Any], code: int) -> None:
    assert ctx["code"] == code, ctx["stderr"]


@then("周 0〜2 の地点データ・配車・JITAI・JITAI の集計のファイルがある")
def _t_files(ctx: dict[str, Any]) -> None:
    for r in range(ctx["rounds"] + 1):
        for name in ("stage", "dispatch", "jitai", "jitai-summary"):
            assert (ctx["out"] / f"{name}.r{r}.json").exists(), f"{name}.r{r}.json"
    assert (ctx["out"] / "summary.md").exists()
    assert (ctx["out"] / "summary.json").exists()


@then("周 1 以降の地点データの需要は、前の周の JITAI の需要の集計と一致する")
def _t_demand_chain(ctx: dict[str, Any]) -> None:
    for r in range(1, ctx["rounds"] + 1):
        stage = json.loads((ctx["out"] / f"stage.r{r}.json").read_text(encoding="utf-8"))
        jitai = json.loads((ctx["out"] / f"jitai.r{r - 1}.json").read_text(encoding="utf-8"))
        written = sorted(
            (s["id"], d["hour"], d["weight"]) for s in stage["spots"] for d in s["demand"]
        )
        cells = sorted((c["spot_id"], c["hour"], c["weight"]) for c in jitai["demand"]["cells"])
        assert written == cells and cells
        d = json.loads((ctx["out"] / f"dispatch.r{r}.json").read_text(encoding="utf-8"))
        assert "default-demand" not in d["generator"]


@then("JITAI の集計は契約に合い、地点データとの食い違いが無い")
def _t_summary_contract(ctx: dict[str, Any]) -> None:
    for r in range(ctx["rounds"] + 1):
        stage = Stage.model_validate_json((ctx["out"] / f"stage.r{r}.json").read_text("utf-8"))
        summary = JitaiSummary.model_validate_json(
            (ctx["out"] / f"jitai-summary.r{r}.json").read_text("utf-8")
        )
        assert jitai_summary_errors_against_stage(summary, stage) == []


@then("表には周 0〜2 の 3 行があり、介入率・過緊張の窓・負担・需要の地点数が並ぶ")
def _t_table(ctx: dict[str, Any]) -> None:
    rows = json.loads((ctx["out"] / "summary.json").read_text(encoding="utf-8"))["rounds"]
    assert [r["round"] for r in rows] == list(range(ctx["rounds"] + 1))
    for r in rows:
        assert {
            "intervention_rate",
            "hyperstress_windows",
            "episode_burden",
            "demand_spots",
        } <= set(r)
    md = (ctx["out"] / "summary.md").read_text(encoding="utf-8")
    assert "介入率" in md and "過緊張の窓" in md


@then("周 1 と周 2 の地点データは同じになる")
def _t_fixed_point(ctx: dict[str, Any]) -> None:
    # 需要が配車の結果に左右されなければ、周 1 の需要の集計は周 2 でも変わらない
    r1 = (ctx["out"] / "stage.r1.json").read_text(encoding="utf-8")
    r2 = (ctx["out"] / "stage.r2.json").read_text(encoding="utf-8")
    assert r1 == r2


@then("JITAI の需要は介入しない場合の、予定の空いた判定点で数えたと記録される")
def _t_basis(ctx: dict[str, Any]) -> None:
    for r in range(ctx["rounds"] + 1):
        jitai = json.loads((ctx["out"] / f"jitai.r{r}.json").read_text(encoding="utf-8"))
        assert jitai["demand"]["basis"] == "no_intervention_free_time"
