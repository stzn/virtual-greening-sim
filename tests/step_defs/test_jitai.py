"""JITAI の模擬（vmgs jitai）。"""

import json
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli

scenarios("jitai/jitai.feature")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx() -> dict[str, Any]:
    return {}


@given(parsers.parse('サンプルの舞台 "{stage}" と配車 "{dispatch}" がある'))
def _g_inputs(ctx: dict[str, Any], stage: str, dispatch: str) -> None:
    ctx["stage"], ctx["dispatch"] = ROOT / stage, ROOT / dispatch


@when(parsers.parse('気象 "{scenario}" でワーカー {n:d} 人の JITAI を回す'))
def _w_run(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
) -> None:
    monkeypatch.setattr(cli, "WEATHER_DIR", ROOT / "data" / "weather")
    out = tmp_path / "jitai.json"
    ctx["code"] = cli.main(
        [
            "jitai",
            "--stage",
            str(ctx["stage"]),
            "--dispatch",
            str(ctx["dispatch"]),
            "--scenario",
            scenario,
            "--workers",
            str(n),
            "--seed",
            "0",
            "--out",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    ctx["stdout"], ctx["stderr"] = captured.out, captured.err
    ctx["raw"] = out.read_text(encoding="utf-8") if out.exists() else ""
    ctx["result"] = json.loads(ctx["raw"]) if ctx["raw"] else {}


def _records(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = ctx["result"]["records"]
    return records


def _hour(rec: dict[str, Any]) -> int:
    return int(rec["time"].split(":")[0])


@then(parsers.parse("終了コードは {code:d} になる"))
def _t_code(ctx: dict[str, Any], code: int) -> None:
    assert ctx["code"] == code, ctx["stderr"]


@then("介入の記録には、時刻・ワーカー・結果・見送った段階と理由がある")
def _t_record_keys(ctx: dict[str, Any]) -> None:
    records = _records(ctx)
    assert records
    for r in records:
        assert {"time", "worker_id", "outcome", "stage", "reason"} <= set(r)
        assert r["outcome"] in {"intervened", "skipped"}
        if r["outcome"] == "skipped":
            assert r["stage"] in {"calendar", "pods", "pet"}
            assert r["reason"]
        else:
            assert r["stage"] is None and r["reason"] is None


@then("介入した記録には、ポッドと地点と道のりがある")
def _t_intervened_fields(ctx: dict[str, Any]) -> None:
    for r in _records(ctx):
        if r["outcome"] == "intervened":
            assert r["pod_id"] and r["spot_id"]
            assert 0 <= r["walk_m"] <= 400


@then("介入した記録が 1 件以上ある")
def _t_some_interventions(ctx: dict[str, Any]) -> None:
    assert any(r["outcome"] == "intervened" for r in _records(ctx))


@then("集計の介入率は、介入の件数を判定の件数で割った値になる")
def _t_rate(ctx: dict[str, Any]) -> None:
    s = ctx["result"]["summary"]
    records = _records(ctx)
    assert s["decisions"] == len(records)
    assert s["interventions"] == sum(r["outcome"] == "intervened" for r in records)
    assert s["intervention_rate"] == pytest.approx(s["interventions"] / s["decisions"])


@then("段階別の見送り件数の合計は、判定の件数から介入の件数を引いた値になる")
def _t_skips(ctx: dict[str, Any]) -> None:
    s = ctx["result"]["summary"]
    assert sum(s["skipped_by_stage"].values()) == s["decisions"] - s["interventions"]
    assert sum(s["skipped_by_reason"].values()) == s["decisions"] - s["interventions"]


@then("11〜14 時の外の判定では、介入していない")
def _t_outside_hours(ctx: dict[str, Any]) -> None:
    outside = [r for r in _records(ctx) if not 11 <= _hour(r) <= 14]
    assert outside
    assert all(r["outcome"] == "skipped" for r in outside)


@then(parsers.parse('段階 "{stage}" の理由 "{reason}" で見送った記録がある'))
def _t_reason(ctx: dict[str, Any], stage: str, reason: str) -> None:
    assert any(r["stage"] == stage and r["reason"] == reason for r in _records(ctx))


@then("介入した記録は 1 件もない")
def _t_no_interventions(ctx: dict[str, Any]) -> None:
    assert not any(r["outcome"] == "intervened" for r in _records(ctx))
    assert ctx["result"]["summary"]["interventions"] == 0


@then("出力に HRV の値が含まれない")
def _t_no_hrv(ctx: dict[str, Any]) -> None:
    raw = ctx["raw"].lower()
    for word in ("ln_hf", "lnhf", "hrv", "rr_ms", "z_score", "baseline"):
        assert word not in raw, word


@then("需要の集計は地点・時刻・件数だけで、3 人未満の地点・時刻を含まない")
def _t_demand(ctx: dict[str, Any]) -> None:
    demand = ctx["result"]["demand"]
    assert demand["min_workers"] == 3
    for cell in demand["cells"]:
        assert set(cell) == {"spot_id", "hour", "weight"}
        assert cell["weight"] >= 3
