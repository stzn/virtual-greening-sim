"""JITAI の模擬（vmgs jitai）。"""

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli
from vmgs.jitai import message

scenarios("jitai/jitai.feature")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx() -> dict[str, Any]:
    return {}


@given(parsers.parse('サンプルの地点データ "{stage}" と配車 "{dispatch}" がある'))
def _g_inputs(ctx: dict[str, Any], stage: str, dispatch: str) -> None:
    ctx["stage"], ctx["dispatch"] = ROOT / stage, ROOT / dispatch


def _run(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
    extra: list[str],
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
            *extra,
        ]
    )
    captured = capsys.readouterr()
    ctx["stdout"], ctx["stderr"] = captured.out, captured.err
    ctx["raw"] = out.read_text(encoding="utf-8") if out.exists() else ""
    ctx["result"] = json.loads(ctx["raw"]) if ctx["raw"] else {}


@when(parsers.parse('気象 "{scenario}" でワーカー {n:d} 人の JITAI を回す'))
def _w_run(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
) -> None:
    _run(ctx, tmp_path, capsys, monkeypatch, scenario, n, [])


@when(parsers.parse('気象 "{scenario}" でワーカー {n:d} 人の JITAI を雨よけ無しで回す'))
def _w_run_no_cover(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
) -> None:
    _run(ctx, tmp_path, capsys, monkeypatch, scenario, n, ["--no-rain-cover"])


@when(parsers.parse('気象 "{scenario}" でワーカー {n:d} 人の JITAI を通知文つきで回す'))
def _w_run_messages(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
) -> None:
    _run(ctx, tmp_path, capsys, monkeypatch, scenario, n, ["--messages"])


@when(parsers.parse('気象 "{scenario}" でワーカー {n:d} 人の JITAI を LLM の言い換えつきで回す'))
def _w_run_llm(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
) -> None:
    _run(ctx, tmp_path, capsys, monkeypatch, scenario, n, ["--messages", "--llm", "ollama:fake"])


@given("LLM は徒歩の分を 1 分多く読み違えて言い換える")
def _g_bad_llm(ctx: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # 実際の ollama は呼ばない。指示の「元の文」の徒歩の分だけを 1 分多くして返す偽の LLM
    def misread(prompt: str) -> str:
        line = next(x for x in prompt.splitlines() if x.startswith("元の文: "))
        text = line.removeprefix("元の文: ")
        out = re.sub(r"徒歩 (\d+) 分", lambda m: f"徒歩 {int(m.group(1)) + 1} 分", text)
        ctx.setdefault("llm_outputs", []).append(out)
        return out

    monkeypatch.setattr(message, "ollama_paraphraser", lambda model: misread)


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


@then(parsers.parse("記録の規則に、雨よけ{cover}・雨のしきい値 {mm:d} mm/h がある"))
def _t_rain_rule(ctx: dict[str, Any], cover: str, mm: int) -> None:
    assert ctx["result"]["rules"]["rain_cover"] is (cover == "有り")
    assert ctx["result"]["rules"]["rain_max_mm_per_h"] == float(mm)


@then("出力に HRV の値が含まれない")
def _t_no_hrv(ctx: dict[str, Any]) -> None:
    raw = ctx["raw"].lower()
    for word in ("ln_hf", "lnhf", "hrv", "rr_ms", "z_score", "baseline"):
        assert word not in raw, word


@then("需要の集計は徒歩圏の区画・時刻ごとで、3 人未満の区画・時刻は時刻ごとの合算にだけ含める")
def _t_demand(ctx: dict[str, Any]) -> None:
    demand = ctx["result"]["demand"]
    assert demand["min_workers"] == 3
    assert demand["zone_m"] == 400
    for cell in demand["cells"]:
        assert set(cell) == {"spot_id", "hour", "weight"}
    # 区画・時刻ごとの件数は 3 人以上のまとまりだけ（1 人 1 件以上なので件数も 3 以上）
    for zone in demand["zones"]:
        assert set(zone) == {"zone", "hour", "count", "spots"}
        assert zone["count"] >= 3
    # 落とした区画は時刻ごとの合算（件数と配る地点の数）だけで、人数や区画は出さない
    for res in demand["residual"]:
        assert set(res) == {"hour", "count", "spots"}
        assert res["count"] >= 3


@when("JITAI の需要を地点データに書き戻す")
def _w_writeback(ctx: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jitai = tmp_path / "jitai.json"
    ctx["demand_stage"] = tmp_path / "stage.demand.json"
    ctx["demand_code"] = cli.main(
        [
            "demand",
            "--stage",
            str(ctx["stage"]),
            "--jitai",
            str(jitai),
            "--out",
            str(ctx["demand_stage"]),
        ]
    )
    ctx["demand_stderr"] = capsys.readouterr().err


@then("書き戻しの終了コードは 0 になる")
def _t_writeback_code(ctx: dict[str, Any]) -> None:
    assert ctx["demand_code"] == 0, ctx["demand_stderr"]


@then("書き戻した地点データの需要は、JITAI の需要の集計と一致する")
def _t_writeback_matches(ctx: dict[str, Any]) -> None:
    stage = json.loads(ctx["demand_stage"].read_text(encoding="utf-8"))
    written = sorted((s["id"], d["hour"], d["weight"]) for s in stage["spots"] for d in s["demand"])
    cells = sorted((c["spot_id"], c["hour"], c["weight"]) for c in ctx["result"]["demand"]["cells"])
    assert written == cells and cells


@when("書き戻した地点データで配車する")
def _w_dispatch(ctx: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "dispatch.demand.json"
    code = cli.main(
        ["dispatch", "--stage", str(ctx["demand_stage"]), "--scenario", "normal", "--out", str(out)]
    )
    assert code == 0, capsys.readouterr().err
    ctx["dispatch_out"] = json.loads(out.read_text(encoding="utf-8"))


@then("配車は既定の需要を使っていない")
def _t_no_default_demand(ctx: dict[str, Any]) -> None:
    assert "default-demand" not in ctx["dispatch_out"]["generator"]


@when("同じ条件で、回復を反映せずに JITAI を回す")
def _w_no_recovery(ctx: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "jitai.norecovery.json"
    code = cli.main(
        [
            "jitai",
            "--stage",
            str(ctx["stage"]),
            "--dispatch",
            str(ctx["dispatch"]),
            "--scenario",
            "normal",
            "--workers",
            "30",
            "--seed",
            "0",
            "--no-recovery",
            "--out",
            str(out),
        ]
    )
    assert code == 0, capsys.readouterr().err
    ctx["no_recovery"] = json.loads(out.read_text(encoding="utf-8"))


@then("回復を反映したほうが、過緊張と判定した窓が少ない")
def _t_fewer_windows(ctx: dict[str, Any]) -> None:
    with_r = ctx["result"]["summary"]["recovery"]
    without = ctx["no_recovery"]["summary"]["recovery"]
    assert with_r["enabled"] and not without["enabled"]
    assert with_r["hyperstress_windows"] < without["hyperstress_windows"]
    assert with_r["episode_burden"] < without["episode_burden"]


@then("集計には回復の前提（持ち越しの分）が記録されている")
def _t_recovery_rules(ctx: dict[str, Any]) -> None:
    rules = ctx["result"]["rules"]
    assert rules["recovery"] is True
    assert rules["recovery_carryover_minutes"] == 30
    assert rules["recovery_pod_green"] == 0.8


def _intervened(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in _records(ctx) if r["outcome"] == "intervened"]


@then("介入した記録には、地点名と徒歩の分と休憩の分を含む通知文がある")
def _t_messages(ctx: dict[str, Any]) -> None:
    stage = json.loads(ctx["stage"].read_text(encoding="utf-8"))
    names = {s["id"]: s["name"] for s in stage["spots"]}
    done = _intervened(ctx)
    assert done
    for r in done:
        m = r["message"]
        assert m["source"] == "template"
        assert names[r["spot_id"]] in m["text"]
        walk_min = math.ceil(r["walk_m"] / 80)
        assert f"徒歩 {walk_min} 分" in m["text"]
        assert "10 分休みませんか" in m["text"]
        assert m["text"].startswith(r["time"])


@then("見送った記録には通知文が無い")
def _t_no_message_for_skips(ctx: dict[str, Any]) -> None:
    skipped = [r for r in _records(ctx) if r["outcome"] == "skipped"]
    assert skipped
    assert all(r["message"] is None for r in skipped)


@then("介入した記録の通知文はすべてテンプレートで、LLM の出力と外れた理由が記録されている")
def _t_fallback(ctx: dict[str, Any]) -> None:
    done = _intervened(ctx)
    assert done
    assert len(ctx["llm_outputs"]) == len(done)
    for r in done:
        m = r["message"]
        assert m["source"] == "template"
        assert m["llm_text"] in ctx["llm_outputs"]
        assert any(p.startswith(("unknown_number:", "missing_walk")) for p in m["llm_problems"])
    assert ctx["result"]["messages"]["llm"] == "ollama:fake"
    assert ctx["result"]["messages"]["fallbacks"] == len(done)
