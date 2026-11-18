"""地点データの地点での複数ポッドの配車（vmgs dispatch）。"""

import re
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs import cli
from vmgs.contracts import Dispatch, Stage, dispatch_errors_against_stage

scenarios("fleet/dispatch.feature")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def ctx() -> dict[str, Any]:
    return {}


@given(parsers.parse('サンプルの地点データ "{path}" がある'))
def _g_stage(ctx: dict[str, Any], path: str) -> None:
    ctx["stage"] = ROOT / path


def _run(
    ctx: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    out = tmp_path / "dispatch.json"
    ctx["out"] = out
    ctx["code"] = cli.main(["dispatch", "--stage", str(ctx["stage"]), "--out", str(out), *args])
    captured = capsys.readouterr()
    ctx["stdout"], ctx["stderr"] = captured.out, captured.err


@when(parsers.parse('気象 "{scenario}" でポッド {n:d} 台を "{policy}" で配車し、書き出す'))
def _w_dispatch(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
    policy: str,
) -> None:
    monkeypatch.setattr(cli, "WEATHER_DIR", ROOT / "data" / "weather")
    _run(ctx, tmp_path, capsys, ["--scenario", scenario, "--pods", str(n), "--policy", policy])


@when(parsers.parse('気象 "{scenario}" でポッド {n:d} 台を "{policy}" で配車し、比較表も出す'))
def _w_compare(
    ctx: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    n: int,
    policy: str,
) -> None:
    monkeypatch.setattr(cli, "WEATHER_DIR", ROOT / "data" / "weather")
    _run(
        ctx,
        tmp_path,
        capsys,
        ["--scenario", scenario, "--pods", str(n), "--policy", policy, "--compare"],
    )


@then(parsers.parse("終了コードは {code:d} になる"))
def _t_code(ctx: dict[str, Any], code: int) -> None:
    assert ctx["code"] == code, ctx["stderr"]


@then("書き出した配車は契約 v1 に合い、地点データとの食い違いが無い")
def _t_contract(ctx: dict[str, Any]) -> None:
    dispatch = Dispatch.model_validate_json(ctx["out"].read_text(encoding="utf-8"))
    stage = Stage.model_validate_json(ctx["stage"].read_text(encoding="utf-8"))
    assert dispatch_errors_against_stage(dispatch, stage) == []
    ctx["dispatch"], ctx["stage_obj"] = dispatch, stage


@then("各ポッドの時刻表は 0〜23 時の 24 件ある")
def _t_timeline(ctx: dict[str, Any]) -> None:
    for pod in ctx["dispatch"].pods:
        assert [s.hour for s in pod.timeline] == list(range(24))


@then("快適度は全地点・全時刻の PET を含む")
def _t_comfort(ctx: dict[str, Any]) -> None:
    cells = {(c.spot_id, c.hour) for c in ctx["dispatch"].comfort}
    assert cells == {(s.id, h) for s in ctx["stage_obj"].spots for h in range(24)}


def _row(stdout: str, name: str) -> list[str]:
    for line in stdout.splitlines():
        cols = [c.strip() for c in line.strip("|").split("|")]
        if cols and cols[0] == name:
            return cols
    raise AssertionError(f"比較表に {name} の行が無い:\n{stdout}")


@then(parsers.parse('比較表に "{a}"・"{b}"・"{c}"・"{d}" の行がある'))
def _t_rows(ctx: dict[str, Any], a: str, b: str, c: str, d: str) -> None:
    for name in (a, b, c, d):
        _row(ctx["stdout"], name)


@then("需要地点の PET は、貪欲法が固定配置以下で、ポッドなしより低い")
def _t_pet_order(ctx: dict[str, Any]) -> None:
    pet = {n: float(_row(ctx["stdout"], n)[1]) for n in ("no_pods", "greedy", "fixed")}
    assert pet["greedy"] <= pet["fixed"] < pet["no_pods"]


@then(parsers.parse('エラーに "{text}" が含まれる'))
def _t_error(ctx: dict[str, Any], text: str) -> None:
    assert re.search(re.escape(text), ctx["stderr"]), ctx["stderr"]
    assert not ctx["out"].exists()
