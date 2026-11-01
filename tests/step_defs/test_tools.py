"""5ツールの契約。"""

from typing import Any

from pytest_bdd import parsers, scenarios, then, when

from tests.step_defs.conftest import Ctx, find_key, keys_of

scenarios("tools/tools.feature")


# ---------------- When ----------------
@when(parsers.parse("get_weather を lat={lat:f} lon={lon:f} hours_ahead={h:d} で呼ぶ"))
def _call_weather(ctx: Ctx, lat: float, lon: float, h: int) -> None:
    ctx.call_tool("get_weather", lat=lat, lon=lon, hours_ahead=h)


@when("get_plant_state を呼ぶ")
def _call_state(ctx: Ctx) -> None:
    ctx.call_tool("get_plant_state")


@when("get_plant_state を2回呼ぶ")
def _call_state_twice(ctx: Ctx) -> None:
    ctx.call_tool("get_plant_state")
    first = ctx.result
    ctx.call_tool("get_plant_state")
    ctx.results = [first, ctx.result]


@when(parsers.parse('move_to_spot を spot_id="{spot}" で呼ぶ'))
def _call_move(ctx: Ctx, spot: str) -> None:
    ctx.call_tool("move_to_spot", spot_id=spot)


@when(parsers.parse("irrigate を amount_ml={n:d} で呼ぶ"))
def _call_irrigate(ctx: Ctx, n: int) -> None:
    ctx.call_tool("irrigate", amount_ml=n)


@when(parsers.parse('log_research_note を text="{text}" tags=[{tags}] で呼ぶ'))
def _call_note(ctx: Ctx, text: str, tags: str) -> None:
    tag_list = [t.strip().strip('"') for t in tags.split(",") if t.strip()]
    ctx.call_tool("log_research_note", text=text, tags=tag_list)


@when("log_research_note を 1001文字の text で呼ぶ")
def _call_note_long(ctx: Ctx) -> None:
    ctx.call_tool("log_research_note", text="あ" * 1001)


@when("不正な引数で全ツールを順に呼ぶ")
def _call_all_garbage(ctx: Ctx) -> None:
    garbage: list[dict[str, Any]] = [
        {"lat": "x", "lon": None},
        {"unknown": 1},
        {"spot_id": 123},
        {"amount_ml": "many"},
        {"text": 42, "tags": "no"},
    ]
    ctx.results = []
    ctx.error = None
    try:
        for spec in ctx.toolbox.specs:
            for kwargs in garbage:
                ctx.results.append(ctx.toolbox.call(spec.name, **kwargs))
        ctx.results.append(ctx.toolbox.call("no_such_tool"))
    except Exception as e:  # noqa: BLE001  例外が漏れたことをテスト側で検出するため
        ctx.results.append({"raised": repr(e)})


# ---------------- Then ----------------
@then(parsers.parse("結果に {names} が含まれる"))
@then(parsers.parse("結果に現在の {names} が含まれる"))
def _has_keys(ctx: Ctx, names: str) -> None:
    assert ctx.result["ok"] is True, ctx.result
    for key in keys_of(names):
        assert find_key(ctx.result, key), f"{key} が結果にありません: {ctx.result}"


@then("結果に消費バッテリーが含まれる")
def _has_battery_used(ctx: Ctx) -> None:
    assert "battery_used_pct" in ctx.result


@then(parsers.parse("予報は {n:d} 件である"))
def _forecast_len(ctx: Ctx, n: int) -> None:
    assert len(ctx.result["forecast"]) == n


@then(parsers.parse("結果の時刻は {hhmm} である"))
def _result_time(ctx: Ctx, hhmm: str) -> None:
    assert ctx.result["time"] == hhmm


@then("2回の結果は同一である")
def _same(ctx: Ctx) -> None:
    assert ctx.results[0] == ctx.results[1]


@then("結果の直前の移動は無い")
def _no_last_move(ctx: Ctx) -> None:
    assert "last_move" in ctx.result
    assert ctx.result["last_move"] is None


@then(parsers.parse('結果の直前の移動は "{a}" から "{b}" へ {n:d}時間前 である'))
def _last_move(ctx: Ctx, a: str, b: str, n: int) -> None:
    assert ctx.result["last_move"] == {"from": a, "to": b, "hours_ago": n}


@then(parsers.parse("結果は 切り詰められ、要求 {r:d}ml・実際 {a:d}ml である"))
def _truncated_result(ctx: Ctx, r: int, a: int) -> None:
    res = ctx.result
    assert res["ok"] is True
    assert (res["truncated"], res["requested_ml"], res["applied_ml"]) == (True, r, a)


@then(parsers.parse('結果のメッセージに "{text}" が含まれる'))
def _result_message(ctx: Ctx, text: str) -> None:
    assert text in ctx.result["message"]


@then("結果に 切り詰め・要求量・実際の量 は含まれない")
def _no_truncation_keys(ctx: Ctx) -> None:
    assert not any(k in ctx.result for k in ("truncated", "requested_ml", "applied_ml", "message"))


@then("結果は成功である")
def _ok(ctx: Ctx) -> None:
    assert ctx.result["ok"] is True, ctx.result


@then(parsers.parse('結果の新スポットは "{spot}" である'))
def _new_spot(ctx: Ctx, spot: str) -> None:
    assert ctx.result["new_spot_id"] == spot


@then(parsers.parse('ツールはエラー "{code}" を返す'))
def _tool_error(ctx: Ctx, code: str) -> None:
    assert ctx.result["ok"] is False, ctx.result
    assert ctx.result["error"]["code"] == code


@then("水タンク残量は変化しない")
def _tank_unchanged(ctx: Ctx) -> None:
    assert ctx.world.plant.water_tank_ml == ctx.before["water_tank_ml"]


@then("エラーに現在のタンク残量が含まれる")
def _error_has_tank(ctx: Ctx) -> None:
    assert ctx.result["error"]["details"]["water_tank_ml"] == ctx.world.plant.water_tank_ml


@then(parsers.parse('研究ログファイルに "{text}" が追記されている'))
def _note_saved(ctx: Ctx, text: str) -> None:
    assert text in ctx.world.notes_path.read_text(encoding="utf-8")
    assert ctx.result["saved_to"] == str(ctx.world.notes_path)


@then("どのツールも例外を送出しない")
def _no_raise(ctx: Ctx) -> None:
    raised = [r for r in ctx.results if "raised" in r]
    assert not raised, raised
    assert all(r["ok"] is False for r in ctx.results)
    assert ctx.results, "結果が空"


@then(parsers.parse('エラーのメッセージに "{a}" と "{b}" が含まれる'))
def _error_message_has(ctx: Ctx, a: str, b: str) -> None:
    message = ctx.result["error"]["message"]
    assert a in message and b in message, message


@then(parsers.parse('エラーのメッセージは "{prefix}" で始まる'))
def _error_message_starts(ctx: Ctx, prefix: str) -> None:
    assert ctx.result["error"]["message"].startswith(prefix), ctx.result


@then("エラーのメッセージは 従来どおり である")
def _error_message_plain(ctx: Ctx) -> None:
    assert ctx.result["error"]["message"] == "引数が不正です", ctx.result
    assert "hint" not in ctx.result["error"].get("details", {}), ctx.result


@then("結果に message は含まれない")
def _result_no_message(ctx: Ctx) -> None:
    assert "message" not in ctx.result, ctx.result


def _description(ctx: Ctx, name: str) -> str:
    return next(sp.description for sp in ctx.toolbox.specs if sp.name == name)


@then(parsers.parse('ツール "{name}" の説明に "{a}" と "{b}" が含まれる'))
def _desc_has(ctx: Ctx, name: str, a: str, b: str) -> None:
    d = _description(ctx, name)
    assert a in d and b in d, d


@then(parsers.parse('ツール "{name}" の説明に "{a}" は含まれない'))
def _desc_has_not(ctx: Ctx, name: str, a: str) -> None:
    assert a not in _description(ctx, name)
