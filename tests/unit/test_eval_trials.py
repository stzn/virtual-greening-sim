import pytest
from langchain_core.messages import AIMessage

from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.runner import TurnResult
from vmgs.eval.trials import (
    ForecastPoint,
    TrialSetup,
    build_weather,
    build_world,
    checked_state_before_acting,
    irrigation_within_tank,
    moved_to,
    not_called,
    run_trials,
    success_count,
    vary_setup,
)


def _result(*calls: tuple[str, dict[str, object], bool]) -> TurnResult:
    return TurnResult(
        decision=TurnDecision(),
        tool_calls=[ToolCallRecord(name=n, args=a, result={"ok": ok}) for n, a, ok in calls],
        messages=[],
    )


def test_予報の降水が指定した時間後に反映される() -> None:
    setup = TrialSetup(forecast=[ForecastPoint(1, 80, 6.0)])
    w = build_weather(setup)
    now = build_world(setup).clock.now
    assert w.at(now).precipitation_mm == 0.0
    nxt = w.forecast(now, 2)
    assert (nxt[0].precipitation_prob_pct, nxt[0].precipitation_mm) == (80, 6.0)
    assert nxt[1].precipitation_prob_pct == 5.0


def test_build_world_は初期状態を反映する() -> None:
    w = build_world(TrialSetup(spot="B", moisture=40, tank_ml=120, battery_pct=8))
    assert (w.plant.spot_id, w.plant.moisture, w.plant.water_tank_ml, w.plant.battery_pct) == (
        "B",
        40,
        120,
        8,
    )


def test_述語() -> None:
    r = _result(("get_plant_state", {}, True), ("irrigate", {"amount_ml": 100}, True))
    assert checked_state_before_acting(r)
    assert not_called("move_to_spot")(r)
    assert not moved_to("B")(r)
    assert irrigation_within_tank(120)(r)
    assert not irrigation_within_tank(50)(r)


def test_状態確認より前に行動したら不合格() -> None:
    r = _result(("irrigate", {"amount_ml": 100}, True), ("get_plant_state", {}, True))
    assert not checked_state_before_acting(r)
    assert not checked_state_before_acting(_result())


def test_失敗した移動は移動したとみなさない() -> None:
    r = _result(("move_to_spot", {"spot_id": "A"}, False))
    assert not moved_to("A")(r)


def test_run_trials_は試行ごとに独立し成功回数を数える() -> None:
    model = ScriptedChatModel(
        responses=[tool_call_message("get_plant_state", {}, "c"), AIMessage(content="維持")],
        repeat_last=False,
    )
    # 1試行 = LLM呼び出し2回（状態確認→最終回答）。応答もちょうど2回分
    results = run_trials(model, TrialSetup(), n=1)
    assert len(results) == 1
    assert success_count(results, [checked_state_before_acting]) == 1


def test_LLM失敗の試行は成功に数えない() -> None:
    failed = TurnResult(decision=TurnDecision(failed=True, error="x"), tool_calls=[], messages=[])
    assert success_count([failed], [not_called("irrigate")]) == 0


def test_build_world_は直前の移動を反映し_試みない述語は失敗した試行も検出する() -> None:
    from vmgs.eval.trials import not_attempted_move_to

    w = build_world(TrialSetup(spot="B", last_move=("A", "B", 1)))
    last = w.state().last_move
    assert last is not None
    assert (last.from_spot, last.hours_ago) == ("A", 1)
    refused = _result(("move_to_spot", {"spot_id": "A"}, False))
    assert not moved_to("A")(refused)  # 実際には移動していない
    assert not not_attempted_move_to("A")(refused)  # だが試みた


def test_build_world_は移動予算の消費を反映する() -> None:
    w = build_world(TrialSetup(spot="A", recent_moves_hours_ago=[3, 7]))
    st = w.state()
    assert (st.moves_in_window, st.move_budget_remaining) == (2, 0)


# ---- 試行ごとの条件の揺らぎ（temperature=0 で5回が同一になるのを避ける） ----
_BASE = TrialSetup(spot="C", moisture=35, tank_ml=8000, battery_pct=70, temperature_c=34)


def test_揺らぎの試行0は条件そのまま() -> None:
    assert vary_setup(_BASE, seed=0, index=0) == _BASE


def test_揺らぎは試行ごとに変わり同じシードと番号なら同一() -> None:
    a, b = vary_setup(_BASE, 0, 1), vary_setup(_BASE, 0, 2)
    assert a != _BASE and a != b
    assert vary_setup(_BASE, 0, 1) == a
    assert vary_setup(_BASE, 1, 1) != a


def test_揺らぎはスポットとタンクを変えない() -> None:
    for i in range(1, 30):
        v = vary_setup(_BASE, 0, i)
        assert (v.tank_ml, v.spot) == (_BASE.tank_ml, _BASE.spot)


def test_揺らぎは水分_気温_日射_風_湿度_バッテリーを変える() -> None:
    vs = [vary_setup(_BASE, 0, i) for i in range(1, 30)]
    for field in (
        "moisture",
        "temperature_c",
        "shortwave_w_m2",
        "wind_ms",
        "humidity_pct",
        "battery_pct",
    ):
        assert len({getattr(v, field) for v in vs}) > 1, field
    assert all(abs(v.moisture - _BASE.moisture) <= 5 for v in vs)
    assert all(abs(v.temperature_c - _BASE.temperature_c) <= 2 for v in vs)
    assert all(
        abs(v.shortwave_w_m2 - _BASE.shortwave_w_m2) <= _BASE.shortwave_w_m2 * 0.1 for v in vs
    )
    assert all(abs(v.wind_ms - _BASE.wind_ms) <= 1.5 and v.wind_ms >= 0 for v in vs)
    assert all(abs(v.humidity_pct - _BASE.humidity_pct) <= 15 for v in vs)
    assert all(abs(v.battery_pct - _BASE.battery_pct) <= 8 for v in vs)


def _side(value: float, thresholds: tuple[float, ...]) -> tuple[bool, ...]:
    return tuple(value >= t for t in thresholds)


def test_揺らぎは判断ルールの閾値をまたがない() -> None:
    """水分 30/40/50/70/85・気温 33・日射 700 は、元の値と同じ側にとどまる（境界上の値も）。"""
    for m in (15.0, 25.0, 29.5, 30.0, 35.0, 40.0, 50.0, 55.0, 70.0, 84.0, 90.0):
        for t in (26.0, 32.5, 33.0, 36.0):
            for sw in (200.0, 699.0, 700.0, 900.0):
                base = TrialSetup(moisture=m, temperature_c=t, shortwave_w_m2=sw)
                for i in range(1, 25):
                    v = vary_setup(base, 0, i)
                    assert _side(v.moisture, (30, 40, 50, 70, 85)) == _side(m, (30, 40, 50, 70, 85))
                    assert _side(v.temperature_c, (33,)) == _side(t, (33,))
                    assert _side(v.shortwave_w_m2, (700,)) == _side(sw, (700,))
                    assert 0 <= v.moisture <= 100


def test_バッテリーは15パーセントの閾値の同じ側にとどまり100を超えない() -> None:
    for base, low in ((8.0, True), (14.0, True), (15.0, False), (16.0, False), (100.0, False)):
        for i in range(1, 60):
            b = vary_setup(TrialSetup(battery_pct=base), 0, i).battery_pct
            assert b <= 100
            assert (b < 15) is low


def test_run_trials_は試行ごとに揺らした条件で世界を作る(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[float] = []
    real = build_world

    def spy(setup: TrialSetup):  # type: ignore[no-untyped-def]
        seen.append(setup.battery_pct)
        return real(setup)

    monkeypatch.setattr("vmgs.eval.trials.build_world", spy)
    model = ScriptedChatModel(responses=[AIMessage(content="維持")])  # 最後の応答を繰り返す
    run_trials(model, TrialSetup(battery_pct=70), n=4)
    assert seen[0] == 70 and len(set(seen)) > 1


def test_揺らぎは湿度と水分と風速を範囲内に収める() -> None:
    for i in range(1, 60):
        v = vary_setup(TrialSetup(humidity_pct=98, moisture=99, wind_ms=0.2, battery_pct=2), 0, i)
        assert 0 <= v.humidity_pct <= 100
        assert 0 <= v.moisture <= 100
        assert v.wind_ms >= 0
        assert 0 <= v.battery_pct < 15
