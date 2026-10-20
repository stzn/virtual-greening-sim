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
