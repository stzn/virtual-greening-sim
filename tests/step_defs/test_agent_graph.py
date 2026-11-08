"""1ターンのグラフ制御。LLMは ScriptedChatModel（固定応答）で置き換える。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import Ctx
from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.graph import AgentState, TurnDecision
from vmgs.agent.prompts import TurnSummary
from vmgs.agent.runner import AgentRunner, TurnResult

scenarios("agent/graph.feature")


class AgentCtx:
    """エージェント用の共有状態（Ctx に重ねて使う）。"""

    def __init__(self) -> None:
        self.model: ScriptedChatModel | None = None
        self.history: list[TurnSummary] = []
        self.result: TurnResult | None = None
        self.raised: Exception | None = None


@pytest.fixture
def actx() -> AgentCtx:
    return AgentCtx()


def _table_rows(datatable: list[list[str]]) -> list[dict[str, str]]:
    header, *rows = datatable
    return [dict(zip(header, r, strict=True)) for r in rows]


def _final(text: str) -> AIMessage:
    return AIMessage(content=text)


# ---------------- Given ----------------
@given(parsers.parse('LLMは何もツールを呼ばず "{text}" と答える'))
def _llm_no_tools(actx: AgentCtx, text: str) -> None:
    actx.model = ScriptedChatModel(responses=[_final(text)])


def _tool_messages(datatable: list[list[str]]) -> list[AIMessage]:
    return [
        tool_call_message(row["ツール"], json.loads(row.get("引数", "{}")), f"call_{i}")
        for i, row in enumerate(_table_rows(datatable))
    ]


def _json_body(summary: str) -> str:
    return json.dumps({"action_summary": summary, "rationale": "理由"}, ensure_ascii=False)


@given("LLMは次の順に応答する")
def _llm_script(actx: AgentCtx, datatable: list[list[str]]) -> None:
    responses: list[AIMessage] = []
    for i, row in enumerate(_table_rows(datatable)):
        if row["応答"] == "ツール呼び出し":
            args = json.loads(row["引数"] or "{}")
            responses.append(tool_call_message(row["ツール"], args, f"call_{i}"))
        elif row["応答"] == "本文のツール呼び出し":
            call = {"name": row["ツール"], "arguments": json.loads(row["引数"] or "{}")}
            responses.append(_final(json.dumps(call, ensure_ascii=False)))
        elif row["応答"] == "本文":
            responses.append(_final(row["要約"]))
        else:
            responses.append(_final(_json_body(row["要約"])))
    actx.model = ScriptedChatModel(responses=responses)


@given(parsers.parse('LLMは次の順にツールを呼んでから "{text}" と答える'))
def _llm_sequence(actx: AgentCtx, datatable: list[list[str]], text: str) -> None:
    actx.model = ScriptedChatModel(responses=[*_tool_messages(datatable), _final(text)])


@given(parsers.parse('LLMは次の順にツールを呼んでから JSON で行動要約 "{text}" と答える'))
def _llm_sequence_json(actx: AgentCtx, datatable: list[list[str]], text: str) -> None:
    actx.model = ScriptedChatModel(responses=[*_tool_messages(datatable), _final(_json_body(text))])


@given(parsers.parse('LLMは何もツールを呼ばず 判断メモのあとにJSONで行動要約 "{text}" と答える'))
def _llm_memo_then_json(actx: AgentCtx, text: str) -> None:
    memo = "#1: 降水確率5%<60% × → 非該当\n#5: 水分55% ∈[30,70] ○ → 何もしない\n\n"
    actx.model = ScriptedChatModel(responses=[_final(memo + _json_body(text))])


@given(parsers.parse("LLMは {tool} を永遠に呼び続ける"))
def _llm_forever(actx: AgentCtx, tool: str) -> None:
    actx.model = ScriptedChatModel(
        responses=[tool_call_message(tool, {}, "call")], repeat_last=True
    )


@given(parsers.parse('LLMは何もツールを呼ばず JSON で行動要約 "{text}" と答える'))
def _llm_json_final(actx: AgentCtx, text: str) -> None:
    body = json.dumps({"action_summary": text, "rationale": "理由"}, ensure_ascii=False)
    actx.model = ScriptedChatModel(responses=[_final(body)])


@given("LLMの呼び出しは例外で失敗する")
def _llm_fail(actx: AgentCtx) -> None:
    actx.model = ScriptedChatModel(responses=[], fail=True)


@given("前のターンの判断要約が次のとおり2件ある")
def _history(actx: AgentCtx, datatable: list[list[str]]) -> None:
    actx.history = [
        TurnSummary(time=r["時刻"], action_summary=r["行動要約"]) for r in _table_rows(datatable)
    ]


# ---------------- When ----------------
@when("エージェントに1ターン判断させる")
def _run_turn(ctx: Ctx, actx: AgentCtx) -> None:
    assert actx.model is not None
    ctx.before = ctx.snapshot()
    runner = AgentRunner(actx.model, ctx.world)
    runner.history = list(actx.history)
    try:
        actx.result = runner.run_turn()
    except Exception as e:  # noqa: BLE001  「例外は発生しない」を検証するため捕捉して保持する
        actx.raised = e


# ---------------- Then ----------------
def _result(actx: AgentCtx) -> TurnResult:
    assert actx.raised is None, f"例外が発生: {actx.raised!r}"
    assert actx.result is not None
    return actx.result


@then(parsers.parse("ツール呼び出しは {n:d} 回である"))
def _count(actx: AgentCtx, n: int) -> None:
    assert len(_result(actx).tool_calls) == n


@then("ツール呼び出しは次の順に記録される")
def _order(actx: AgentCtx, datatable: list[list[str]]) -> None:
    expected = [r["ツール"] for r in _table_rows(datatable)]
    assert [c.name for c in _result(actx).tool_calls] == expected


@then(parsers.parse('判断の行動要約は "{text}" である'))
def _summary(actx: AgentCtx, text: str) -> None:
    assert _result(actx).decision.action_summary == text


@then(parsers.parse("判断に {names} が含まれる"))
def _decision_fields(actx: AgentCtx, names: str) -> None:
    fields = {"行動要約": "action_summary", "判断理由": "rationale", "リスクメモ": "risk_notes"}
    dumped = _result(actx).decision.model_dump()
    for n in names.split("・"):
        assert fields[n.strip()] in dumped
    assert isinstance(_result(actx).decision, TurnDecision)


@then(parsers.parse('判断理由に "{text}" が含まれる'))
def _rationale_has(actx: AgentCtx, text: str) -> None:
    assert text in _result(actx).decision.rationale


@then("判断は再確認を経ている")
def _retried(actx: AgentCtx) -> None:
    assert _result(actx).decision.claim_retry is True


@then("判断は再確認を経ていない")
def _not_retried(actx: AgentCtx) -> None:
    assert _result(actx).decision.claim_retry is False


@then("判断はツール呼び出しの形式の再確認を経ている")
def _text_tool_retried(actx: AgentCtx) -> None:
    assert _result(actx).decision.text_tool_call_retry is True


@then("判断はツール呼び出しの形式の再確認を経ていない")
def _text_tool_not_retried(actx: AgentCtx) -> None:
    assert _result(actx).decision.text_tool_call_retry is False


@then("判断は宣言のみとして記録される")
def _unexecuted(actx: AgentCtx) -> None:
    assert _result(actx).decision.unexecuted_claim is True


@then("判断は宣言のみではない")
def _not_unexecuted(actx: AgentCtx) -> None:
    assert _result(actx).decision.unexecuted_claim is False


@then(parsers.parse('判断が元の要約として "{text}" を保持している'))
def _claimed_kept(actx: AgentCtx, text: str) -> None:
    assert _result(actx).decision.claimed_summary == text


@then(parsers.parse("LLMへの呼び出しは {n:d} 回である"))
def _llm_calls(actx: AgentCtx, n: int) -> None:
    assert actx.model is not None
    assert len(actx.model.inputs) == n


@then(parsers.parse('LLMへの再確認の入力に "{text}" が含まれる'))
def _nudge_has(actx: AgentCtx, text: str) -> None:
    assert actx.model is not None
    nudges = [
        str(m.content)
        for batch in actx.model.inputs
        for m in batch
        if isinstance(m, HumanMessage) and "【確認】" in str(m.content)
    ]
    assert nudges, "再確認のメッセージがLLMに渡されていない"
    assert any(text in n for n in nudges)


@then("判断は上限到達で打ち切られたと記録される")
def _truncated(actx: AgentCtx) -> None:
    assert _result(actx).decision.truncated is True


@then("判断は要約崩れとして記録される")
def _malformed(actx: AgentCtx) -> None:
    assert _result(actx).decision.raw_fallback is True


@then("判断は要約崩れではない")
def _not_malformed(actx: AgentCtx) -> None:
    assert _result(actx).decision.raw_fallback is False


@then("判断は失敗として記録される")
def _failed(actx: AgentCtx) -> None:
    d = _result(actx).decision
    assert d.failed is True
    assert d.error


@then("例外は発生しない")
def _no_exception(actx: AgentCtx) -> None:
    assert actx.raised is None, f"例外が発生: {actx.raised!r}"


@then(parsers.parse('{n:d}回目のツール結果はエラー "{code}" である'))
def _nth_error(actx: AgentCtx, n: int, code: str) -> None:
    r = _result(actx).tool_calls[n - 1].result
    assert r["ok"] is False
    assert r["error"]["code"] == code


@then(parsers.parse("{n:d}回目のツール結果は成功である"))
def _nth_ok(actx: AgentCtx, n: int) -> None:
    assert _result(actx).tool_calls[n - 1].result["ok"] is True


@then("LLMには1回目のエラーが ToolMessage として渡されている")
def _error_passed(actx: AgentCtx) -> None:
    assert actx.model is not None
    second_input = actx.model.inputs[1]
    tool_msgs = [m for m in second_input if isinstance(m, ToolMessage)]
    assert tool_msgs
    assert json.loads(str(tool_msgs[0].content))["error"]["code"] == "INSUFFICIENT_WATER"


@then("ユニットの状態は変化しない")
def _unchanged(ctx: Ctx) -> None:
    assert ctx.snapshot() == ctx.before


def _first_input_text(actx: AgentCtx) -> str:
    assert actx.model is not None
    return "\n".join(str(m.content) for m in actx.model.inputs[0])


@then(parsers.parse('LLMへの最初の入力のシステムプロンプトに "{text}" が含まれる'))
def _system_prompt(actx: AgentCtx, text: str) -> None:
    assert actx.model is not None
    first = actx.model.inputs[0][0]
    assert isinstance(first, SystemMessage)
    assert text in str(first.content)


@then(parsers.parse('LLMへの入力に仮想時刻 "{hhmm}" が含まれる'))
def _has_time(actx: AgentCtx, hhmm: str) -> None:
    assert hhmm in _first_input_text(actx)


@then(parsers.parse('LLMへの入力に現在のスポット "{spot}" が含まれる'))
def _has_spot(actx: AgentCtx, spot: str) -> None:
    assert f"現在のスポット: {spot}" in _first_input_text(actx)


@then(parsers.parse('LLMへの入力に "{text}" が含まれる'))
def _has_text(actx: AgentCtx, text: str) -> None:
    assert text in _first_input_text(actx)


@then("前のターンの会話メッセージはそのままは引き継がれない")
def _no_carryover(actx: AgentCtx) -> None:
    assert actx.model is not None
    first = actx.model.inputs[0]
    assert [type(m) for m in first] == [SystemMessage, HumanMessage]


@then(parsers.parse("仮想時刻は {hhmm} のままである"))
def _time_unchanged(ctx: Ctx, hhmm: str) -> None:
    assert ctx.world.clock.now.strftime("%H:%M") == hhmm


def test_agent_state_keys() -> None:
    """AgentState のキーが想定どおりであること（グラフ入出力の契約）。"""
    keys: set[str] = set(AgentState.__annotations__)
    assert keys == {
        "messages",
        "tool_call_count",
        "tool_log",
        "truncated",
        "nudged",
        "retry_requested",
        "text_tool_nudged",
        "decision",
    }
