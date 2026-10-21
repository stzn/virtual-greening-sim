"""1ターン（=仮想時間の1時間）の判断を行うLangGraph。

START → agent ⇄ tools → finalize → END
  - agent: LLM呼び出し。失敗したら判断を「失敗」として記録して終了（例外は外に出さない）
  - tools: tool_calls を ToolBox で実行し、結果（エラー含む）を ToolMessage で返す
  - finalize: 最終回答を TurnDecision に変換
ツール呼び出しが上限に達したら打ち切って finalize へ進む。
"""

import json
import logging
import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, model_validator

from vmgs.agent.claims import claimed_actions
from vmgs.tools.tools import ToolBox

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_TURN = 8


_DECODER = json.JSONDecoder()


def _extract_decision_json(text: str) -> dict[str, Any] | None:
    """本文中のJSONオブジェクトのうち、action_summary を持つ最後のものを返す。

    最終回答は「判断メモのあとにJSONを1つ」の形を許す（行動ツールを呼ばないターンでは、
    メモが最後のメッセージになるため）。コードブロックで囲まれていても拾える。
    """
    found: dict[str, Any] | None = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = _DECODER.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action_summary" in obj:
            found = obj
    return found


class TurnDecision(BaseModel):
    action_summary: str = ""
    rationale: str = ""
    risk_notes: str = ""
    truncated: bool = False  # ツール呼び出し上限で打ち切られた
    failed: bool = False  # LLM呼び出しが失敗した
    error: str | None = None
    raw_fallback: bool = False  # 最終回答がJSONにならず、本文をそのまま要約にした（要約崩れ）
    claim_retry: bool = False  # 要約が述べた行動が未実行だったため、再確認を促した
    unexecuted_claim: bool = False  # 再確認後も未実行で、要約を実際の行動で置き換えた（宣言のみ）
    claimed_summary: str | None = None  # 置き換える前の、モデルが述べた要約

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_raw_fallback(cls, data: Any) -> Any:
        """旧ログ互換: raw_fallback の記録が無く、要約と判断理由が同一なら本文のフォールバックとみなす。

        旧パーサーは「全体がJSON」でないと本文をそのまま要約と判断理由に入れていた。
        本文に（判断メモのあとの）JSONが含まれていれば取り出し直し、無ければ要約崩れとする。
        """
        if isinstance(data, dict) and "raw_fallback" not in data:
            summary = data.get("action_summary")
            if summary and summary == data.get("rationale"):
                obj = _extract_decision_json(summary)
                if obj is not None:
                    return {
                        **data,
                        "raw_fallback": False,
                        "action_summary": str(obj.get("action_summary", "")),
                        "rationale": str(obj.get("rationale", "")),
                        "risk_notes": str(obj.get("risk_notes", "")),
                    }
                return {**data, "raw_fallback": True}
        return data


class ToolCallRecord(BaseModel):
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_call_count: int
    tool_log: Annotated[list[ToolCallRecord], operator.add]
    truncated: bool
    nudged: bool  # 宣言と実行の不一致で、再確認を促し済み（1ターンに1回まで）
    retry_requested: bool  # 直前の verify が再確認を促した
    decision: TurnDecision | None


def message_text(msg: AnyMessage | BaseMessage) -> str:
    """content が文字列でもブロック列でもテキストを取り出す。"""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = [b if isinstance(b, str) else str(b.get("text", "")) for b in c]
    return "".join(parts)


def parse_decision(text: str) -> TurnDecision:
    """最終回答から判断を取り出す。JSONが無ければ本文を判断理由とし、raw_fallback を立てる。"""
    obj = _extract_decision_json(text)
    if obj is None:
        return TurnDecision(action_summary=text.strip(), rationale=text.strip(), raw_fallback=True)
    return TurnDecision(
        raw_fallback=False,
        action_summary=str(obj.get("action_summary", "")),
        rationale=str(obj.get("rationale", "")),
        risk_notes=str(obj.get("risk_notes", "")),
    )


def summarize_executed(calls: list["ToolCallRecord"]) -> str:
    """実際に成功した行動（移動・散水）から要約を作る。JSONが無いときに、本文の代わりに使う。"""
    parts: list[str] = []
    for c in calls:
        if not c.result.get("ok"):
            continue
        if c.name == "move_to_spot":
            parts.append(f"{c.result['new_spot_id']}へ移動")
        elif c.name == "irrigate":
            parts.append(f"{c.args['amount_ml']}ml散水")
    return "、".join(parts) if parts else "何もしない"


def executed_actions(calls: list[ToolCallRecord]) -> set[str]:
    """成功した行動ツール（移動・散水）の名前。"""
    return {c.name for c in calls if c.result.get("ok") and c.name in ("move_to_spot", "irrigate")}


def nudge_text(claims: set[str], calls: list[ToolCallRecord]) -> str:
    """宣言と実行の不一致を伝え、実行か撤回かを選ばせる再確認の文面。"""
    names = "・".join(sorted(claims))
    text = (
        f"【確認】最終回答の行動要約は {names} を実行したと述べていますが、"
        f"この1時間に成功した {names} の呼び出しがありません。"
    )
    failed = [
        f"{c.name}: {c.result['error']['code']}"
        for c in calls
        if c.name in claims and not c.result.get("ok")
    ]
    if failed:
        text += f"失敗した呼び出し: {', '.join(failed)}。"
    return (
        text + "実行する場合は、いま対応するツールを呼び出してください。"
        "実行しない場合は、行動要約を実際の行動（何もしない等）に直して、最終回答のJSONを返し直してください。"
    )


def langchain_tools(toolbox: ToolBox) -> list[StructuredTool]:
    """LLMにスキーマを見せるためだけのツール定義。実行は ToolBox.call が行う。"""

    def _noop(**_: Any) -> str:
        raise RuntimeError("スキーマ用であり直接実行しない")

    return [
        StructuredTool.from_function(
            func=_noop, name=s.name, description=s.description, args_schema=s.args_model
        )
        for s in toolbox.specs
    ]


def build_turn_graph(
    model: BaseChatModel,
    toolbox: ToolBox,
    *,
    max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
) -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    bound = model.bind_tools(langchain_tools(toolbox))

    def agent_node(state: AgentState) -> dict[str, Any]:
        try:
            ai = bound.invoke(state["messages"])
        except Exception as e:  # LLM/ネットワーク障害でターンを落とさず、失敗として記録する
            logger.exception("LLM呼び出しに失敗しました")
            return {
                "decision": TurnDecision(
                    action_summary="判断失敗（LLM呼び出しエラー）",
                    failed=True,
                    error=f"{type(e).__name__}: {e}",
                )
            }
        return {"messages": [ai]}

    def tools_node(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        count = state["tool_call_count"]
        out: list[ToolMessage] = []
        log: list[ToolCallRecord] = []
        for tc in last.tool_calls:
            if count >= max_tool_calls:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": {
                        "code": "TOOL_CALL_LIMIT",
                        "message": f"1ターンのツール呼び出し上限（{max_tool_calls}回）に達しました",
                        "details": {},
                    },
                }
            else:
                count += 1
                result = toolbox.call(tc["name"], **tc["args"])
                log.append(ToolCallRecord(name=tc["name"], args=tc["args"], result=result))
            out.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
        return {
            "messages": out,
            "tool_call_count": count,
            "tool_log": log,
            "truncated": count >= max_tool_calls,
        }

    def verify_node(state: AgentState) -> dict[str, Any]:
        """最終回答が「実行した」と述べた行動に、成功したツール呼び出しが無ければ、1回だけ再確認を促す。"""
        ais = [m for m in state["messages"] if isinstance(m, AIMessage)]
        decision = parse_decision(message_text(ais[-1]) if ais else "")
        if decision.raw_fallback or state["nudged"]:
            return {"retry_requested": False}  # 本文そのままの回答は宣言の判定が当てにならない
        mismatch = claimed_actions(decision.action_summary) - executed_actions(state["tool_log"])
        if not mismatch:
            return {"retry_requested": False}
        nudge = HumanMessage(nudge_text(mismatch, state["tool_log"]))
        return {"messages": [nudge], "nudged": True, "retry_requested": True}

    def finalize_node(state: AgentState) -> dict[str, Any]:
        ais = [m for m in state["messages"] if isinstance(m, AIMessage)]
        text = message_text(ais[-1]) if ais else ""
        decision = parse_decision(text)
        decision.claim_retry = state["nudged"]
        if decision.raw_fallback:
            # 次ターンの記憶になる要約に本文（判断メモ）を入れない。実行した行動から作る
            decision.action_summary = summarize_executed(state["tool_log"])
        elif not state["truncated"] and (
            claimed_actions(decision.action_summary) - executed_actions(state["tool_log"])
        ):
            # 再確認後も未実行。偽の宣言を次ターンの記憶に残さず、実際の行動で置き換える
            decision.claimed_summary = decision.action_summary
            decision.action_summary = summarize_executed(state["tool_log"])
            decision.unexecuted_claim = True
        if state["truncated"]:
            decision.truncated = True
            if decision.action_summary == "何もしない":
                decision.action_summary = "ツール呼び出し上限に達したため打ち切り"
        return {"decision": decision}

    def after_agent(state: AgentState) -> Literal["tools", "verify", "__end__"]:
        if state["decision"] is not None:
            return "__end__"
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        return "tools" if last.tool_calls else "verify"

    def after_verify(state: AgentState) -> Literal["agent", "finalize"]:
        return "agent" if state["retry_requested"] else "finalize"

    def after_tools(state: AgentState) -> Literal["agent", "finalize"]:
        return "finalize" if state["truncated"] else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("verify", verify_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_conditional_edges("verify", after_verify)
    g.add_conditional_edges("tools", after_tools)
    g.add_edge("finalize", END)
    return g.compile()
