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
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, model_validator

from vmgs.tools.tools import ToolBox

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS_PER_TURN = 8


class TurnDecision(BaseModel):
    action_summary: str = ""
    rationale: str = ""
    risk_notes: str = ""
    truncated: bool = False  # ツール呼び出し上限で打ち切られた
    failed: bool = False  # LLM呼び出しが失敗した
    error: str | None = None
    raw_fallback: bool = False  # 最終回答がJSONにならず、本文をそのまま要約にした（要約崩れ）

    @model_validator(mode="before")
    @classmethod
    def _infer_legacy_raw_fallback(cls, data: Any) -> Any:
        """旧ログ互換: raw_fallback の記録が無く、要約と判断理由が同一なら要約崩れとみなす。

        parse_decision のフォールバックは要約と判断理由に同じ本文を入れていたため。
        """
        if isinstance(data, dict) and "raw_fallback" not in data:
            summary = data.get("action_summary")
            if summary and summary == data.get("rationale"):
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
    decision: TurnDecision | None


def message_text(msg: AnyMessage) -> str:
    """content が文字列でもブロック列でもテキストを取り出す。"""
    c = msg.content
    if isinstance(c, str):
        return c
    parts = [b if isinstance(b, str) else str(b.get("text", "")) for b in c]
    return "".join(parts)


def parse_decision(text: str) -> TurnDecision:
    """最終回答のJSONを解釈する。JSONでなければ本文を行動要約として扱う。"""
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return TurnDecision(action_summary=text.strip(), rationale=text.strip(), raw_fallback=True)
    if not isinstance(data, dict):
        return TurnDecision(action_summary=text.strip(), rationale=text.strip(), raw_fallback=True)
    return TurnDecision(
        raw_fallback=False,
        action_summary=str(data.get("action_summary", "")),
        rationale=str(data.get("rationale", "")),
        risk_notes=str(data.get("risk_notes", "")),
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

    def finalize_node(state: AgentState) -> dict[str, Any]:
        ais = [m for m in state["messages"] if isinstance(m, AIMessage)]
        text = message_text(ais[-1]) if ais else ""
        decision = parse_decision(text)
        if state["truncated"]:
            decision.truncated = True
            if not decision.action_summary:
                decision.action_summary = "ツール呼び出し上限に達したため打ち切り"
        return {"decision": decision}

    def after_agent(state: AgentState) -> Literal["tools", "finalize", "__end__"]:
        if state["decision"] is not None:
            return "__end__"
        last = state["messages"][-1]
        assert isinstance(last, AIMessage)
        return "tools" if last.tool_calls else "finalize"

    def after_tools(state: AgentState) -> Literal["agent", "finalize"]:
        return "finalize" if state["truncated"] else "agent"

    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("tools", tools_node)
    g.add_node("finalize", finalize_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent)
    g.add_conditional_edges("tools", after_tools)
    g.add_edge("finalize", END)
    return g.compile()
