"""1ターン分の入力を組み立ててグラフを実行し、判断要約を次ターンへ引き継ぐ。"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from vmgs.agent.graph import (
    MAX_TOOL_CALLS_PER_TURN,
    AgentState,
    ToolCallRecord,
    TurnDecision,
    build_turn_graph,
)
from vmgs.agent.prompts import SYSTEM_PROMPT_V1, TurnSummary, build_turn_input
from vmgs.tools.tools import ToolBox
from vmgs.world.world import World


class TurnResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    decision: TurnDecision
    tool_calls: list[ToolCallRecord]
    messages: list[BaseMessage]


class AgentRunner:
    """会話はターンごとにリセットし、直近の判断要約だけを持ち越す（トークン抑制）。"""

    def __init__(
        self,
        model: BaseChatModel,
        world: World,
        *,
        system_prompt: str = SYSTEM_PROMPT_V1,
        max_tool_calls: int = MAX_TOOL_CALLS_PER_TURN,
        history_turns: int = 3,
    ) -> None:
        self.world = world
        self.system_prompt = system_prompt
        self.max_tool_calls = max_tool_calls
        self.history_turns = history_turns
        self.history: list[TurnSummary] = []
        self._graph = build_turn_graph(model, ToolBox(world), max_tool_calls=max_tool_calls)

    def run_turn(self) -> TurnResult:
        now = self.world.clock.now
        state = self.world.state()
        human = build_turn_input(
            now.strftime("%H:%M"),
            state.spot_id,
            state.spot_name,
            self.history[-self.history_turns :],
        )
        init: AgentState = {
            "messages": [SystemMessage(self.system_prompt), HumanMessage(human)],
            "tool_call_count": 0,
            "tool_log": [],
            "truncated": False,
            "decision": None,
        }
        final = self._graph.invoke(init, config={"recursion_limit": self.max_tool_calls * 2 + 10})
        decision = final["decision"]
        assert decision is not None
        self.history.append(
            TurnSummary(time=now.strftime("%H:%M"), action_summary=decision.action_summary)
        )
        return TurnResult(
            decision=decision, tool_calls=final["tool_log"], messages=final["messages"]
        )
