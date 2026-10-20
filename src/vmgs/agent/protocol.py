"""シミュレーターが扱うエージェントの共通インターフェース。LLMでもルールベースでも差し替えられる。"""

from collections.abc import Callable
from typing import Protocol

from vmgs.agent.prompts import TurnSummary
from vmgs.agent.runner import TurnResult
from vmgs.world.world import World


class TurnAgent(Protocol):
    # シミュレーターが毎ターン直前に、直近の判断要約を代入する
    history: list[TurnSummary]

    def run_turn(self) -> TurnResult: ...


AgentFactory = Callable[[World], TurnAgent]
