"""シミュレーターが扱うエージェントの共通インターフェース。LLMでもルールベースでも差し替えられる。"""

from collections.abc import Callable
from typing import Any, Protocol

from vmgs.agent.prompts import TurnSummary
from vmgs.agent.runner import TurnResult
from vmgs.world.world import World


class TurnAgent(Protocol):
    # シミュレーターが毎ターン直前に、直近の判断要約を代入する
    history: list[TurnSummary]
    # シミュレーターが毎ターン直前に、トレース用の実行情報（run_name / tags / metadata）を代入する
    run_config: dict[str, Any]

    def run_turn(self) -> TurnResult: ...


AgentFactory = Callable[[World], TurnAgent]
