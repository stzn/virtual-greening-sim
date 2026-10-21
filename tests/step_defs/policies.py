"""ストレス状況の比較に使う、単純な方針のテスト用エージェント。

ベースラインを上回れる方針が存在する（＝余裕が本物である）ことと、わな（水を足し続けると悪化する）の確認に使う。
"""

from typing import Any

from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.prompts import TurnSummary
from vmgs.agent.protocol import AgentFactory
from vmgs.agent.runner import TurnResult
from vmgs.tools.tools import ToolBox
from vmgs.world.spots import OTEMACHI_LAT, OTEMACHI_LON
from vmgs.world.world import World

RAIN_PROB_PCT = 60.0
RAIN_MM = 1.0


class RuleAgent:
    """moisture が dry 未満なら amount ml 散水。rain_ok=False なら雨予報のときは散水しない。
    shade_hour 以降は、日向にいれば日影の B へ退避する。
    """

    def __init__(
        self,
        world: World,
        *,
        dry: float = 30.0,
        amount: int = 1000,
        rain_ok: bool = False,
        shade_hour: int | None = None,
    ) -> None:
        self.world = world
        self.toolbox = ToolBox(world)
        self.dry, self.amount, self.rain_ok, self.shade_hour = dry, amount, rain_ok, shade_hour
        self.history: list[TurnSummary] = []
        self.run_config: dict[str, Any] = {}

    def run_turn(self) -> TurnResult:
        calls: list[ToolCallRecord] = []

        def call(name: str, **kwargs: Any) -> dict[str, Any]:
            result = self.toolbox.call(name, **kwargs)
            calls.append(ToolCallRecord(name=name, args=kwargs, result=result))
            return result

        state = call("get_plant_state")
        weather = call("get_weather", lat=OTEMACHI_LAT, lon=OTEMACHI_LON, hours_ahead=3)
        rain_soon = any(
            (f["precipitation_prob_pct"] or 0.0) >= RAIN_PROB_PCT
            or f["precipitation_mm"] >= RAIN_MM
            for f in weather["forecast"]
        )
        hour = self.world.clock.now.hour
        if (
            self.shade_hour is not None
            and hour >= self.shade_hour
            and state["spot_id"] != "B"
            and state["battery_pct"] >= 15
        ):
            call("move_to_spot", spot_id="B")
        elif state["soil_moisture_pct"] < self.dry and (self.rain_ok or not rain_soon):
            amount = min(self.amount, state["water_tank_ml"])
            if amount >= 50:
                call("irrigate", amount_ml=amount)
        return TurnResult(
            decision=TurnDecision(action_summary="方針に従う"), tool_calls=calls, messages=[]
        )


def _rule(**kwargs: Any) -> AgentFactory:
    return lambda world: RuleAgent(world, **kwargs)


POLICIES: dict[str, AgentFactory] = {
    # 雨の予報があっても、乾燥していれば散水する（ベースラインの「雨前は散水しない」規則を外す）
    "rain_ignoring": _rule(rain_ok=True),
    # 7時以降は日向なら日影へ退避し、水分が30%未満なら500mlずつ散水する（節水）
    "early_shade": _rule(shade_hour=7, amount=500),
    # 水分が90%未満なら、雨でも常に500ml散水する（水を足し続けるわな）
    "always_irrigate": _rule(dry=90.0, amount=500, rain_ok=True),
}
