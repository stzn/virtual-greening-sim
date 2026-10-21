"""ルールベースのエージェント。LLMが「ルールより賢いか」を測る物差しで、同じ入力には常に同じ行動を返す。"""

from typing import Any

from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.prompts import TurnSummary
from vmgs.agent.runner import TurnResult
from vmgs.tools.tools import ToolBox
from vmgs.world.spots import OTEMACHI_LAT, OTEMACHI_LON
from vmgs.world.world import World

RAIN_PROB_PCT = 60.0
RAIN_MM = 1.0
HOT_TEMP_C = 33.0
HOT_SUN_W_M2 = 700.0
MIN_BATTERY_TO_MOVE = 15.0
DRY_MOISTURE = 30.0
SUN_RETREAT_MOISTURE = 40.0
MAX_IRRIGATION_ML = 1000
MIN_IRRIGATION_ML = 50  # ツールの下限


class NoopAgent:
    """何もしない（放置）。比較の下限。"""

    def __init__(self, world: World) -> None:
        self.history: list[TurnSummary] = []
        self.run_config: dict[str, Any] = {}

    def run_turn(self) -> TurnResult:
        return TurnResult(
            decision=TurnDecision(action_summary="何もしない", rationale="noop"),
            tool_calls=[],
            messages=[],
        )


class BaselineAgent:
    """状態と天気を確認し、チェックリストに従って移動・散水を決める。"""

    def __init__(self, world: World) -> None:
        self.toolbox = ToolBox(world)
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
        if state["status"] == "DEAD" or not weather["ok"]:
            return self._result(calls, "何もしない", "枯死または天気取得失敗のため行動しない")

        cur = weather["current"]
        rain_soon = any(
            (f["precipitation_prob_pct"] or 0.0) >= RAIN_PROB_PCT
            or f["precipitation_mm"] >= RAIN_MM
            for f in weather["forecast"]
        )
        moisture, spot = state["soil_moisture_pct"], state["spot_id"]
        hot_sun = cur["temperature_c"] >= HOT_TEMP_C or cur["shortwave_w_m2"] >= HOT_SUN_W_M2

        if (
            spot in ("A", "C")
            and hot_sun
            and moisture < SUN_RETREAT_MOISTURE
            and state["battery_pct"] >= MIN_BATTERY_TO_MOVE
        ):
            call("move_to_spot", spot_id="B")
            return self._result(calls, "Bへ退避", "日向が猛暑・強日射で水分が少ないため日影へ")

        amount = min(MAX_IRRIGATION_ML, state["water_tank_ml"])
        if moisture < DRY_MOISTURE and not rain_soon and amount >= MIN_IRRIGATION_ML:
            call("irrigate", amount_ml=amount)
            return self._result(calls, f"{amount}ml散水", "乾燥しており直近の雨も無いため")

        return self._result(calls, "現状維持", "対処が必要な条件に当てはまらない")

    @staticmethod
    def _result(calls: list[ToolCallRecord], summary: str, rationale: str) -> TurnResult:
        return TurnResult(
            decision=TurnDecision(action_summary=summary, rationale=rationale),
            tool_calls=calls,
            messages=[],
        )
