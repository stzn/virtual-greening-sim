"""シミュレーションでの散水の切り詰め。"""

from pytest_bdd import given, parsers, scenarios, then

from tests.step_defs.sim_steps import SimCtx
from vmgs.agent.baseline import NoopAgent
from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.runner import TurnResult
from vmgs.tools.tools import ToolBox
from vmgs.world.world import World

scenarios("sim/irrigation_truncation_sim.feature")


@given(parsers.parse("水タンク {tank:d}ml から始めて、毎時 {ml:d}ml 散水するエージェントを使う"))
def _g_irrigator(sc: SimCtx, tank: int, ml: int) -> None:
    class Irrigator(NoopAgent):
        def __init__(self, world: World) -> None:
            super().__init__(world)
            self.toolbox = ToolBox(world)

        def run_turn(self) -> TurnResult:
            result = self.toolbox.call("irrigate", amount_ml=ml)
            return TurnResult(
                decision=TurnDecision(action_summary="散水を試みた"),
                tool_calls=[ToolCallRecord(name="irrigate", args={"amount_ml": ml}, result=result)],
                messages=[],
            )

    sc.overrides["initial_tank_ml"] = tank
    sc.factory = Irrigator


@given("散水の切り詰めを無効にして実行する")
def _g_no_truncate(sc: SimCtx) -> None:
    sc.overrides["truncate_irrigation"] = False


def _irrigate_result(sc: SimCtx, hour: int) -> dict[str, object]:
    call = next(c for c in sc.last.steps[hour - 1].tool_calls if c.name == "irrigate")
    return call.result


@then(parsers.parse("{h:d}時間目の散水は成功し、実際の散水量は {n:d}ml である"))
def _t_ok(sc: SimCtx, h: int, n: int) -> None:
    r = _irrigate_result(sc, h)
    assert r["ok"] is True
    assert r["applied_ml"] == n


@then(parsers.parse('{h:d}時間目の散水はエラー "{code}" で拒否される'))
def _t_error(sc: SimCtx, h: int, code: str) -> None:
    r = _irrigate_result(sc, h)
    assert r["ok"] is False
    assert r["error"]["code"] == code  # type: ignore[index]
