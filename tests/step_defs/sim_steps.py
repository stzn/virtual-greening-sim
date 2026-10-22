"""24時間シミュレーション・チェックポイント・指標の共有ステップ定義（conftest から取り込む）。"""

import itertools
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from pytest_bdd import given, parsers, then, when

from vmgs.agent.baseline import BaselineAgent, NoopAgent
from vmgs.agent.fake import ScriptedChatModel
from vmgs.agent.graph import ToolCallRecord, TurnDecision
from vmgs.agent.protocol import AgentFactory, TurnAgent
from vmgs.agent.runner import AgentRunner, TurnResult
from vmgs.eval.metrics import (
    Metrics,
    MoveEvent,
    compute_metrics,
    count_irrigation_before_rain,
    count_wasted_moves,
    metrics_from_run,
)
from vmgs.eval.report import ComparisonReport, compare_agents
from vmgs.sim.records import StepRecord
from vmgs.sim.simulate import (
    SimConfig,
    SimulationError,
    SimulationResult,
    WallClock,
    resume_simulation,
    run_simulation,
)
from vmgs.sim.store import RunStore
from vmgs.tools.tools import ToolBox
from vmgs.weather.cache import load_records
from vmgs.weather.providers import ConstantWeather, SeriesWeather
from vmgs.world.world import World

DATA = Path(__file__).resolve().parents[2] / "data" / "weather"

METRIC_KEYS = {
    "最終健康度": "final_health",
    "最低健康度": "min_health",
    "暑熱緩和スコア": "heat_mitigation_score",
    "移動回数": "move_count",
    "散水総量": "irrigation_total_ml",
    "最終バッテリー": "final_battery",
    "最終タンク残量": "final_tank_ml",
    "拒否された移動": "rejected_moves",
    "ガードによる拒否": "guard_rejections",
    "要約崩れ": "malformed_summaries",
    "宣言のみ": "unexecuted_claims",
    "再確認": "claim_retries",
    "切り詰められた散水": "truncated_irrigations",
}


def _empty_result(decision: TurnDecision) -> TurnResult:
    return TurnResult(decision=decision, tool_calls=[], messages=[])


class SimCtx:
    def __init__(self, tmp_path: Path) -> None:
        self.runs_dir = tmp_path / "runs"
        self.scenario = "normal"
        self.weather: SeriesWeather | None = None
        self.overrides: dict[str, Any] = {}
        self.factory: AgentFactory = BaselineAgent
        self.hours = 24
        self.results: list[SimulationResult] = []
        self.error: Exception | None = None
        self.seen: dict[int, list[str]] = {}  # ステップ番号 → そのターンの入力に入っていた要約
        self.call_times: list[datetime] = []
        self.moves: list[MoveEvent] = []
        self.recomputed: Metrics | None = None
        self.irrigation_times: list[datetime] = []
        self.constant_weather = ConstantWeather(precipitation_prob_pct=80.0)
        self.counted = 0
        self.metrics: Metrics | None = None
        self.report: ComparisonReport | None = None
        self.synthetic: list[StepRecord] = []
        self.clock: WallClock = time.perf_counter
        self.agent_info: dict[str, str] = {}
        self.run_configs: list[dict[str, Any]] = []

    def require_weather(self) -> SeriesWeather:
        assert self.weather is not None
        return self.weather

    def config(self, hours: int) -> SimConfig:
        self.hours = hours
        return SimConfig(scenario=self.scenario, hours=hours, **self.overrides)

    def run(self, hours: int, *, stop_after: int | None = None) -> SimulationResult:
        run_id = f"run{len(self.results) + 1}"
        self.error = None
        try:
            res = run_simulation(
                self.config(hours),
                self.require_weather(),
                self.factory,
                self.runs_dir,
                run_id,
                agent_info=self.agent_info or None,
                stop_after=stop_after,
                clock=self.clock,
            )
        except Exception as e:  # 「例外は発生しない」を検証するため保持する
            self.error = e
            raise
        self.results.append(res)
        return res

    def resume(self, run_id: str) -> SimulationResult:
        return resume_simulation(
            run_id, self.runs_dir, self.require_weather(), self.factory, clock=self.clock
        )

    @property
    def last(self) -> SimulationResult:
        return self.results[-1]


@pytest.fixture
def sc(tmp_path: Path) -> SimCtx:
    return SimCtx(tmp_path)


# ---------------- テスト用エージェント ----------------
def _failing_at(step_hour: int) -> AgentFactory:
    class Failing(NoopAgent):
        def __init__(self, world: World) -> None:
            super().__init__(world)
            self.world = world

        def run_turn(self) -> TurnResult:
            if self.world.clock.now.hour + 1 == step_hour:
                raise RuntimeError("scripted agent failure")
            return super().run_turn()

    return Failing


def _pingpong(world_holder: list[World]) -> AgentFactory:
    """毎時、いまいない方（A か B）へ移動しようとする。往復・予算ガードの回帰用。"""

    class PingPong(NoopAgent):
        def __init__(self, world: World) -> None:
            super().__init__(world)
            self.world = world
            self.toolbox = ToolBox(world)
            world_holder.append(world)

        def run_turn(self) -> TurnResult:
            target = "B" if self.world.plant.spot_id != "B" else "A"
            result = self.toolbox.call("move_to_spot", spot_id=target)
            return TurnResult(
                decision=TurnDecision(action_summary=f"{target}へ移動を試みた"),
                tool_calls=[
                    ToolCallRecord(name="move_to_spot", args={"spot_id": target}, result=result)
                ],
                messages=[],
            )

    return PingPong


class StepClock:
    """1ステップの判断に seconds 秒かかったように見せる時計（ランナーは1ステップで2回だけ呼ぶ）。"""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.n = 0

    def __call__(self) -> float:
        v = (self.n // 2) * 1000.0 + (self.n % 2) * self.seconds
        self.n += 1
        return v


def _run_config_recording(store: list[dict[str, Any]]) -> AgentFactory:
    class Recorder(NoopAgent):
        def run_turn(self) -> TurnResult:
            store.append(json.loads(json.dumps(self.run_config)))  # その時点の値を保存
            return super().run_turn()

    return Recorder


def _numbered_llm() -> AgentFactory:
    """毎ターン『要約k』(k=時間目)をJSONで返すLLMエージェント。再開しても番号が続く。"""

    def make(world: World) -> TurnAgent:
        start = world.clock.now.hour
        responses = [
            AIMessage(
                content=json.dumps(
                    {"action_summary": f"要約{k}", "rationale": "r"}, ensure_ascii=False
                )
            )
            for k in range(start + 1, 25)
        ]
        return AgentRunner(ScriptedChatModel(responses=responses), world)

    return make


def _time_recording(times: list[datetime]) -> AgentFactory:
    class Recording(NoopAgent):
        def __init__(self, world: World) -> None:
            super().__init__(world)
            self.world = world

        def run_turn(self) -> TurnResult:
            times.append(self.world.clock.now)
            return super().run_turn()

    return Recording


def _unique_summary(seen: dict[int, list[str]]) -> AgentFactory:
    class Unique(NoopAgent):
        def __init__(self, world: World) -> None:
            super().__init__(world)
            self.world = world

        def run_turn(self) -> TurnResult:
            k = (
                self.world.clock.now.hour + 1
            )  # 開始が0時なので = ステップ番号（再開しても保たれる）
            seen[k] = [h.action_summary for h in self.history]
            return _empty_result(TurnDecision(action_summary=f"要約{k}"))

    return Unique


# ---------------- Given ----------------
@given(parsers.parse('シナリオ "{name}" の気象データがある'))
def _g_weather(sc: SimCtx, name: str) -> None:
    sc.scenario = name
    sc.weather = SeriesWeather(load_records(DATA / f"{name}.json"))


@given(parsers.parse('シナリオ "{name}" の気象データが {n:d}時間分しかない'))
def _g_short_weather(sc: SimCtx, name: str, n: int) -> None:
    sc.scenario = name
    sc.weather = SeriesWeather(load_records(DATA / f"{name}.json")[:n])


@given("何もしないエージェントを使う")
def _g_noop(sc: SimCtx) -> None:
    sc.factory = NoopAgent


@given(parsers.parse("初期の土壌水分は {m:g}% で健康度は {h:g} である"))
def _g_initial(sc: SimCtx, m: float, h: float) -> None:
    sc.overrides.update(initial_moisture=float(m), initial_health=float(h))


@given(parsers.parse("初期のバッテリー残量は {b:g}% である"))
def _g_initial_battery(sc: SimCtx, b: float) -> None:
    sc.overrides["initial_battery_pct"] = float(b)


@given(parsers.parse("{n:d}時間目だけ判断に失敗するエージェントを使う"))
def _g_failing(sc: SimCtx, n: int) -> None:
    sc.factory = _failing_at(n)


@given("毎時 A と B を行き来しようとするエージェントを使う")
def _g_pingpong(sc: SimCtx) -> None:
    sc.factory = _pingpong([])


@given("トレース用の実行情報を記録するエージェントを使う")
def _g_run_config_agent(sc: SimCtx) -> None:
    sc.factory = _run_config_recording(sc.run_configs)


@given("毎ターン固有の要約をJSONで返すLLMエージェントを使う")
def _g_numbered_llm(sc: SimCtx) -> None:
    sc.factory = _numbered_llm()


@given(parsers.parse('エージェント情報としてプロンプト "{v}" とモデル "{m}" が設定されている'))
def _g_agent_info(sc: SimCtx, v: str, m: str) -> None:
    sc.agent_info = {"prompt_version": v, "model": m}


@given(parsers.parse("1回の判断に {s:g} 秒かかる時計を使う"))
def _g_clock(sc: SimCtx, s: float) -> None:
    sc.clock = StepClock(float(s))


@when(
    parsers.parse(
        "1回の判断に {s:g} 秒かかる時計に替えて同じ条件でもう一度 {n:d}時間シミュレーションする"
    )
)
def _w_clock_rerun(sc: SimCtx, s: float, n: int) -> None:
    sc.clock = StepClock(float(s))
    sc.run(n)


@given("呼び出されるたびに仮想時刻を記録するエージェントを使う")
def _g_time_recording(sc: SimCtx) -> None:
    sc.factory = _time_recording(sc.call_times)


@given("毎ターン固有の要約を返すエージェントを使う")
def _g_unique(sc: SimCtx) -> None:
    sc.factory = _unique_summary(sc.seen)


@given("移動記録が次のとおりある")
def _g_moves(sc: SimCtx, datatable: list[list[str]]) -> None:
    header, *rows = datatable
    sc.moves = []
    for row in rows:
        rec = dict(zip(header, row, strict=True))
        a, b = rec["移動"].split("→")
        hh, mm = rec["時刻"].split(":")
        sc.moves.append(MoveEvent(datetime(2025, 8, 1, int(hh), int(mm)), a, b))


_BASE_STATE: dict[str, Any] = {
    "spot_id": "A",
    "health": 80.0,
    "heat_mitigation_score": 0.0,
    "battery_pct": 90.0,
    "water_tank_ml": 5000,
}


def _step(i: int, decision: TurnDecision, calls: list[ToolCallRecord] | None = None) -> StepRecord:
    return StepRecord(
        step=i,
        time=f"2025-08-01T{i % 24:02d}:00",
        state_before=_BASE_STATE,
        state_after=_BASE_STATE,
        tool_calls=calls or [],
        decision=decision,
    )


def _move_call(ok: bool, code: str = "") -> ToolCallRecord:
    result: dict[str, Any] = (
        {"ok": True, "new_spot_id": "B"} if ok else {"ok": False, "error": {"code": code}}
    )
    return ToolCallRecord(name="move_to_spot", args={"spot_id": "B"}, result=result)


@given("移動の試行が次の結果で記録されている")
def _g_attempts(sc: SimCtx, datatable: list[list[str]]) -> None:
    calls = [
        _move_call(True) if row[0] == "成功" else _move_call(False, row[0]) for row in datatable[1:]
    ]
    sc.synthetic = [_step(1, TurnDecision(action_summary="現状維持"), calls)]


@given(parsers.parse("要約が崩れたステップが {a:d} 件、正常なステップが {b:d} 件ある"))
def _g_malformed(sc: SimCtx, a: int, b: int) -> None:
    broken = TurnDecision(
        action_summary="#1: 移動 → 非該当", rationale="#1: 移動 → 非該当", raw_fallback=True
    )
    sc.synthetic = [_step(i + 1, broken) for i in range(a)]
    sc.synthetic += [_step(a + i + 1, TurnDecision(action_summary="現状維持")) for i in range(b)]


@given(parsers.parse('判断の行動要約は "{summary}" で、ツール呼び出しは {tool} である'))
def _g_claim(sc: SimCtx, summary: str, tool: str) -> None:
    calls: list[ToolCallRecord] = []
    if tool.startswith("失敗した "):
        calls = [
            ToolCallRecord(
                name=tool.removeprefix("失敗した "),
                args={"spot_id": "B"},
                result={"ok": False, "error": {"code": "X"}},
            )
        ]
    elif tool != "なし":
        args: dict[str, Any] = {"amount_ml": 500} if tool == "irrigate" else {"spot_id": "B"}
        result: dict[str, Any] = (
            {"ok": True, "new_spot_id": "B"} if tool == "move_to_spot" else {"ok": True}
        )
        calls = [ToolCallRecord(name=tool, args=args, result=result)]
    sc.synthetic = [_step(1, TurnDecision(action_summary=summary), calls)]


@given(parsers.parse("再確認を経たステップが {a:d} 件、経ていないステップが {b:d} 件ある"))
def _g_retried(sc: SimCtx, a: int, b: int) -> None:
    retried = TurnDecision(action_summary="Bへ移動した", claim_retry=True)
    plain = TurnDecision(action_summary="現状維持")
    calls = [_move_call(True)]  # 再確認の結果、移動が実行された
    sc.synthetic = [_step(i + 1, retried, calls) for i in range(a)]
    sc.synthetic += [_step(a + i + 1, plain) for i in range(b)]


@given("宣言のみとして記録されたステップが 1 件ある")
def _g_unexecuted_flag(sc: SimCtx) -> None:
    d = TurnDecision(
        action_summary="何もしない",
        claim_retry=True,
        unexecuted_claim=True,
        claimed_summary="Bへ移動した",
    )
    sc.synthetic = [_step(1, d)]


@given("散水の結果が次のとおり記録されている")
def _g_irrigations(sc: SimCtx, datatable: list[list[str]]) -> None:
    header, *rows = datatable
    calls: list[ToolCallRecord] = []
    for row in rows:
        rec = dict(zip(header, row, strict=True))
        requested, applied = int(rec["要求"]), int(rec["実際"])
        result: dict[str, Any] = {"ok": True}
        if rec["切り詰め"] == "あり":
            result |= {"truncated": True, "requested_ml": requested, "applied_ml": applied}
        calls.append(ToolCallRecord(name="irrigate", args={"amount_ml": requested}, result=result))
    sc.synthetic = [_step(1, TurnDecision(action_summary="散水"), calls)]


@given(parsers.parse("旧形式の散水の結果が要求量 {n:d}ml で記録されている"))
def _g_legacy_irrigation(sc: SimCtx, n: int) -> None:
    call = ToolCallRecord(name="irrigate", args={"amount_ml": n}, result={"ok": True})
    sc.synthetic = [_step(1, TurnDecision(action_summary="散水"), [call])]


@given("旧形式の判断（要約崩れの記録なし）で要約と判断理由が同一のステップが 1 件ある")
def _g_legacy(sc: SimCtx) -> None:
    legacy = {"action_summary": "#1: 降水確率 → 非該当", "rationale": "#1: 降水確率 → 非該当"}
    assert "raw_fallback" not in legacy  # 旧ログには無かったキー
    sc.synthetic = [_step(1, TurnDecision.model_validate(legacy))]


@given("散水の記録の直後の3時間以内に降水確率 70% 以上の時間がある")
def _g_irrigation_before_rain(sc: SimCtx) -> None:
    sc.irrigation_times = [datetime(2025, 8, 1, 13, 0)]  # ConstantWeather は常に降水確率 80%


@given(parsers.parse("判断失敗が {a:d}回、ツールエラーが {b:d}回 記録されている"))
def _g_synthetic(sc: SimCtx, a: int, b: int) -> None:
    from vmgs.agent.graph import ToolCallRecord

    base = {
        "spot_id": "A",
        "health": 80.0,
        "heat_mitigation_score": 0.0,
        "battery_pct": 90.0,
        "water_tank_ml": 5000,
    }
    errors = [
        ToolCallRecord(
            name="irrigate", args={"amount_ml": 100}, result={"ok": False, "error": {"code": "X"}}
        )
        for _ in range(b)
    ]
    sc.synthetic = [
        StepRecord(
            step=i + 1,
            time=f"2025-08-01T{i:02d}:00",
            state_before=base,
            state_after=base,
            tool_calls=errors if i == 0 else [],
            decision=TurnDecision(failed=i < a),
        )
        for i in range(max(a, 1))
    ]


# ---------------- When ----------------
@when(parsers.parse("ベースラインエージェントで {n:d}時間シミュレーションする"))
def _w_baseline(sc: SimCtx, n: int) -> None:
    sc.factory = BaselineAgent
    sc.run(n)


@when(parsers.parse("{n:d}時間シミュレーションする"))
@when(parsers.parse("同じ条件でもう一度 {n:d}時間シミュレーションする"))
def _w_run(sc: SimCtx, n: int) -> None:
    sc.run(n)


@when(parsers.parse("別の実行を {k:d}時間目で中断してから再開して最後まで実行する"))
def _w_interrupt_resume(sc: SimCtx, k: int) -> None:
    first = sc.last
    stopped = sc.run(sc.hours, stop_after=k)
    assert stopped.reason == "STOPPED"
    sc.results.pop()  # 中断した実行そのものは比較対象に含めない
    sc.results.append(sc.resume(stopped.run_id))
    assert sc.results[0] is first


@when(parsers.parse("{k:d}時間目で中断してから再開して {n:d}時間目まで実行する"))
def _w_interrupt_resume_n(sc: SimCtx, k: int, n: int) -> None:
    sc.run(n, stop_after=k)
    sc.seen.clear()  # 再開後の入力だけを検査する
    sc.results.append(sc.resume(sc.last.run_id))


@when("同じ実行を再開する")
def _w_resume_same(sc: SimCtx) -> None:
    sc.results.append(sc.resume(sc.last.run_id))


@when("存在しない実行IDで再開する")
def _w_resume_missing(sc: SimCtx) -> None:
    sc.weather = sc.weather or SeriesWeather(load_records(DATA / "normal.json"))
    try:
        sc.resume("no-such-run")
    except SimulationError as e:
        sc.error = e


@when("ベースラインと何もしないエージェントを同じ気象で比較する")
def _w_compare(sc: SimCtx) -> None:
    sc.report = compare_agents(
        sc.config(24),
        sc.require_weather(),
        {"baseline": BaselineAgent, "noop": NoopAgent},
        sc.runs_dir,
        "cmp",
    )


@when("無駄な移動を数える")
def _w_count_wasted(sc: SimCtx) -> None:
    sc.counted = count_wasted_moves(sc.moves)


@when("降雨直前の散水を数える")
def _w_count_rain(sc: SimCtx) -> None:
    sc.counted = count_irrigation_before_rain(sc.irrigation_times, sc.constant_weather)


@given("旧形式の判断（要約崩れの記録なし）で本文が判断メモのあとにJSONの1件がある")
def _g_legacy_with_json(sc: SimCtx) -> None:
    raw = (
        "#5: 土壌水分60%（30〜70%）、健康度100 → 何もしない\n\n"
        '{"action_summary": "何もしない", "rationale": "#5 に該当", "risk_notes": ""}'
    )
    legacy = {"action_summary": raw, "rationale": raw}  # 旧パーサーは本文を両方に入れていた
    sc.synthetic = [_step(1, TurnDecision.model_validate(legacy))]


@when("保存済みの実行から指標を再計算する")
def _w_recompute(sc: SimCtx) -> None:
    sc.recomputed = metrics_from_run(sc.last.run_dir, sc.require_weather())


@when("指標を算出する")
def _w_metrics(sc: SimCtx) -> None:
    sc.metrics = compute_metrics(sc.synthetic)


# ---------------- Then ----------------
def _dump(steps: list[StepRecord]) -> list[dict[str, Any]]:
    return [s.model_dump(mode="json") for s in steps]


@then(parsers.parse('終了理由は "{reason}" である'))
def _t_reason(sc: SimCtx, reason: str) -> None:
    assert sc.error is None, f"例外が発生: {sc.error!r}"
    assert sc.last.reason == reason


@then(parsers.parse("記録されたステップは {n:d} 件である"))
@then(parsers.parse("記録されたステップは {n:d} 件のままである"))
def _t_steps(sc: SimCtx, n: int) -> None:
    assert len(sc.last.steps) == n
    assert len(RunStore(sc.last.run_dir).read_steps()) == n


@then(parsers.parse("記録されたステップは {n:d} 件より少ない"))
def _t_steps_lt(sc: SimCtx, n: int) -> None:
    assert 0 < len(sc.last.steps) < n


@then("ステップの仮想時刻は 1時間ずつ増えている")
def _t_hourly(sc: SimCtx) -> None:
    times = [datetime.fromisoformat(s.time) for s in sc.last.steps]
    assert all(b - a == timedelta(hours=1) for a, b in itertools.pairwise(times))


@then("各ステップに 時刻・行動前の状態・行動後の状態・ツール呼び出し・判断 が含まれる")
def _t_step_fields(sc: SimCtx) -> None:
    for s in sc.last.steps:
        d = s.model_dump()
        assert {"time", "state_before", "state_after", "tool_calls", "decision"} <= set(d)
        assert "soil_moisture_pct" in s.state_before
        assert "soil_moisture_pct" in s.state_after


def _first_two(sc: SimCtx) -> tuple[SimulationResult, SimulationResult]:
    return sc.results[0], sc.results[1]


@then("2回の最終状態とステップ記録は一致する")
@then("2つの実行の最終状態は一致する")
def _t_same_final(sc: SimCtx) -> None:
    a, b = _first_two(sc)
    assert a.final_state == b.final_state
    assert _dump(a.steps) == _dump(b.steps)


@then("2つの実行のステップ記録は一致する")
def _t_same_steps(sc: SimCtx) -> None:
    a, b = _first_two(sc)
    assert len(a.steps) == len(b.steps) == 24
    assert _dump(a.steps) == _dump(b.steps)


@then(parsers.parse("成功した移動は {n:d} 回以下である"))
def _t_moves_at_most(sc: SimCtx, n: int) -> None:
    moves = [
        c
        for st in sc.last.steps
        for c in st.tool_calls
        if c.name == "move_to_spot" and c.result["ok"]
    ]
    assert 0 < len(moves) <= n, f"成功した移動: {len(moves)} 回"


@then("移動予算超過で拒否された試行がある")
def _t_budget_rejections(sc: SimCtx) -> None:
    codes = [
        c.result["error"]["code"]
        for st in sc.last.steps
        for c in st.tool_calls
        if not c.result["ok"]
    ]
    assert "MOVE_BUDGET_EXCEEDED" in codes


@then("例外は発生しない")
def _t_no_error(sc: SimCtx) -> None:
    assert sc.error is None, f"例外が発生: {sc.error!r}"


@then(parsers.parse("{n:d}時間目のステップは判断失敗として記録される"))
def _t_step_failed(sc: SimCtx, n: int) -> None:
    d = sc.last.steps[n - 1].decision
    assert d.failed is True
    assert d.error


@then(parsers.parse("{n:d}時間目のステップは通常どおり記録される"))
def _t_step_ok(sc: SimCtx, n: int) -> None:
    assert sc.last.steps[n - 1].decision.failed is False


@then(parsers.parse("記録された仮想時刻は 1時間おきの {n:d} 件である"))
def _t_call_times(sc: SimCtx, n: int) -> None:
    assert len(sc.call_times) == n
    assert all(b - a == timedelta(hours=1) for a, b in itertools.pairwise(sc.call_times))


@then(parsers.parse("{k:d}時間目の入力には {a:d}・{b:d}・{c:d}時間目の要約が含まれる"))
def _t_has_summaries(sc: SimCtx, k: int, a: int, b: int, c: int) -> None:
    assert sc.seen[k] == [f"要約{a}", f"要約{b}", f"要約{c}"]


@then(parsers.parse("{k:d}時間目の入力には {a:d}時間目の要約は含まれない"))
def _t_lacks_summary(sc: SimCtx, k: int, a: int) -> None:
    assert f"要約{a}" not in sc.seen[k]


@then(parsers.parse("実行ディレクトリに {name} が {n:d} 行ある"))
def _t_file_lines(sc: SimCtx, name: str, n: int) -> None:
    lines = (sc.last.run_dir / name).read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    if name == "steps.jsonl":
        assert all(json.loads(line)["decision"] is not None for line in lines)


@then(parsers.parse("実行ディレクトリに {name} がある"))
def _t_file_exists(sc: SimCtx, name: str) -> None:
    assert (sc.last.run_dir / name).exists()


@then(
    parsers.parse("{k:d}時間目終了時点のチェックポイントの仮想時刻は 開始から {h:d} 時間後 である")
)
def _t_cp_time(sc: SimCtx, k: int, h: int) -> None:
    store = RunStore(sc.last.run_dir)
    start = datetime.fromisoformat(sc.last.steps[0].time)
    assert store.load_checkpoint(k).sim_time == start + timedelta(hours=h)


@then(
    parsers.parse(
        "{k:d}時間目終了時点のチェックポイントの土壌水分は {k2:d}時間目のステップ記録の行動後と一致する"
    )
)
def _t_cp_moisture(sc: SimCtx, k: int, k2: int) -> None:
    cp = RunStore(sc.last.run_dir).load_checkpoint(k)
    assert cp.plant["moisture"] == pytest.approx(
        sc.last.steps[k2 - 1].state_after["soil_moisture_pct"], abs=0.01
    )


@then(parsers.parse('エラー "{code}" が返る'))
def _t_error_code(sc: SimCtx, code: str) -> None:
    assert isinstance(sc.error, SimulationError)
    assert sc.error.code == code


@then(parsers.parse("指標に {names} が含まれる"))
def _t_metric_keys(sc: SimCtx, names: str) -> None:
    dumped = sc.last.metrics.model_dump() if sc.last.metrics else {}
    for n in names.split("・"):
        assert METRIC_KEYS[n.strip()] in dumped, n


def _metrics(sc: SimCtx) -> Metrics:
    """合成した指標があればそれを、無ければ直近の実行の指標を返す。"""
    m = sc.metrics if sc.metrics is not None else sc.last.metrics
    assert m is not None
    return m


@then(parsers.parse("指標の移動回数は {n:d} である"))
def _t_metric_moves(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).move_count == n


@then(parsers.parse("指標の散水総量は {n:d}ml である"))
def _t_metric_irrigation(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).irrigation_total_ml == n


@then(parsers.parse("無駄な移動は {n:d} 回である"))
@then(parsers.parse("降雨直前の散水は {n:d} 回である"))
def _t_counted(sc: SimCtx, n: int) -> None:
    assert sc.counted == n


@then(parsers.parse("指標の判断失敗回数は {n:d} である"))
def _t_metric_failures(sc: SimCtx, n: int) -> None:
    assert sc.metrics is not None
    assert sc.metrics.decision_failures == n


@then(parsers.parse("指標のツールエラー回数は {n:d} である"))
def _t_metric_tool_errors(sc: SimCtx, n: int) -> None:
    assert sc.metrics is not None
    assert sc.metrics.tool_errors == n


@then(parsers.parse("指標の拒否された移動は {n:d} 回である"))
def _t_rejected(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).rejected_moves == n


@then(parsers.parse("指標のガードによる拒否は {n:d} 回である"))
def _t_guard_rejections(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).guard_rejections == n


@then(parsers.parse("指標の要約崩れは {n:d} である"))
def _t_malformed(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).malformed_summaries == n


@then(parsers.parse("指標の切り詰められた散水は {n:d} 回である"))
def _t_truncated_irrigations(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).truncated_irrigations == n


@then(parsers.parse("指標の再確認は {n:d} 回である"))
def _t_retries(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).claim_retries == n


@then(parsers.parse("指標の宣言のみは {n:d} である"))
def _t_unexecuted(sc: SimCtx, n: int) -> None:
    assert _metrics(sc).unexecuted_claims == n


@then(parsers.parse('取り出し直した行動要約は "{text}" である'))
def _t_reparsed_summary(sc: SimCtx, text: str) -> None:
    assert sc.synthetic[0].decision.action_summary == text


@then("再計算した指標は実行時の指標と一致する")
def _t_recomputed(sc: SimCtx) -> None:
    assert sc.recomputed is not None
    assert sc.last.metrics is not None
    assert sc.recomputed == sc.last.metrics


@then("比較レポートに両方のエージェントの指標が並ぶ")
def _t_report_rows(sc: SimCtx) -> None:
    assert sc.report is not None
    assert set(sc.report.rows) == {"baseline", "noop"}
    md = sc.report.to_markdown()
    assert "baseline" in md
    assert "noop" in md


@then("ベースラインの最終健康度は何もしない場合より高い")
def _t_baseline_better(sc: SimCtx) -> None:
    assert sc.report is not None
    assert sc.report.rows["baseline"].final_health > sc.report.rows["noop"].final_health
