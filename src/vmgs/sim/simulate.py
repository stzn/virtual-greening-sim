"""24時間タイムステップ・シミュレーション。時間を進めるのはここだけで、エージェントは進めない。"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vmgs.agent.graph import TurnDecision
from vmgs.agent.prompts import TurnSummary
from vmgs.agent.protocol import AgentFactory, TurnAgent
from vmgs.eval.metrics import Metrics, compute_metrics
from vmgs.obs.telemetry import (
    StepTelemetry,
    summarize_telemetry,
    trace_record,
    usage_from_messages,
)
from vmgs.obs.tracing import build_run_config
from vmgs.sim.records import Checkpoint, StepRecord, plant_from_dict, plant_to_dict
from vmgs.sim.store import RunStore
from vmgs.weather.providers import SeriesWeather, WeatherUnavailableError
from vmgs.world.clock import SimClock
from vmgs.world.plant import UnitStatus
from vmgs.world.world import World

logger = logging.getLogger(__name__)

HISTORY_TURNS = 3
TERMINAL_REASONS = {"COMPLETED", "DEAD", "WEATHER_EXHAUSTED"}

WallClock = Callable[[], float]  # 秒。テストでは差し替えて、レイテンシを決定論にする


class SimulationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class SimConfig(BaseModel):
    scenario: str
    hours: int = Field(default=24, ge=1)
    start_hour: int = Field(default=0, ge=0, le=23)
    start_time: datetime | None = None
    spot: str = "A"
    initial_moisture: float | None = None
    initial_health: float | None = None
    initial_battery_pct: float | None = None
    initial_tank_ml: int | None = None
    # 計画観測（残り時間あたりの水・電池）を get_plant_state に返す。既定はオフで、オフなら出力は変わらない
    planning_observations: bool = False


@dataclass
class SimulationResult:
    run_id: str
    run_dir: Path
    reason: str  # COMPLETED / DEAD / WEATHER_EXHAUSTED / STOPPED
    steps: list[StepRecord]
    final_state: dict[str, Any]
    metrics: Metrics | None


@dataclass
class _Run:
    """1回の実行（新規でも再開でも）に共通する文脈。"""

    config: SimConfig
    weather: SeriesWeather
    store: RunStore
    run_id: str
    agent_info: dict[str, str]
    clock: WallClock
    stop_after: int | None


def _build_world(config: SimConfig, weather: SeriesWeather, store: RunStore) -> World:
    assert config.start_time is not None
    world = World(
        weather,
        SimClock(config.start_time),
        notes_path=store.notes_path,
        start_spot=config.spot,
        # 終了時刻は設定（開始時刻 + 時間数）から決める。再開しても同じ値になる
        horizon_end=config.start_time + timedelta(hours=config.hours),
        planning_observations=config.planning_observations,
    )
    p = world.plant
    if config.initial_moisture is not None:
        p.moisture = config.initial_moisture
    if config.initial_health is not None:
        p.health = config.initial_health
    if config.initial_battery_pct is not None:
        p.battery_pct = config.initial_battery_pct
    if config.initial_tank_ml is not None:
        p.water_tank_ml = config.initial_tank_ml
    return world


def run_simulation(
    config: SimConfig,
    weather: SeriesWeather,
    agent_factory: AgentFactory,
    runs_dir: Path,
    run_id: str,
    *,
    agent_name: str = "custom",
    agent_info: dict[str, str] | None = None,
    stop_after: int | None = None,
    clock: WallClock = time.perf_counter,
) -> SimulationResult:
    start = config.start_time or weather.first_time.replace(hour=config.start_hour)
    config = config.model_copy(update={"start_time": start})
    store = RunStore(runs_dir / run_id)
    if store.exists():
        raise SimulationError("RUN_EXISTS", f"実行 {run_id!r} はすでに存在します")
    info = dict(agent_info or {})
    store.init(
        {
            "run_id": run_id,
            "agent": agent_name,
            "agent_info": info,
            "config": config.model_dump(mode="json"),
        },
        title=f"{run_id}（{agent_name}）",
    )
    world = _build_world(config, weather, store)
    run = _Run(config, weather, store, run_id, info, clock, stop_after)
    return _loop(run, world, agent_factory(world), [], [])


def resume_simulation(
    run_id: str,
    runs_dir: Path,
    weather: SeriesWeather,
    agent_factory: AgentFactory,
    *,
    stop_after: int | None = None,
    clock: WallClock = time.perf_counter,
) -> SimulationResult:
    """最後のチェックポイントから再開する。完了済みの実行は何もせず結果を返す。"""
    store = RunStore(runs_dir / run_id)
    if not store.exists():
        raise SimulationError("RUN_NOT_FOUND", f"実行 {run_id!r} が見つかりません")
    meta = store.read_meta()
    config = SimConfig.model_validate(meta["config"])
    summary = store.read_summary()
    steps = store.read_steps()
    if summary is not None and summary["reason"] in TERMINAL_REASONS:
        metrics = compute_metrics(steps, weather) if steps else None
        return SimulationResult(
            run_id, store.dir, summary["reason"], steps, summary["final_state"], metrics
        )

    cp_step = store.latest_checkpoint_step()
    steps = steps[:cp_step]
    store.truncate_steps(cp_step)  # チェックポイントより先に書かれた分は捨てて整合させる
    store.truncate_side_files(cp_step)
    world = _build_world(config, weather, store)
    history: list[TurnSummary] = []
    if cp_step > 0:
        cp = store.load_checkpoint(cp_step)
        world.plant = plant_from_dict(cp.plant)
        world.clock.now = cp.sim_time
        history = cp.history
    run = _Run(config, weather, store, run_id, meta.get("agent_info", {}), clock, stop_after)
    return _loop(run, world, agent_factory(world), steps, history)


def _loop(
    run: _Run,
    world: World,
    agent: TurnAgent,
    steps: list[StepRecord],
    history: list[TurnSummary],
) -> SimulationResult:
    config, weather, store = run.config, run.weather, run.store
    reason = "COMPLETED"
    while len(steps) < config.hours:
        if world.plant.status is UnitStatus.DEAD:
            reason = "DEAD"
            break
        if run.stop_after is not None and len(steps) >= run.stop_after:
            reason = "STOPPED"
            break
        try:
            weather.at(world.clock.now)
        except WeatherUnavailableError:
            reason = "WEATHER_EXHAUSTED"
            break
        rec, telemetry, trace = _run_step(len(steps) + 1, run, world, agent, history)
        steps.append(rec)
        store.append_step(rec)
        store.append_telemetry(telemetry)
        store.append_trace(trace)
        store.save_checkpoint(
            Checkpoint(
                step=rec.step,
                sim_time=world.clock.now,
                plant=plant_to_dict(world.plant),
                history=list(history),
            )
        )
    else:
        if world.plant.status is UnitStatus.DEAD:
            reason = "DEAD"

    final_state = world.state().model_dump(mode="json")
    metrics = compute_metrics(steps, weather) if steps else None
    store.write_summary(
        {
            "reason": reason,
            "steps": len(steps),
            "final_state": final_state,
            "metrics": metrics.model_dump() if metrics else None,
            "telemetry": summarize_telemetry(store.read_telemetry()).model_dump(),
        }
    )
    return SimulationResult(run.run_id, store.dir, reason, steps, final_state, metrics)


def _run_step(
    n: int, run: _Run, world: World, agent: TurnAgent, history: list[TurnSummary]
) -> tuple[StepRecord, StepTelemetry, dict[str, Any]]:
    before = world.state().model_dump(mode="json")
    now = world.clock.now
    agent.history = list(history[-HISTORY_TURNS:])
    agent.run_config = build_run_config(
        run_id=run.run_id,
        scenario=run.config.scenario,
        step=n,
        sim_time=now.isoformat(timespec="minutes"),
        prompt_version=run.agent_info.get("prompt_version"),
        model=run.agent_info.get("model"),
    )
    messages: list[Any] = []
    t0 = run.clock()
    try:
        result = agent.run_turn()
        decision, calls, messages = result.decision, result.tool_calls, list(result.messages)
    except Exception as e:  # エージェントの失敗でシミュレーション全体を止めない
        logger.exception("ステップ %d の判断に失敗しました", n)
        decision = TurnDecision(
            action_summary="判断失敗（エージェント例外）",
            failed=True,
            error=f"{type(e).__name__}: {e}",
        )
        calls = []
    latency = run.clock() - t0
    world.advance(1)  # 時間を進めるのはランナーだけ
    history.append(TurnSummary(time=f"{now:%H:%M}", action_summary=decision.action_summary))
    when = now.isoformat(timespec="minutes")
    usage = usage_from_messages(messages)
    rec = StepRecord(
        step=n,
        time=when,
        state_before=before,
        state_after=world.state().model_dump(mode="json"),
        tool_calls=calls,
        decision=decision,
    )
    telemetry = StepTelemetry(
        step=n,
        latency_s=latency,
        llm_calls=usage.llm_calls,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
    )
    return rec, telemetry, trace_record(n, when, messages)
