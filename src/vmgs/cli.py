"""CLI。例: uv run vmgs turn --scenario heatwave --hour 13 --spot C [--dry-run]"""

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from vmgs.agent.baseline import BaselineAgent, NoopAgent
from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.llm import MissingApiKeyError, make_chat_model
from vmgs.agent.prompts import PROMPT_REQUIRES_PLAN_OBS, PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT
from vmgs.agent.protocol import AgentFactory, TurnAgent
from vmgs.agent.runner import AgentRunner
from vmgs.config import Settings
from vmgs.eval.metrics import Metrics, metrics_from_run
from vmgs.eval.report import ComparisonReport, compare_agents, plot_run
from vmgs.experiment.configs import THINK_TIMEOUT_S, ConfigSpec, parse_config_spec
from vmgs.experiment.runner import ExperimentSpec, run_experiment, run_suite
from vmgs.experiment.stress import STRESS_SUITE
from vmgs.experiment.variants import BaseConditions
from vmgs.obs.telemetry import estimate_cost_usd, summarize_telemetry
from vmgs.obs.tracing import apply_tracing, resolve_tracing
from vmgs.sim.presets import PRESETS
from vmgs.sim.simulate import SimConfig, SimulationError, resume_simulation, run_simulation
from vmgs.sim.store import RunStore
from vmgs.weather.cache import load_records
from vmgs.weather.providers import SeriesWeather
from vmgs.world.clock import SimClock
from vmgs.world.world import World

WEATHER_DIR = Path("data/weather")


def _dry_run_model() -> BaseChatModel:
    """APIキー不要の動作確認用。状態確認だけして現状維持と答える。"""
    final = '{"action_summary": "現状維持（dry-run）", "rationale": "Fake LLM", "risk_notes": ""}'
    from langchain_core.messages import AIMessage

    return ScriptedChatModel(
        responses=[tool_call_message("get_plant_state", {}, "call_0"), AIMessage(content=final)]
    )


def cmd_turn(args: argparse.Namespace) -> int:
    path = WEATHER_DIR / f"{args.scenario}.json"
    if not path.exists():
        print(
            f"シナリオが見つかりません: {path}（scripts/fetch_scenarios.py で作成）",
            file=sys.stderr,
        )
        return 2
    records = load_records(path)
    start = records[0].time.replace(hour=args.hour)
    world = World(
        SeriesWeather(records),
        SimClock(start),
        notes_path=Path(tempfile.mkdtemp()) / "notes.md",
        start_spot=args.spot,
    )
    world.plant.moisture = args.moisture
    try:
        overrides: dict[str, str] = {}
        if args.provider:
            overrides["llm_provider"] = args.provider
        if args.model:
            overrides["llm_model"] = args.model
        model = _dry_run_model() if args.dry_run else make_chat_model(Settings(**overrides))  # type: ignore[arg-type]
    except MissingApiKeyError as e:
        print(f"エラー: {e}\n（--dry-run なら APIキーなしで動作確認できます）", file=sys.stderr)
        return 2

    result = AgentRunner(model, world).run_turn()
    print(f"== {start:%Y-%m-%d %H:%M} スポット {args.spot} 水分 {args.moisture}% ==")
    for i, c in enumerate(result.tool_calls, 1):
        print(f"[{i}] {c.name}({json.dumps(c.args, ensure_ascii=False)}) -> ok={c.result['ok']}")
    d = result.decision
    print(f"判断: {d.action_summary}")
    print(f"理由: {d.rationale}")
    if d.risk_notes:
        print(f"懸念: {d.risk_notes}")
    if d.truncated:
        print("※ ツール呼び出し上限で打ち切り")
    if d.failed:
        print(f"※ LLM呼び出しに失敗: {d.error}")
        return 1
    return 0


RUNS_DIR = Path("runs")


def _llm_settings(provider: str | None, model: str | None, reasoning: bool = False) -> Settings:
    overrides: dict[str, object] = {}
    if provider:
        overrides["llm_provider"] = provider
    if model:
        overrides["llm_model"] = model
    if Settings(**overrides).llm_provider == "anthropic":  # type: ignore[arg-type]
        overrides["anthropic_thinking"] = "adaptive" if reasoning else "disabled"
    elif reasoning:
        # 思考モードは長く（または循環して）実験全体を止めうるので、タイムアウトを付ける
        overrides["ollama_reasoning"] = True
        overrides["ollama_timeout_s"] = THINK_TIMEOUT_S
    return Settings(**overrides)  # type: ignore[arg-type]


def _agent_factory(
    name: str,
    provider: str | None,
    model: str | None,
    prompt: str | None = None,
    reasoning: bool = False,
) -> AgentFactory:
    if name == "baseline":
        return BaselineAgent
    if name == "noop":
        return NoopAgent
    chat = make_chat_model(_llm_settings(provider, model, reasoning))
    system_prompt = PROMPTS[prompt] if prompt else SYSTEM_PROMPT
    return lambda world: AgentRunner(chat, world, system_prompt=system_prompt)


def _agent_info(
    name: str,
    provider: str | None,
    model: str | None,
    prompt: str | None,
    reasoning: bool = False,
) -> dict[str, str] | None:
    """実行の設定(run.json)とトレースのメタデータに残す、LLMの識別情報。"""
    if name in ("baseline", "noop"):
        return None
    settings = _llm_settings(provider, model)
    info = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "prompt_version": prompt or PROMPT_VERSION,
    }
    if reasoning:
        info["reasoning"] = "true"  # 思考オフの識別情報にはキーを足さない（従来と同一）
    return info


def _setup_tracing(args: argparse.Namespace) -> None:
    """--trace のときだけ LangSmith へ送る。何が送られるかを必ず表示する。"""
    if not getattr(args, "trace", False):
        return
    cfg = resolve_tracing(Settings(), enable=True, project=getattr(args, "trace_project", None))
    print(cfg.describe(), file=sys.stderr)
    for w in cfg.warnings:
        print(f"警告: {w}", file=sys.stderr)
    apply_tracing(cfg)


def _load_weather(scenario: str) -> SeriesWeather | None:
    path = WEATHER_DIR / f"{scenario}.json"
    if not path.exists():
        print(
            f"シナリオが見つかりません: {path}（scripts/fetch_scenarios.py で作成）",
            file=sys.stderr,
        )
        return None
    return SeriesWeather(load_records(path))


def _build_config(args: argparse.Namespace) -> SimConfig:
    base = PRESETS[args.preset] if args.preset else SimConfig(scenario=args.scenario)
    updates: dict[str, object] = {}
    if args.hours is not None:
        updates["hours"] = args.hours
    if args.spot is not None:
        updates["spot"] = args.spot
    if args.moisture is not None:
        updates["initial_moisture"] = args.moisture
    return base.model_copy(update=updates)


def _print_result(result: object, run_dir: Path) -> None:
    from vmgs.sim.simulate import SimulationResult

    assert isinstance(result, SimulationResult)
    print(f"実行: {result.run_id}  終了理由: {result.reason}  ステップ: {len(result.steps)}")
    m = result.metrics
    if m is not None:
        print(
            f"最終健康度 {m.final_health:.1f} / 最低 {m.min_health:.1f} / 暑熱緩和 {m.heat_mitigation_score:.1f} / "
            f"移動 {m.move_count}（無駄 {m.wasted_moves}）/ 散水 {m.irrigation_total_ml}ml（雨前 {m.irrigation_before_rain}）/ "
            f"電池 {m.final_battery:.1f}% / タンク {m.final_tank_ml}ml / 判断失敗 {m.decision_failures} / 要約崩れ {m.malformed_summaries} / 宣言のみ {m.unexecuted_claims}（再確認 {m.claim_retries}） / 拒否移動 {m.rejected_moves}（ガード {m.guard_rejections}） / ツールエラー {m.tool_errors}"
        )
    if result.steps:
        plot_run(result.steps, run_dir / "timeseries.png", result.run_id)
        print(f"ログ: {run_dir}（steps.jsonl / summary.json / timeseries.png）")


def cmd_run(args: argparse.Namespace) -> int:
    _setup_tracing(args)
    try:
        if args.resume:
            meta = (
                RunStore(RUNS_DIR / args.resume).read_meta()
                if (RUNS_DIR / args.resume / "run.json").exists()
                else None
            )
            if meta is None:
                print(f"エラー: 実行 {args.resume!r} が見つかりません", file=sys.stderr)
                return 2
            config = SimConfig.model_validate(meta["config"])
            weather = _load_weather(config.scenario)
            if weather is None:
                return 2
            recorded_think = (meta.get("agent_info") or {}).get("reasoning") == "true"
            factory = _agent_factory(
                args.agent or meta["agent"],
                args.provider,
                args.model,
                args.prompt,
                reasoning=args.think or recorded_think,
            )
            result = resume_simulation(
                args.resume, RUNS_DIR, weather, factory, stop_after=args.stop_after
            )
        else:
            config = _build_config(args)
            if args.prompt in PROMPT_REQUIRES_PLAN_OBS:
                config = config.model_copy(update={"planning_observations": True})
            weather = _load_weather(config.scenario)
            if weather is None:
                return 2
            agent_name = args.agent or "baseline"
            if args.think and (args.provider or Settings().llm_provider) != "ollama":
                print("エラー: --think（思考モード）は Ollama のみ対応です", file=sys.stderr)
                return 2
            factory = _agent_factory(
                agent_name, args.provider, args.model, args.prompt, reasoning=args.think
            )
            run_id = args.run_id or f"{datetime.now():%Y%m%d-%H%M%S}_{config.scenario}_{agent_name}"
            result = run_simulation(
                config,
                weather,
                factory,
                RUNS_DIR,
                run_id,
                agent_name=agent_name,
                agent_info=_agent_info(
                    agent_name, args.provider, args.model, args.prompt, reasoning=args.think
                ),
                stop_after=args.stop_after,
            )
    except (MissingApiKeyError, SimulationError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    _print_result(result, result.run_dir)
    return 0


def cmd_metrics(args: argparse.Namespace) -> int:
    """保存済みの実行から、現在の定義で指標を再計算して並べる。"""
    rows: dict[str, Metrics] = {}
    scenario = ""
    for run_id in args.run_ids:
        store = RunStore(RUNS_DIR / run_id)
        if not store.exists():
            print(f"エラー: 実行 {run_id!r} が見つかりません", file=sys.stderr)
            return 2
        config = SimConfig.model_validate(store.read_meta()["config"])
        scenario = scenario or config.scenario
        rows[run_id] = metrics_from_run(store.dir, _load_weather(config.scenario))
    print(ComparisonReport(scenario=scenario, rows=rows).to_markdown())
    return 0


def cmd_telemetry(args: argparse.Namespace) -> int:
    """保存済みの実行のトークン・レイテンシ・コスト概算を表示する。"""
    for run_id in args.run_ids:
        store = RunStore(RUNS_DIR / run_id)
        if not store.exists():
            print(f"エラー: 実行 {run_id!r} が見つかりません", file=sys.stderr)
            return 2
        rows = store.read_telemetry()
        if not rows:
            print(f"{run_id}: テレメトリがありません（旧形式の実行）")
            continue
        s = summarize_telemetry(rows)
        reported = [r for r in rows if r.total_tokens is not None]
        tokens = f"{s.total_tokens}" if reported else "報告なし"
        info = store.read_meta().get("agent_info") or {}
        cost = "－"
        if s.llm_calls and info.get("model") and reported:
            usd = estimate_cost_usd(
                info["model"],
                sum(r.input_tokens or 0 for r in rows),
                sum(r.output_tokens or 0 for r in rows),
                provider=info.get("provider"),
            )
            cost = "不明（単価表に無いモデル）" if usd is None else f"約 ${usd:.4f}"
        print(
            f"{run_id}: {s.steps}ステップ / LLM呼び出し {s.llm_calls} / トークン {tokens} / "
            f"合計 {s.total_latency_s:.1f}秒（平均 {s.mean_latency_s:.1f}・p95 {s.p95_latency_s:.1f}）/ コスト {cost}"
        )
    return 0


def _experiment_specs(args: argparse.Namespace) -> dict[str, ConfigSpec]:
    return {spec.name: spec for spec in (parse_config_spec(t) for t in args.configs.split(","))}


def cmd_experiment(args: argparse.Namespace) -> int:
    if args.suite:
        return _cmd_suite(args)
    _setup_tracing(args)
    base_config = _build_config(args)
    weather = _load_weather(base_config.scenario)
    if weather is None:
        return 2
    try:
        specs = _experiment_specs(args)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    configs: dict[str, AgentFactory] = {}
    infos: dict[str, dict[str, str]] = {}
    for name, spec in specs.items():
        if spec.kind == "llm":
            # モデルの生成は実行時まで遅らせる（APIキー未設定などは、その構成だけの失敗として記録される）
            configs[name] = _lazy_llm_factory(spec)
            infos[name] = (
                _agent_info("llm", spec.provider, spec.model, spec.prompt, spec.reasoning) or {}
            )
        else:
            configs[name] = _agent_factory(spec.kind, None, None)
    base = BaseConditions(
        moisture=base_config.initial_moisture if base_config.initial_moisture is not None else 60.0,
        tank_ml=base_config.initial_tank_ml if base_config.initial_tank_ml is not None else 10_000,
        battery_pct=100.0,
        spot=base_config.spot,
        start_hour=base_config.start_hour,
    )
    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}_{base_config.scenario}_exp"
    exp_spec = ExperimentSpec(
        name=name,
        scenario=base_config.scenario,
        configs=configs,
        variants=args.variants,
        seed=args.seed,
        hours=base_config.hours,
        base=base,
        agent_infos=infos,
        planning_configs={n for n, sp in specs.items() if sp.planning_observations},
        truncate_irrigation=not args.no_truncate,
    )
    result = run_experiment(exp_spec, weather, RUNS_DIR, max_runs=args.max_runs)
    print(result.report_md)
    print(
        f"保存先: {result.experiment_dir}（experiment.json / report.md）。各実行は runs/{name}_<構成>_v<番号>/"
    )
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    """余裕の少ない状況（--suite stress）を、状況ごとに別々に集計して比較する。"""
    _setup_tracing(args)
    names = args.conditions.split(",") if args.conditions else list(STRESS_SUITE)
    unknown = [n for n in names if n not in STRESS_SUITE]
    if unknown:
        print(f"エラー: 不明な状況 {unknown}（{', '.join(STRESS_SUITE)}）", file=sys.stderr)
        return 2
    try:
        specs = _experiment_specs(args)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    configs: dict[str, AgentFactory] = {}
    infos: dict[str, dict[str, str]] = {}
    for name, spec in specs.items():
        if spec.kind == "llm":
            configs[name] = _lazy_llm_factory(spec)
            infos[name] = (
                _agent_info("llm", spec.provider, spec.model, spec.prompt, spec.reasoning) or {}
            )
        else:
            configs[name] = _agent_factory(spec.kind, None, None)

    def weather_for(scenario: str) -> SeriesWeather:
        return SeriesWeather(load_records(WEATHER_DIR / f"{scenario}.json"))

    name = args.name or f"{datetime.now():%Y%m%d-%H%M%S}_stress"
    result = run_suite(
        name,
        [STRESS_SUITE[n] for n in names],
        configs,
        weather_for,
        RUNS_DIR,
        variants=args.variants,
        seed=args.seed,
        max_runs=args.max_runs,
        agent_infos=infos,
        planning_configs={n for n, sp in specs.items() if sp.planning_observations},
        truncate_irrigation=not args.no_truncate,
    )
    print(result.summary_md)
    print(
        f"保存先: {result.suite_dir}（suite.json / summary.md）。各実行は runs/{name}_<状況>_<構成>_v<番号>/"
    )
    return 0


def _lazy_llm_factory(spec: ConfigSpec) -> AgentFactory:
    def make(world: World) -> TurnAgent:
        return _agent_factory("llm", spec.provider, spec.model, spec.prompt, spec.reasoning)(world)

    return make


def cmd_compare(args: argparse.Namespace) -> int:
    config = _build_config(args)
    weather = _load_weather(config.scenario)
    if weather is None:
        return 2
    try:
        factories = {
            name: _agent_factory(name, args.provider, args.model, args.prompt)
            for name in args.agents.split(",")
        }
    except MissingApiKeyError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 2
    prefix = f"{datetime.now():%Y%m%d-%H%M%S}_{config.scenario}_cmp"
    report = compare_agents(config, weather, factories, RUNS_DIR, prefix)
    md = report.to_markdown()
    print(md)
    (RUNS_DIR / f"{prefix}.md").write_text(md, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vmgs")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("turn", help="1ターン（仮想1時間）だけエージェントに判断させる")
    t.add_argument("--scenario", default="heatwave", choices=["heatwave", "normal", "rainy"])
    t.add_argument("--hour", type=int, default=13)
    t.add_argument("--spot", default="C", choices=["A", "B", "C"])
    t.add_argument("--moisture", type=float, default=30.0)
    t.add_argument(
        "--provider", choices=["openai", "anthropic", "ollama"], help="LLM_PROVIDER を上書き"
    )
    t.add_argument("--model", help="LLM_MODEL を上書き（例: qwen3:14b）")
    t.add_argument("--dry-run", action="store_true", help="Fake LLM で実行（APIキー不要）")
    t.set_defaults(func=cmd_turn)

    def add_sim_options(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--scenario", default="heatwave", choices=["heatwave", "normal", "rainy"])
        sp.add_argument(
            "--preset",
            choices=sorted(PRESETS),
            help="気象+初期条件のプリセット（--scenario より優先）",
        )
        sp.add_argument("--hours", type=int, help="シミュレーション時間（既定: 24 / preset の値）")
        sp.add_argument("--spot", choices=["A", "B", "C"])
        sp.add_argument("--moisture", type=float, help="初期の土壌水分[%%]")
        sp.add_argument("--provider", choices=["openai", "anthropic", "ollama"])
        sp.add_argument("--model")
        sp.add_argument(
            "--prompt", choices=sorted(PROMPTS), help="システムプロンプトのバージョン（既定: 最新）"
        )

    r = sub.add_parser("run", help="24時間シミュレーションを実行する")
    add_sim_options(r)
    r.add_argument(
        "--agent",
        choices=["baseline", "noop", "llm"],
        help="既定: baseline（--resume では記録済みの値）",
    )
    r.add_argument("--run-id")
    r.add_argument("--stop-after", type=int, help="N ステップで中断する（--resume で再開できる）")
    r.add_argument("--resume", metavar="RUN_ID", help="チェックポイントから再開する")
    r.set_defaults(func=cmd_run)

    r.add_argument(
        "--think",
        action="store_true",
        help="Ollama の思考モードをオンにする（タイムアウト付き。トレースに思考の全文を保存）",
    )
    r.add_argument(
        "--trace",
        action="store_true",
        help="LangSmith へトレースを送る（要 APIキー。プロンプトと応答が外部へ送られる）",
    )
    r.add_argument("--trace-project", help="LangSmith のプロジェクト名（既定: vmgs）")

    tl = sub.add_parser(
        "telemetry", help="保存済みの実行のトークン・レイテンシ・コスト概算を表示する"
    )
    tl.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    tl.set_defaults(func=cmd_telemetry)

    ex = sub.add_parser("experiment", help="複数の構成を、初期条件を変えたバリアントで比較する")
    add_sim_options(ex)
    ex.add_argument(
        "--name",
        help="実験名（既定: 日時_シナリオ_exp）。同じ名前なら完了済みの実行を再利用して再開する",
    )
    ex.add_argument(
        "--configs",
        default="baseline,noop",
        help="カンマ区切り。baseline / noop / llm:<provider>:<model>[:v数字]",
    )
    ex.add_argument("--variants", type=int, default=5)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument(
        "--suite",
        choices=["stress"],
        help="余裕の少ない状況を状況ごとに比較する（--scenario は無視）",
    )
    ex.add_argument(
        "--conditions",
        help="--suite の状況をカンマ区切りで絞り込む（例: dry_before_rain,wet_rain）",
    )
    ex.add_argument(
        "--no-truncate",
        action="store_true",
        help="残量を超える散水を切り詰めず、INSUFFICIENT_WATER で拒否する（切り詰め導入前の条件）",
    )
    ex.add_argument("--max-runs", type=int, help="この呼び出しで新たに実行する件数の上限")
    ex.add_argument("--trace", action="store_true", help="LangSmith へトレースを送る")
    ex.add_argument("--trace-project")
    ex.set_defaults(func=cmd_experiment)

    mt = sub.add_parser("metrics", help="保存済みの実行から指標を再計算して並べる")
    mt.add_argument("run_ids", nargs="+", metavar="RUN_ID")
    mt.set_defaults(func=cmd_metrics)

    c = sub.add_parser("compare", help="複数のエージェントを同じ気象で比較する")
    add_sim_options(c)
    c.add_argument("--agents", default="baseline,noop", help="カンマ区切り（baseline,noop,llm）")
    c.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
