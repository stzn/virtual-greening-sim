"""複数回・複数構成の比較実験。

構成（エージェント・プロンプト・モデル）× バリアント の全組み合わせを実行し、構成ごとに指標を集計する。
同じバリアント同士で対にして比較し、完了済みの実行は再利用する（実験は途中から再開できる）。
構成の失敗は他の構成へ波及させない。
"""

import json
import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pydantic import BaseModel

from vmgs.agent.protocol import AgentFactory
from vmgs.eval.metrics import Metrics
from vmgs.experiment.stress import Condition
from vmgs.experiment.variants import BaseConditions, make_variant
from vmgs.sim.records import StepRecord
from vmgs.sim.simulate import (
    TERMINAL_REASONS,
    SimConfig,
    SimulationResult,
    WallClock,
    resume_simulation,
    run_simulation,
)
from vmgs.sim.store import RunStore
from vmgs.weather.providers import SeriesWeather

logger = logging.getLogger(__name__)

TIE_TOLERANCE = 1.0  # 最終健康度の差がこの範囲なら引き分け
MIN_SAMPLES = 5  # これ未満は、差が偶然の可能性があると警告する
CEILING_HEALTH = 99.0  # 基準構成の最終健康度の平均がこれ以上なら、上限に張り付いて差を測れない
REPORT_COLUMNS: list[tuple[str, str]] = [
    ("final_health", "最終健康度"),
    ("min_health", "最低健康度"),
    ("heat_mitigation_score", "暑熱緩和"),
    ("move_count", "移動"),
    ("wasted_moves", "無駄な移動"),
    ("irrigation_total_ml", "散水(ml)"),
    ("final_battery", "電池%"),
    ("unexecuted_claims", "宣言のみ"),
]


class Stat(BaseModel):
    mean: float
    std: float | None  # 標本が1件のときは算出できない
    min: float
    max: float
    n: int


def summarize_values(values: Sequence[float]) -> Stat:
    n = len(values)
    return Stat(
        mean=statistics.fmean(values),
        std=statistics.stdev(values) if n >= 2 else None,  # 標本標準偏差
        min=min(values),
        max=max(values),
        n=n,
    )


def aggregate_metrics(by_config: dict[str, list[Metrics]]) -> dict[str, dict[str, Stat]]:
    out: dict[str, dict[str, Stat]] = {}
    for cfg, rows in by_config.items():
        if not rows:
            continue
        out[cfg] = {
            name: summarize_values([float(getattr(m, name)) for m in rows])
            for name in Metrics.model_fields
        }
    return out


def compare_paired(
    baseline: dict[int, float], other: dict[int, float], tolerance: float = TIE_TOLERANCE
) -> tuple[int, int, int]:
    """同じバリアント同士の差で、other の（勝ち, 引き分け, 負け）を数える。"""
    wins = ties = losses = 0
    for idx in sorted(baseline.keys() & other.keys()):
        diff = other[idx] - baseline[idx]
        if abs(diff) <= tolerance:
            ties += 1
        elif diff > 0:
            wins += 1
        else:
            losses += 1
    return wins, ties, losses


@dataclass
class ExperimentSpec:
    name: str
    scenario: str
    configs: dict[str, AgentFactory]
    variants: int = 5
    seed: int = 0
    hours: int = 24
    base: BaseConditions = field(default_factory=BaseConditions)
    agent_infos: dict[str, dict[str, str]] = field(default_factory=dict)
    # 計画観測（残り時間あたりの水・電池）を get_plant_state に返す構成。他の構成の世界は変わらない
    planning_configs: set[str] = field(default_factory=set)
    condition_configs: set[str] = field(default_factory=set)
    # 残量を超える散水の切り詰め。False にすると、全構成の世界で切り詰めをやめる（過去の比較との互換用）
    truncate_irrigation: bool = True


@dataclass
class RunOutcome:
    config: str
    variant: int
    run_id: str
    status: str  # COMPLETED / ABORTED / FAILED
    metrics: Metrics | None = None
    reason: str | None = None


@dataclass
class ExperimentResult:
    name: str
    scenario: str
    outcomes: list[RunOutcome]
    aborted: dict[str, str]  # 構成 → 打ち切り理由（全ステップが判断失敗など）
    failed: dict[str, str]  # 構成 → 失敗理由（作成時の例外など）
    stats: dict[str, dict[str, Stat]]
    baseline: str | None
    pairs: dict[str, tuple[int, int, int]]
    warnings: list[str]
    report_md: str
    experiment_dir: Path
    executed: int = 0  # この呼び出しで新たに実行した件数（完了済みの再利用は含まない）

    def completed(self, config: str) -> list[RunOutcome]:
        return [o for o in self.outcomes if o.config == config and o.status == "COMPLETED"]


def _abort_reason(steps: Sequence[StepRecord]) -> str:
    err = steps[0].decision.error or ""
    if "does not support tools" in err:
        return (
            f"全ステップが判断失敗: モデルがツール呼び出しに対応していない可能性があります（{err}）"
        )
    return f"全ステップが判断失敗（{err}）"


def _sim_config(spec: ExperimentSpec, index: int, cfg: str) -> SimConfig:
    v = make_variant(spec.base, spec.seed, index)
    return SimConfig(
        scenario=spec.scenario,
        hours=spec.hours,
        start_hour=v.start_hour,
        spot=v.spot,
        initial_moisture=v.moisture,
        initial_tank_ml=v.tank_ml,
        initial_battery_pct=v.battery_pct,
        planning_observations=cfg in spec.planning_configs,
        condition_observations=cfg in spec.condition_configs,
        truncate_irrigation=spec.truncate_irrigation,
    )


def _is_complete(runs_dir: Path, run_id: str) -> bool:
    summary = RunStore(runs_dir / run_id).read_summary()
    return summary is not None and summary["reason"] in TERMINAL_REASONS


def run_experiment(
    spec: ExperimentSpec,
    weather: SeriesWeather,
    runs_dir: Path,
    *,
    max_runs: int | None = None,
    clock: WallClock = time.perf_counter,
) -> ExperimentResult:
    """max_runs: この呼び出しで新たに実行する上限（完了済みの再利用は数えない）。"""
    exp_dir = runs_dir / spec.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "scenario": spec.scenario,
                "seed": spec.seed,
                "configs": list(spec.configs),
                "variants": spec.variants,
                "hours": spec.hours,
                "base": {
                    "moisture": spec.base.moisture,
                    "tank_ml": spec.base.tank_ml,
                    "battery_pct": spec.base.battery_pct,
                    "spot": spec.base.spot,
                    "start_hour": spec.base.start_hour,
                },
                "agent_infos": spec.agent_infos,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )

    outcomes: list[RunOutcome] = []
    aborted: dict[str, str] = {}
    failed: dict[str, str] = {}
    executed = 0
    out_of_budget = False

    for cfg, factory in spec.configs.items():
        if out_of_budget:
            break
        for i in range(spec.variants):
            run_id = f"{spec.name}_{cfg}_v{i}"
            reused = _is_complete(runs_dir, run_id)
            if not reused and max_runs is not None and executed >= max_runs:
                out_of_budget = True
                break
            try:
                res = _run_one(spec, weather, runs_dir, cfg, factory, i, run_id, clock)
            except Exception as e:  # 構成の失敗は他の構成へ波及させない
                logger.exception("構成 %s の実行 %s に失敗しました", cfg, run_id)
                failed[cfg] = f"{type(e).__name__}: {e}"
                outcomes.append(RunOutcome(cfg, i, run_id, "FAILED", reason=failed[cfg]))
                break
            if not reused:
                executed += 1
            if res.steps and all(s.decision.failed for s in res.steps):
                aborted[cfg] = _abort_reason(res.steps)
                outcomes.append(RunOutcome(cfg, i, run_id, "ABORTED", reason=aborted[cfg]))
                break  # 残りのバリアントも同じ失敗になるため実行しない
            outcomes.append(RunOutcome(cfg, i, run_id, "COMPLETED", metrics=res.metrics))

    return _finish(spec, outcomes, aborted, failed, exp_dir, executed)


def _run_one(
    spec: ExperimentSpec,
    weather: SeriesWeather,
    runs_dir: Path,
    cfg: str,
    factory: AgentFactory,
    index: int,
    run_id: str,
    clock: WallClock,
) -> SimulationResult:
    if (runs_dir / run_id / "run.json").exists():
        return resume_simulation(run_id, runs_dir, weather, factory, clock=clock)
    return run_simulation(
        _sim_config(spec, index, cfg),
        weather,
        factory,
        runs_dir,
        run_id,
        agent_name=cfg,
        agent_info=spec.agent_infos.get(cfg),
        clock=clock,
    )


def _finish(
    spec: ExperimentSpec,
    outcomes: list[RunOutcome],
    aborted: dict[str, str],
    failed: dict[str, str],
    exp_dir: Path,
    executed: int,
) -> ExperimentResult:
    by_config: dict[str, list[Metrics]] = {}
    for o in outcomes:
        if o.status == "COMPLETED" and o.metrics is not None:
            by_config.setdefault(o.config, []).append(o.metrics)
    stats = aggregate_metrics(by_config)

    baseline = "baseline" if "baseline" in spec.configs else next(iter(spec.configs), None)
    health: dict[str, dict[int, float]] = {}
    for o in outcomes:
        if o.status == "COMPLETED" and o.metrics is not None:
            health.setdefault(o.config, {})[o.variant] = o.metrics.final_health
    pairs = {
        cfg: compare_paired(health[baseline], vals)
        for cfg, vals in health.items()
        if baseline in health and cfg != baseline
    }

    warnings: list[str] = []
    sample_sizes = [s["final_health"].n for s in stats.values()]
    if sample_sizes and min(sample_sizes) < MIN_SAMPLES:
        warnings.append(
            f"標本数が少ない（n={min(sample_sizes)} < {MIN_SAMPLES}）ため、構成間の差は偶然の可能性があります。"
        )
    if baseline in stats and stats[baseline]["final_health"].mean >= CEILING_HEALTH:
        warnings.append(
            f"天井効果: 基準構成 {baseline} の最終健康度が上限（100）に張り付いており、構成間の差を検出できません。"
            "余裕の少ない状況（--suite stress）で比較してください。"
        )
    result = ExperimentResult(
        name=spec.name,
        scenario=spec.scenario,
        outcomes=outcomes,
        aborted=aborted,
        failed=failed,
        stats=stats,
        baseline=baseline,
        pairs=pairs,
        warnings=warnings,
        report_md="",
        experiment_dir=exp_dir,
        executed=executed,
    )
    result.report_md = render_report(spec, result)
    (exp_dir / "report.md").write_text(result.report_md, encoding="utf-8")
    return result


def _cell(stat: Stat) -> str:
    return f"{stat.mean:.1f}±{stat.std:.1f}" if stat.std is not None else f"{stat.mean:.1f}"


def render_report(spec: ExperimentSpec, result: ExperimentResult) -> str:
    lines = [
        f"## 実験: {spec.name}（{spec.scenario}）",
        "",
        f"シード {spec.seed} / バリアント {spec.variants} / {spec.hours}時間 / 平均±標準偏差（標本標準偏差）",
        "",
    ]
    for w in result.warnings:
        lines += [f"> ⚠ {w}", ""]
    header = "| 構成 | n | " + " | ".join(label for _, label in REPORT_COLUMNS) + " |"
    lines += [header, "| --- " * (len(REPORT_COLUMNS) + 2) + "|"]
    for cfg, stat in result.stats.items():
        cells = [_cell(stat[key]) for key, _ in REPORT_COLUMNS]
        lines.append(f"| {cfg} | {stat['final_health'].n} | " + " | ".join(cells) + " |")
    if result.baseline and result.pairs:
        lines += [
            "",
            f"### {result.baseline} を基準とした対比較（最終健康度、同じバリアント同士、許容差 {TIE_TOLERANCE}）",
            "",
            "| 構成 | 勝ち | 引き分け | 負け |",
            "| --- | --- | --- | --- |",
        ]
        for name, (wins, ties, losses) in result.pairs.items():
            lines.append(f"| {name} | {wins} | {ties} | {losses} |")
    problems = {
        **{c: f"失敗: {r}" for c, r in result.failed.items()},
        **{c: f"打ち切り: {r}" for c, r in result.aborted.items()},
    }
    if problems:
        lines += ["", "### 失敗・打ち切りの構成", ""]
        lines += [f"- {cfg}: {reason}" for cfg, reason in problems.items()]
    return "\n".join(lines) + "\n"


@dataclass
class SuiteResult:
    name: str
    conditions: dict[str, ExperimentResult]  # 状況名 → その状況の実験結果
    baseline: str | None
    totals: dict[str, tuple[int, int, int]]  # 構成 → 全状況の（勝ち, 引き分け, 負け）合計
    summary_md: str
    suite_dir: Path
    executed: int = 0


def run_suite(
    name: str,
    conditions: Sequence[Condition],
    configs: dict[str, AgentFactory],
    weather_for: Callable[[str], SeriesWeather],
    runs_dir: Path,
    *,
    variants: int = 5,
    seed: int = 0,
    max_runs: int | None = None,
    agent_infos: dict[str, dict[str, str]] | None = None,
    planning_configs: set[str] | None = None,
    condition_configs: set[str] | None = None,
    truncate_irrigation: bool = True,
    clock: WallClock = time.perf_counter,
) -> SuiteResult:
    """状況ごとに実験を実行し、状況ごとに別々に集計する（状況をまたいで平均しない）。"""
    suite_dir = runs_dir / name
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "suite.json").write_text(
        json.dumps(
            {
                "name": name,
                "seed": seed,
                "variants": variants,
                "configs": list(configs),
                "conditions": [
                    {
                        "name": c.name,
                        "scenario": c.scenario,
                        "hours": c.hours,
                        "base": asdict(c.base),
                    }
                    for c in conditions
                ],
                "agent_infos": agent_infos or {},
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    results: dict[str, ExperimentResult] = {}
    executed = 0
    for cond in conditions:
        remaining = None if max_runs is None else max(0, max_runs - executed)
        spec = ExperimentSpec(
            name=f"{name}_{cond.name}",  # 実行IDは {suite}_{状況}_{構成}_v{番号}
            scenario=cond.scenario,
            configs=configs,
            variants=variants,
            seed=seed,
            hours=cond.hours,
            base=cond.base,
            agent_infos=agent_infos or {},
            planning_configs=planning_configs or set(),
            condition_configs=condition_configs or set(),
            truncate_irrigation=truncate_irrigation,
        )
        res = run_experiment(
            spec, weather_for(cond.scenario), runs_dir, max_runs=remaining, clock=clock
        )
        executed += res.executed
        results[cond.name] = res

    baseline = "baseline" if "baseline" in configs else next(iter(configs), None)
    totals: dict[str, tuple[int, int, int]] = {}
    for res in results.values():
        for cfg, (w, t, lo) in res.pairs.items():
            a, b, c = totals.get(cfg, (0, 0, 0))
            totals[cfg] = (a + w, b + t, c + lo)
    summary = render_suite_summary(name, conditions, results, baseline, totals, variants, seed)
    (suite_dir / "summary.md").write_text(summary, encoding="utf-8")
    return SuiteResult(name, results, baseline, totals, summary, suite_dir, executed)


def _stats_table(result: ExperimentResult) -> list[str]:
    header = "| 構成 | n | " + " | ".join(label for _, label in REPORT_COLUMNS) + " |"
    lines = [header, "| --- " * (len(REPORT_COLUMNS) + 2) + "|"]
    for cfg, stat in result.stats.items():
        cells = [_cell(stat[key]) for key, _ in REPORT_COLUMNS]
        lines.append(f"| {cfg} | {stat['final_health'].n} | " + " | ".join(cells) + " |")
    return lines


def render_suite_summary(
    name: str,
    conditions: Sequence[Condition],
    results: dict[str, ExperimentResult],
    baseline: str | None,
    totals: dict[str, tuple[int, int, int]],
    variants: int,
    seed: int,
) -> str:
    lines = [
        f"## スイート: {name}",
        "",
        f"シード {seed} / 各状況 {variants} バリアント / 平均±標準偏差",
        "",
    ]
    for cond in conditions:
        res = results[cond.name]
        lines += [f"### {cond.name}（{cond.scenario}・{cond.hours}時間）", "", cond.description, ""]
        lines += _stats_table(res)
        lines.append("")
    if baseline and totals:
        names = [c.name for c in conditions]
        lines += [
            f"### 全状況の合計（{baseline} を基準、最終健康度、同じバリアント同士、許容差 {TIE_TOLERANCE}）",
            "",
            "| 構成 | " + " | ".join(names) + " | 合計（勝ち/引き分け/負け） |",
            "| --- " * (len(names) + 2) + "|",
        ]
        for cfg, (w, t, lo) in totals.items():
            per = [
                "{}-{}-{}".format(*results[n].pairs[cfg]) if cfg in results[n].pairs else "－"
                for n in names
            ]
            lines.append(f"| {cfg} | " + " | ".join(per) + f" | {w} / {t} / {lo} |")
        lines.append("")
    problems = [
        f"- {cond}/{cfg}: {kind}: {reason}"
        for cond, res in results.items()
        for kind, table in (("失敗", res.failed), ("打ち切り", res.aborted))
        for cfg, reason in table.items()
    ]
    if problems:
        lines += ["### 失敗・打ち切りの構成", "", *problems, ""]
    warnings = [f"- {cond}: {w}" for cond, res in results.items() for w in res.warnings]
    if warnings:
        lines += ["### 警告", "", *warnings, ""]
    return "\n".join(lines) + "\n"
