"""複数回・複数構成の比較実験の共有ステップ定義（conftest から取り込む）。"""

import json
import re
from pathlib import Path
from typing import Any

import pytest
from pytest_bdd import given, parsers, then, when

from tests.step_defs.policies import POLICIES
from tests.step_defs.sim_steps import SimCtx
from vmgs.agent.baseline import BaselineAgent, NoopAgent
from vmgs.agent.protocol import AgentFactory, TurnAgent
from vmgs.agent.runner import TurnResult
from vmgs.eval.metrics import Metrics
from vmgs.experiment.runner import (
    ExperimentResult,
    ExperimentSpec,
    Stat,
    SuiteResult,
    aggregate_metrics,
    compare_paired,
    run_experiment,
    run_suite,
)
from vmgs.experiment.stress import STRESS_SUITE, Condition
from vmgs.experiment.variants import BaseConditions
from vmgs.sim.store import RunStore
from vmgs.weather.cache import load_records
from vmgs.weather.providers import SeriesWeather
from vmgs.world.world import World

DATA = Path(__file__).resolve().parents[2] / "data" / "weather"


def _metrics(final_health: float) -> Metrics:
    return Metrics(
        steps=24,
        final_health=final_health,
        min_health=final_health,
        heat_mitigation_score=0.0,
        move_count=0,
        wasted_moves=0,
        rejected_moves=0,
        guard_rejections=0,
        irrigation_total_ml=0,
        irrigation_before_rain=0,
        final_battery=100.0,
        final_tank_ml=0,
        decision_failures=0,
        malformed_summaries=0,
        unexecuted_claims=0,
        claim_retries=0,
        tool_errors=0,
    )


class ExCtx:
    def __init__(self, sc: SimCtx) -> None:
        self.sc = sc
        self.name = "e1"
        self.configs: list[str] = ["baseline"]
        self.variants = 3
        self.seed = 0
        self.factories: dict[str, AgentFactory] = {}
        self.created = 0
        self.results: list[ExperimentResult] = []
        self.values: dict[str, list[float]] = {}
        self.paired: dict[str, dict[int, float]] = {}
        self.stats: dict[str, dict[str, Stat]] = {}
        self.pair_results: dict[str, tuple[int, int, int]] = {}
        self.condition: Condition | None = None
        self.planning_configs: set[str] = set()
        self.suite_results: list[SuiteResult] = []
        self._suite_mode = False

    def _counting(self, factory: AgentFactory) -> AgentFactory:
        def make(world: World) -> TurnAgent:
            self.created += 1
            return factory(world)

        return make

    def factory_for(self, cfg: str) -> AgentFactory:
        if cfg in self.factories:
            return self._counting(self.factories[cfg])
        if cfg in POLICIES:
            return self._counting(POLICIES[cfg])
        return self._counting(NoopAgent if cfg == "noop" else BaselineAgent)

    def weather_for(self, scenario: str) -> SeriesWeather:
        """状況（ストレス状況・スイート）は気象シナリオを自分で持つ。通常の実験は Background の気象を使う。"""
        if self.condition is not None or self._suite_mode:
            return SeriesWeather(load_records(DATA / f"{scenario}.json"))
        return self.sc.require_weather()

    def spec(self) -> ExperimentSpec:
        cond = self.condition
        return ExperimentSpec(
            name=self.name,
            scenario=cond.scenario if cond else self.sc.scenario,
            configs={c: self.factory_for(c) for c in self.configs},
            variants=self.variants,
            seed=self.seed,
            hours=cond.hours if cond else 24,
            base=cond.base if cond else BaseConditions(),
            planning_configs=set(self.planning_configs),
        )

    def run(self, **kwargs: Any) -> ExperimentResult:
        self.sc.error = None
        try:
            spec = self.spec()
            res = run_experiment(spec, self.weather_for(spec.scenario), self.sc.runs_dir, **kwargs)
        except Exception as e:
            self.sc.error = e
            raise
        self.results.append(res)
        return res

    def run_suite(self, **kwargs: Any) -> SuiteResult:
        self._suite_mode = True
        self.sc.error = None
        try:
            res = run_suite(
                self.name,
                list(STRESS_SUITE.values()),
                {c: self.factory_for(c) for c in self.configs},
                self.weather_for,
                self.sc.runs_dir,
                variants=self.variants,
                seed=self.seed,
                planning_configs=set(self.planning_configs),
                **kwargs,
            )
        except Exception as e:
            self.sc.error = e
            raise
        self.suite_results.append(res)
        return res

    @property
    def last(self) -> ExperimentResult:
        return self.results[-1]

    @property
    def last_suite(self) -> SuiteResult:
        return self.suite_results[-1]


@pytest.fixture
def ex(sc: SimCtx) -> ExCtx:
    return ExCtx(sc)


def _quoted(text: str) -> list[str]:
    return re.findall(r'"([^"]+)"', text)


# ---------------- テスト用の構成 ----------------
def _raising_on_create(msg: str) -> AgentFactory:
    def make(world: World) -> TurnAgent:
        raise RuntimeError(msg)

    return make


def _failing_every_turn(msg: str) -> AgentFactory:
    class Failing(NoopAgent):
        def run_turn(self) -> TurnResult:
            raise RuntimeError(msg)

    return Failing


# ---------------- Given ----------------
@given(parsers.parse('実験 "{name}" の構成は {names} で、バリアントは {n:d} 件である'))
def _g_experiment(ex: ExCtx, name: str, names: str, n: int) -> None:
    ex.name, ex.configs, ex.variants = name, _quoted(names), n


@given(parsers.parse("シードは {s:d} である"))
def _g_seed(ex: ExCtx, s: int) -> None:
    ex.seed = s


@given("エージェントの生成回数を数える")
def _g_count(ex: ExCtx) -> None:
    ex.created = 0  # 生成は常に数えている。ここで0に戻す


@given(parsers.parse('構成 "{c}" は作成時に例外を送出する'))
def _g_broken(ex: ExCtx, c: str) -> None:
    ex.factories[c] = _raising_on_create("cannot create agent")


@given(parsers.parse('構成 "{c}" は毎ターン判断に失敗する'))
def _g_failing(ex: ExCtx, c: str) -> None:
    ex.factories[c] = _failing_every_turn("scripted failure")


@given(parsers.parse('構成 "{c}" は "{msg}" という例外で毎ターン失敗する'))
def _g_failing_msg(ex: ExCtx, c: str, msg: str) -> None:
    ex.factories[c] = _failing_every_turn(msg)


@given(parsers.re(r'^構成 "(?P<c>[^"]+)" の最終健康度が (?P<v>.+) である$'))
def _g_values(ex: ExCtx, c: str, v: str) -> None:
    if "バリアント" in v:
        ex.paired[c] = {int(i): float(x) for i, x in re.findall(r"バリアント(\d+)で ([\d.]+)", v)}
    else:
        ex.values[c] = [float(x) for x in re.findall(r"[\d.]+", v)]


# ---------------- When ----------------
@when("実験を実行する")
def _w_run(ex: ExCtx) -> None:
    ex.run()


@when("同じ実験をもう一度実行する")
def _w_run_again(ex: ExCtx) -> None:
    ex.run()


@when(parsers.parse("実験を {k:d} 件の実行で打ち切る"))
def _w_run_limited(ex: ExCtx, k: int) -> None:
    ex.run(max_runs=k)


@when("構成ごとに集計する")
def _w_aggregate(ex: ExCtx) -> None:
    ex.stats = aggregate_metrics({c: [_metrics(v) for v in vs] for c, vs in ex.values.items()})


@when(parsers.parse('構成 "{b}" を基準に最終健康度で対比較する'))
def _w_pair(ex: ExCtx, b: str) -> None:
    ex.pair_results = {
        c: compare_paired(ex.paired[b], vals) for c, vals in ex.paired.items() if c != b
    }


# ---------------- Then ----------------
@then(parsers.parse("実行は {n:d} 件で、実行IDは次のとおりである"))
def _t_run_ids(ex: ExCtx, datatable: list[list[str]]) -> None:
    expected = sorted(row[0] for row in datatable[1:])
    assert len(expected) == 6
    assert sorted(o.run_id for o in ex.last.outcomes) == expected
    assert all((ex.sc.runs_dir / rid / "run.json").exists() for rid in expected)


def _run_config(ex: ExCtx, cfg: str, variant: int) -> dict[str, Any]:
    meta = RunStore(ex.sc.runs_dir / f"{ex.name}_{cfg}_v{variant}").read_meta()
    config: dict[str, Any] = meta["config"]
    return config


@then(parsers.parse("バリアント {i:d} の実行の初期の土壌水分は基準と同じである"))
def _t_v_same(ex: ExCtx, i: int) -> None:
    assert _run_config(ex, ex.configs[0], i)["initial_moisture"] == BaseConditions().moisture


@then(parsers.parse("バリアント {i:d} の実行の初期の土壌水分は基準と異なる"))
def _t_v_diff(ex: ExCtx, i: int) -> None:
    assert _run_config(ex, ex.configs[0], i)["initial_moisture"] != BaseConditions().moisture


@then(parsers.parse('各バリアントで "{a}" と "{b}" の初期条件は同じである'))
def _t_same_conditions(ex: ExCtx, a: str, b: str) -> None:
    for i in range(ex.variants):
        assert _run_config(ex, a, i) == _run_config(ex, b, i)


@then(
    parsers.parse(
        '構成 "{c}" の最終健康度は 平均 {m:g}・標準偏差 {s:g}・最小 {lo:g}・最大 {hi:g}・件数 {n:d} である'
    )
)
def _t_stat(ex: ExCtx, c: str, m: float, s: float, lo: float, hi: float, n: int) -> None:
    st = ex.stats[c]["final_health"]
    assert st.mean == pytest.approx(m)
    assert st.std == pytest.approx(s)
    assert (st.min, st.max, st.n) == (lo, hi, n)


@then(parsers.parse('構成 "{c}" の最終健康度の標準偏差はなしである'))
def _t_std_none(ex: ExCtx, c: str) -> None:
    assert ex.stats[c]["final_health"].std is None


@then(parsers.parse('構成 "{x}" は 勝ち {w:d}・引き分け {t:d}・負け {lo:d} である'))
def _t_pair_synthetic(ex: ExCtx, x: str, w: int, t: int, lo: int) -> None:
    assert ex.pair_results[x] == (w, t, lo)


@then(parsers.parse('構成 "{x}" は "{b}" に対して 勝ち {w:d}・引き分け {t:d}・負け {lo:d} である'))
def _t_pair_experiment(ex: ExCtx, x: str, b: str, w: int, t: int, lo: int) -> None:
    assert ex.last.baseline == b
    assert ex.last.pairs[x] == (w, t, lo)


@then(parsers.parse("エージェントの生成回数は {n:d} 回である"))
def _t_created(ex: ExCtx, n: int) -> None:
    assert ex.created == n


@then("2回の集計結果は一致する")
def _t_same_stats(ex: ExCtx) -> None:
    a, b = ex.results[0], ex.results[-1]
    assert {c: {k: v.model_dump() for k, v in s.items()} for c, s in a.stats.items()} == {
        c: {k: v.model_dump() for k, v in s.items()} for c, s in b.stats.items()
    }


@then(parsers.parse("実行は {n:d} 件が完了している"))
def _t_completed_total(ex: ExCtx, n: int) -> None:
    assert sum(1 for o in ex.last.outcomes if o.status == "COMPLETED") == n


@then(parsers.parse('構成 "{c}" は {n:d} 件完了している'))
def _t_completed(ex: ExCtx, c: str, n: int) -> None:
    assert len(ex.last.completed(c)) == n


@then(parsers.parse('構成 "{c}" は 失敗として記録され、理由が残る'))
def _t_failed(ex: ExCtx, c: str) -> None:
    assert ex.last.failed[c]
    assert any(o.config == c and o.status == "FAILED" for o in ex.last.outcomes)


@then(parsers.parse('構成 "{c}" は バリアント 0 のあと打ち切られ、残りは実行されない'))
def _t_aborted(ex: ExCtx, c: str) -> None:
    outs = [o for o in ex.last.outcomes if o.config == c]
    assert [(o.variant, o.status) for o in outs] == [(0, "ABORTED")]
    assert not Path(ex.sc.runs_dir / f"{ex.name}_{c}_v1").exists()


@then(parsers.parse('構成 "{c}" の打ち切り理由は "{text}" を含む'))
def _t_abort_reason(ex: ExCtx, c: str, text: str) -> None:
    assert text in ex.last.aborted[c]


@then(parsers.parse('レポートに構成 "{a}" "{b}" の最終健康度の 平均±標準偏差 と 件数 が含まれる'))
def _t_report_stats(ex: ExCtx, a: str, b: str) -> None:
    md = ex.last.report_md
    for cfg in (a, b):
        rows = [line for line in md.splitlines() if line.startswith(f"| {cfg} |")]
        assert rows, cfg
        assert "±" in rows[0]
        assert re.search(rf"\| {cfg} \| \d+ \|", rows[0])


@then(parsers.parse('レポートに "{b}" を基準とした対比較の勝敗が含まれる'))
def _t_report_pairs(ex: ExCtx, b: str) -> None:
    md = ex.last.report_md
    assert f"{b} を基準とした対比較" in md
    assert "勝ち" in md


@then(parsers.parse('レポートに失敗した構成 "{c}" とその理由が含まれる'))
def _t_report_failed(ex: ExCtx, c: str) -> None:
    md = ex.last.report_md
    assert f"- {c}: 失敗" in md
    assert "cannot create agent" in md


@then(parsers.parse('レポートに "{text}" を含む警告がある'))
def _t_warning(ex: ExCtx, text: str) -> None:
    assert any(text in w for w in ex.last.warnings)
    assert text in ex.last.report_md


@then(parsers.parse('レポートに "{text}" を含む警告はない'))
def _t_no_warning(ex: ExCtx, text: str) -> None:
    assert not any(text in w for w in ex.last.warnings)
    assert text not in ex.last.report_md


@then(
    parsers.parse(
        'experiment.json にシード {s:d}・構成 "{a}" "{b}"・バリアント数 {n:d}・シナリオ "{sc}" が記録されている'
    )
)
def _t_experiment_json(ex: ExCtx, s: int, a: str, b: str, n: int, sc: str) -> None:
    data = json.loads((ex.sc.runs_dir / ex.name / "experiment.json").read_text(encoding="utf-8"))
    assert data["seed"] == s
    assert data["configs"] == [a, b]
    assert data["variants"] == n
    assert data["scenario"] == sc


# ================= ストレス状況 =================
@given(parsers.parse('ストレス状況 "{c}" を使う'))
def _g_condition(ex: ExCtx, c: str) -> None:
    ex.condition = STRESS_SUITE[c]


@when(parsers.parse('ストレス状況 "{c}" を取得する'))
def _w_get_condition(ex: ExCtx, c: str) -> None:
    ex.condition = STRESS_SUITE[c]


@then(
    parsers.parse(
        '気象は "{w}" で {h:d} 時間・土壌水分 {m:g}%・水タンク {t:d}ml・スポット "{s}"・{start:d}時開始である'
    )
)
def _t_condition_def(ex: ExCtx, w: str, h: int, m: float, t: int, s: str, start: int) -> None:
    cond = ex.condition
    assert cond is not None
    assert (cond.scenario, cond.hours) == (w, h)
    assert (cond.base.moisture, cond.base.tank_ml, cond.base.spot, cond.base.start_hour) == (
        float(m),
        t,
        s,
        start,
    )


@when(parsers.parse('ストレス状況 "{c}" で {names} をバリアント {n:d} 件で実行する'))
def _w_run_condition(ex: ExCtx, c: str, names: str, n: int) -> None:
    ex.condition = STRESS_SUITE[c]
    ex.configs = _quoted(names)
    ex.variants = n
    ex.name = f"st_{c}"
    ex.run()


def _final_health(ex: ExCtx, cfg: str) -> Stat:
    return ex.last.stats[cfg]["final_health"]


@then(parsers.parse('"{c}" の最終健康度の平均は {v:g} 未満である'))
def _t_mean_below(ex: ExCtx, c: str, v: float) -> None:
    assert _final_health(ex, c).mean < v


@then(parsers.parse('"{c}" の最終健康度の標準偏差は {v:g} 以上である'))
def _t_std_at_least(ex: ExCtx, c: str, v: float) -> None:
    std = _final_health(ex, c).std
    assert std is not None
    assert std >= v


@then(parsers.parse('"{x}" は "{b}" に対して 勝ち {w:d}・引き分け {t:d}・負け {lo:d} である'))
def _t_pair_stress(ex: ExCtx, x: str, b: str, w: int, t: int, lo: int) -> None:
    assert ex.last.baseline == b
    assert ex.last.pairs[x] == (w, t, lo)


@then(parsers.parse('"{x}" の最終健康度の平均は "{b}" より {v:g} 以上高い'))
def _t_higher(ex: ExCtx, x: str, b: str, v: float) -> None:
    assert _final_health(ex, x).mean - _final_health(ex, b).mean >= v


@then(parsers.parse('"{x}" の最終健康度の平均は "{b}" より {v:g} 以上低い'))
def _t_lower(ex: ExCtx, x: str, b: str, v: float) -> None:
    assert _final_health(ex, b).mean - _final_health(ex, x).mean >= v


# ================= スイート =================
@given(parsers.parse('スイート "{name}" の構成は {names} で、バリアントは {n:d} 件である'))
@given(parsers.parse('スイート "{name}" の構成は {names} で、バリアントは {n:d} 件である'))
def _g_suite(ex: ExCtx, name: str, names: str, n: int) -> None:
    ex.name, ex.configs, ex.variants = name, _quoted(names), n


@given(parsers.parse("スイートのシードは {s:d} である"))
def _g_suite_seed(ex: ExCtx, s: int) -> None:
    ex.seed = s


@when("スイートを実行する")
def _w_suite(ex: ExCtx) -> None:
    ex.run_suite()


@when(parsers.parse("スイートを {k:d} 件の実行で打ち切る"))
def _w_suite_limited(ex: ExCtx, k: int) -> None:
    ex.run_suite(max_runs=k)


@then(parsers.parse("実行は {n:d} 件で、実行IDに次が含まれる"))
def _t_suite_ids(ex: ExCtx, datatable: list[list[str]]) -> None:
    ids = {o.run_id for res in ex.last_suite.conditions.values() for o in res.outcomes}
    assert len(ids) == 16
    for row in datatable[1:]:
        assert row[0] in ids, row[0]
        assert (ex.sc.runs_dir / row[0] / "run.json").exists()


@then(parsers.parse("スイートの要約に状況 {names} の表がそれぞれある"))
def _t_suite_tables(ex: ExCtx, names: str) -> None:
    md = ex.last_suite.summary_md
    for n in _quoted(names):
        assert f"### {n}（" in md


@then(parsers.parse('状況 "{c1}" の "{g1}" の最終健康度の平均は 状況 "{c2}" の "{g2}" と異なる'))
def _t_not_averaged(ex: ExCtx, c1: str, g1: str, c2: str, g2: str) -> None:
    res = ex.last_suite.conditions
    assert res[c1].stats[g1]["final_health"].mean != res[c2].stats[g2]["final_health"].mean


@then(
    parsers.parse(
        '全状況の合計で "{x}" は "{b}" に対して 勝ち {w:d}・引き分け {t:d}・負け {lo:d} である'
    )
)
def _t_suite_totals(ex: ExCtx, x: str, b: str, w: int, t: int, lo: int) -> None:
    assert ex.last_suite.baseline == b
    assert ex.last_suite.totals[x] == (w, t, lo)


@then("スイートの要約に全状況の合計の対比較が含まれる")
def _t_suite_summary_totals(ex: ExCtx) -> None:
    assert "全状況の合計" in ex.last_suite.summary_md


@then(parsers.parse("スイートの実行は {n:d} 件が完了している"))
def _t_suite_completed(ex: ExCtx, n: int) -> None:
    done = sum(
        1 for r in ex.last_suite.conditions.values() for o in r.outcomes if o.status == "COMPLETED"
    )
    assert done == n


@then(parsers.parse('全ての状況で構成 "{c}" は {n:d} 件完了している'))
def _t_all_completed(ex: ExCtx, c: str, n: int) -> None:
    assert all(len(r.completed(c)) == n for r in ex.last_suite.conditions.values())


@then(parsers.parse('全ての状況で構成 "{c}" は失敗として記録されている'))
def _t_all_failed(ex: ExCtx, c: str) -> None:
    assert len(ex.last_suite.conditions) == len(STRESS_SUITE)
    assert all(c in r.failed for r in ex.last_suite.conditions.values())


@then(
    parsers.parse(
        'suite.json にシード {s:d}・状況 {k:d} 件・構成 "{a}" "{b}"・バリアント数 {n:d} が記録されている'
    )
)
def _t_suite_json(ex: ExCtx, s: int, k: int, a: str, b: str, n: int) -> None:
    data = json.loads((ex.sc.runs_dir / ex.name / "suite.json").read_text(encoding="utf-8"))
    assert data["seed"] == s
    assert len(data["conditions"]) == k
    assert data["configs"] == [a, b]
    assert data["variants"] == n


# ================= 計画観測（構成ごと） =================
OBS_KEYS = ("hours_remaining", "water_per_remaining_hour_ml", "battery_per_remaining_hour_pct")


@given(parsers.parse('構成 "{c}" は計画観測が有効である'))
def _g_planning(ex: ExCtx, c: str) -> None:
    ex.planning_configs.add(c)


def _run_dirs(ex: ExCtx, cfg: str) -> list[Path]:
    prefix = f"{ex.name}_{cfg}_v"
    dirs = sorted(p for p in ex.sc.runs_dir.iterdir() if p.name.startswith(prefix))
    assert dirs, f"{prefix} の実行が無い"
    return dirs


def _planning_flag(run_dir: Path) -> bool:
    return bool(RunStore(run_dir).read_meta()["config"]["planning_observations"])


def _has_obs(run_dir: Path) -> bool:
    calls = [c for s in RunStore(run_dir).read_steps() for c in s.tool_calls]
    states = [c for c in calls if c.name == "get_plant_state"]
    assert states
    return all(all(k in c.result for k in OBS_KEYS) for c in states)


def _lacks_obs(run_dir: Path) -> bool:
    calls = [c for s in RunStore(run_dir).read_steps() for c in s.tool_calls]
    states = [c for c in calls if c.name == "get_plant_state"]
    assert states
    return not any(k in c.result for c in states for k in OBS_KEYS)


@then(parsers.parse('構成 "{c}" の全ての実行で計画観測が有効である'))
def _t_cfg_enabled(ex: ExCtx, c: str) -> None:
    assert all(_planning_flag(d) for d in _run_dirs(ex, c))


@then(parsers.parse('構成 "{c}" の全ての実行で計画観測は無効である'))
def _t_cfg_disabled(ex: ExCtx, c: str) -> None:
    assert not any(_planning_flag(d) for d in _run_dirs(ex, c))


@then(parsers.parse('構成 "{c}" の全ての実行のツール結果に計画観測が含まれる'))
def _t_cfg_has_obs(ex: ExCtx, c: str) -> None:
    assert all(_has_obs(d) for d in _run_dirs(ex, c))


@then(parsers.parse('構成 "{c}" の全ての実行のツール結果に計画観測は含まれない'))
def _t_cfg_lacks_obs(ex: ExCtx, c: str) -> None:
    assert all(_lacks_obs(d) for d in _run_dirs(ex, c))


def _suite_run_dirs(ex: ExCtx, cfg: str) -> list[Path]:
    dirs = [
        p
        for p in ex.sc.runs_dir.iterdir()
        if p.name.startswith(f"{ex.name}_") and f"_{cfg}_v" in p.name and p.is_dir()
    ]
    assert len(dirs) == len(STRESS_SUITE)
    return dirs


@then(parsers.parse('全ての状況で構成 "{c}" の実行は計画観測が有効である'))
def _t_suite_enabled(ex: ExCtx, c: str) -> None:
    assert all(_planning_flag(d) for d in _suite_run_dirs(ex, c))


@then(parsers.parse('全ての状況で構成 "{c}" の実行は計画観測が無効である'))
def _t_suite_disabled(ex: ExCtx, c: str) -> None:
    assert not any(_planning_flag(d) for d in _suite_run_dirs(ex, c))
