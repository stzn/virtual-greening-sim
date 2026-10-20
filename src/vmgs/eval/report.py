"""複数エージェントの比較レポートと、1回の実行の時系列グラフ。"""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pydantic import BaseModel

from vmgs.agent.protocol import AgentFactory
from vmgs.eval.metrics import Metrics
from vmgs.sim.records import StepRecord
from vmgs.sim.simulate import SimConfig, run_simulation
from vmgs.weather.providers import SeriesWeather

COLUMNS: list[tuple[str, str]] = [
    ("final_health", "最終健康度"),
    ("min_health", "最低健康度"),
    ("heat_mitigation_score", "暑熱緩和"),
    ("move_count", "移動"),
    ("wasted_moves", "無駄な移動"),
    ("rejected_moves", "拒否移動"),
    ("guard_rejections", "ガード拒否"),
    ("irrigation_total_ml", "散水(ml)"),
    ("irrigation_before_rain", "雨前散水"),
    ("final_battery", "電池%"),
    ("final_tank_ml", "タンク(ml)"),
    ("decision_failures", "判断失敗"),
    ("malformed_summaries", "要約崩れ"),
    ("unexecuted_claims", "宣言のみ"),
    ("tool_errors", "ツールエラー"),
]


class ComparisonReport(BaseModel):
    scenario: str
    rows: dict[str, Metrics]

    def to_markdown(self) -> str:
        head = "| エージェント | " + " | ".join(label for _, label in COLUMNS) + " |"
        sep = "| --- " * (len(COLUMNS) + 1) + "|"
        lines = [f"## 比較: {self.scenario}", "", head, sep]
        for name, m in self.rows.items():
            cells = [
                f"{getattr(m, key):.1f}"
                if isinstance(getattr(m, key), float)
                else str(getattr(m, key))
                for key, _ in COLUMNS
            ]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def compare_agents(
    config: SimConfig,
    weather: SeriesWeather,
    factories: dict[str, AgentFactory],
    runs_dir: Path,
    run_prefix: str,
) -> ComparisonReport:
    """同じ気象・同じ初期条件で複数のエージェントを実行し、指標を並べる。"""
    rows: dict[str, Metrics] = {}
    for name, factory in factories.items():
        result = run_simulation(
            config, weather, factory, runs_dir, f"{run_prefix}_{name}", agent_name=name
        )
        if result.metrics is not None:
            rows[name] = result.metrics
    return ComparisonReport(scenario=config.scenario, rows=rows)


def plot_run(steps: Sequence[StepRecord], out_path: Path, title: str) -> None:
    """土壌水分・健康度・バッテリー・タンクの推移と、移動・散水のタイミングを描く。"""
    hours = [0, *(s.step for s in steps)]
    series = {
        "soil moisture %": [
            steps[0].state_before["soil_moisture_pct"],
            *(s.state_after["soil_moisture_pct"] for s in steps),
        ],
        "health": [steps[0].state_before["health"], *(s.state_after["health"] for s in steps)],
        "battery %": [
            steps[0].state_before["battery_pct"],
            *(s.state_after["battery_pct"] for s in steps),
        ],
        "tank (x100 ml)": [
            steps[0].state_before["water_tank_ml"] / 100,
            *(s.state_after["water_tank_ml"] / 100 for s in steps),
        ],
    }
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for label, ys in series.items():
        ax.plot(hours, ys, label=label)
    for s in steps:
        for c in s.tool_calls:
            if c.result.get("ok") and c.name == "move_to_spot":
                ax.axvline(s.step - 0.5, color="tab:purple", alpha=0.35, linestyle="--")
                ax.text(s.step - 0.5, 103, f"->{c.args['spot_id']}", fontsize=8, ha="center")
            elif c.result.get("ok") and c.name == "irrigate":
                ax.axvline(s.step - 0.5, color="tab:blue", alpha=0.25)
    ax.set_xlabel("step (hour)")
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
