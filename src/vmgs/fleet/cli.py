"""`vmgs dispatch`: stage.json から配車を計算し、dispatch.json（契約 v1）を書き出す。

例: uv run vmgs dispatch --stage contracts/fixtures/stage.sample.json --scenario heatwave \\
        --out runs/dispatch.json --compare
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_OUT = Path("runs/dispatch.json")


def add_dispatch_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    d = sub.add_parser(
        "dispatch", help="舞台の stage.json から複数ポッドを配車し、dispatch.json を書き出す"
    )
    d.add_argument("--stage", type=Path, required=True, help="舞台の stage.json")
    d.add_argument("--scenario", default="heatwave", choices=["heatwave", "normal", "rainy"])
    d.add_argument("--out", type=Path, default=DEFAULT_OUT)
    d.add_argument(
        "--policy",
        default="coverage",
        choices=["greedy", "coverage", "fixed", "none"],
        help="配車の方式（既定 coverage: 徒歩圏で覆える需要を見る）",
    )
    d.add_argument("--pods", type=int, default=3, help="ポッドの台数")
    d.add_argument("--max-move-m", type=float, help="1 時間に動ける距離 [m]（既定: 1000）")
    d.add_argument(
        "--compare",
        action="store_true",
        help="ポッドなし・greedy・fixed・none の比較表（人の快適度と植物の健康）も出す",
    )


def _git_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "vmgs"
    return f"vmgs@{out.stdout.strip()}"


def cmd_dispatch(args: argparse.Namespace, weather_dir: Path) -> int:
    # PET の計算（pythermalcomfort / numba）は読み込みが重いので、このコマンドでだけ読み込む
    from dataclasses import replace

    from vmgs.contracts import dispatch_errors_against_stage
    from vmgs.fleet.comfort import PetTable
    from vmgs.fleet.compare import comparison_markdown, no_pod_summary, summarize
    from vmgs.fleet.dispatch import POLICIES, FleetParams, place_initial, run_fleet, to_dispatch
    from vmgs.fleet.stage_world import StageWorld, load_stage
    from vmgs.weather.cache import load_records

    weather_path = weather_dir / f"{args.scenario}.json"
    if not weather_path.exists():
        print(f"シナリオが見つかりません: {weather_path}", file=sys.stderr)
        return 1
    stage = load_stage(args.stage)
    world = StageWorld.from_stage(stage)
    weather = {r.time.hour: r for r in load_records(weather_path)}
    pets = PetTable(world, weather)
    fleet = FleetParams(pods=args.pods)
    if args.max_move_m is not None:
        fleet = replace(fleet, max_move_m=args.max_move_m)
    hours = sorted(set(weather) & world.hours)
    try:
        if not hours:
            raise ValueError("気象と舞台に共通の時刻がありません")
        # 方式を比べるときは同じ初期配置（greedy の置き方）から始める。1 つの方式だけなら、
        # その方式の置き方で始める（coverage は徒歩圏で覆える需要で置く）
        starts = (
            place_initial(world, pets, hours, fleet.pods, decay=fleet.pod_value_decay)
            if args.compare
            else None
        )
        policies = POLICIES if args.compare else (args.policy,)
        results = {
            p: run_fleet(world, pets, weather, policy=p, fleet=fleet, initial_spots=starts)
            for p in policies
        }
    except ValueError as e:
        print(f"配車できません: {e}", file=sys.stderr)
        return 1

    result = results[args.policy]
    generator = f"{_git_version()} {args.policy}"
    if world.uses_default_demand:
        generator += " default-demand"  # 舞台に需要が無く、既定の需要（8〜19 時に一様）を使った
    dispatch = to_dispatch(result, stage, generator=generator, scenario=args.scenario)
    errors = dispatch_errors_against_stage(dispatch, stage)
    if errors:
        print("配車が舞台と食い違います:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(dispatch.model_dump(mode="json"), ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(
        f"書き出した: {args.out}（{len(stage.spots)} 地点・ポッド {fleet.pods} 台・"
        f"{args.policy}・移動 {result.moves} 回）"
    )
    if world.uses_default_demand:
        print("注意: 舞台に需要が無いため、8〜19 時に全地点で一様の需要を仮定した")
    if args.compare:
        rows = [no_pod_summary(world, pets, hours)]
        rows += [summarize(results[p], world, pets) for p in POLICIES]
        print()
        print(comparison_markdown(rows))
    return 0
