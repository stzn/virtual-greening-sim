"""`vmgs jitai`: 舞台の stage.json と配車の dispatch.json から、合成のワーカーで JITAI を回す。

例: uv run vmgs jitai --stage contracts/fixtures/stage.sample.json \\
        --dispatch contracts/fixtures/dispatch.sample.json --scenario normal --out runs/jitai.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path("runs/jitai.json")


def add_jitai_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    j = sub.add_parser(
        "jitai", help="合成のワーカーで JITAI（検知→予定→ポッド→PET）を回し、介入の記録を書き出す"
    )
    j.add_argument("--stage", type=Path, required=True, help="舞台の stage.json")
    j.add_argument("--dispatch", type=Path, required=True, help="配車の dispatch.json")
    j.add_argument(
        "--scenario",
        choices=["heatwave", "normal", "rainy"],
        help="PET の計算に使う気象（既定: dispatch.json の weather_scenario）",
    )
    j.add_argument("--workers", type=int, default=30, help="合成のワーカーの人数")
    j.add_argument("--seed", type=int, default=0)
    j.add_argument(
        "--no-correction", action="store_true", help="日内変動を補正せずに検知する（比較用）"
    )
    j.add_argument(
        "--no-recovery", action="store_true", help="介入後の回復を次の判定に反映しない（比較用）"
    )
    j.add_argument(
        "--carryover-min",
        type=float,
        default=None,
        help="帰った後に回復が減る時定数 [分]（既定: 30。根拠のない仮の値）",
    )
    j.add_argument("--out", type=Path, default=DEFAULT_OUT)
    j.add_argument(
        "--summary-out",
        type=Path,
        help="舞台に渡す JITAI の集計（契約 v1 の jitai-summary.json）の出力先",
    )


def add_demand_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    d = sub.add_parser(
        "demand", help="vmgs jitai の需要の集計を stage.json の demand に書き戻した舞台を書き出す"
    )
    d.add_argument(
        "--stage", type=Path, required=True, help="元の舞台の stage.json（上書きしない）"
    )
    d.add_argument("--jitai", type=Path, required=True, help="vmgs jitai の出力")
    d.add_argument("--out", type=Path, required=True, help="書き戻した舞台の出力先")


def cmd_demand(args: argparse.Namespace) -> int:
    from vmgs.fleet.stage_world import load_stage
    from vmgs.jitai.demand import apply_demand

    if args.out.resolve() == args.stage.resolve():
        print("--out は元の舞台と別のファイルにしてください", file=sys.stderr)
        return 1
    stage = load_stage(args.stage)
    jitai_out = json.loads(args.jitai.read_text(encoding="utf-8"))
    try:
        out = apply_demand(stage, jitai_out)
    except ValueError as e:
        print(f"需要を書き戻せません: {e}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out.model_dump_json(indent=1) + "\n", encoding="utf-8")
    cells = sum(len(s.demand) for s in out.spots)
    spots = sum(1 for s in out.spots if s.demand)
    print(f"書き出した: {args.out}（需要 {cells} セル・{spots}/{len(out.spots)} 地点）")
    return 0


def _markdown(summary: dict[str, Any]) -> str:
    rate = summary["intervention_rate"]
    head = (
        f"判定 {summary['decisions']} 件・介入 {summary['interventions']} 件"
        f"（介入率 {rate:.1%}、介入したワーカー {summary['workers_with_intervention']}"
        f"/{summary['workers']} 人）"
    )
    lines = [
        head,
        "",
        "| 段階 | 理由 | 見送り |",
        "| --- | --- | --- |",
    ]
    from vmgs.jitai.engine import REASONS

    for reason, stage in REASONS.items():
        lines.append(f"| {stage} | {reason} | {summary['skipped_by_reason'][reason]} |")
    d = summary["detection"]
    lines += [
        "",
        "| 検知 | 感度 | エピソード外の検知（窓） | うち 13 時以降 |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("corrected", "naive"):
        s = d[name]
        lines.append(
            f"| {name} | {s['detected_episodes']}/{s['episodes']} "
            f"| {s['false_alarm_windows']} | {s['false_alarm_windows_afternoon']} |"
        )
    return "\n".join(lines)


def cmd_jitai(args: argparse.Namespace, weather_dir: Path) -> int:
    # PET の計算（pythermalcomfort / numba）は読み込みが重いので、このコマンドでだけ読み込む
    from vmgs.contracts import Dispatch, dispatch_errors_against_stage
    from vmgs.fleet.cli import _git_version
    from vmgs.fleet.comfort import PetTable
    from vmgs.fleet.stage_world import StageWorld, load_stage
    from vmgs.jitai import detect, pods, safety
    from vmgs.jitai.calendar import MIN_FREE_MIN
    from vmgs.jitai.demand import DEMAND_ZONE_M, demand_zones
    from vmgs.jitai.engine import (
        COOLDOWN_AFTER_INTERVENTION_MIN,
        DEMAND_MIN_WORKERS,
        RETRY_AFTER_SKIP_MIN,
        demand_cells,
        demand_zone_counts,
        run_jitai,
        summarize,
    )
    from vmgs.jitai.recovery import CARRYOVER_MIN, POD_GREEN
    from vmgs.jitai.workers import make_workers
    from vmgs.weather.cache import load_records

    stage = load_stage(args.stage)
    dispatch = Dispatch.model_validate_json(args.dispatch.read_text(encoding="utf-8"))
    errors = dispatch_errors_against_stage(dispatch, stage)
    if errors:
        print("配車が舞台と食い違います:", *errors, sep="\n  ", file=sys.stderr)
        return 1
    scenario = args.scenario or dispatch.weather_scenario
    weather_path = weather_dir / f"{scenario}.json"
    if not weather_path.exists():
        print(f"シナリオが見つかりません: {weather_path}", file=sys.stderr)
        return 1
    if args.workers < 1:
        print("--workers は 1 以上にしてください", file=sys.stderr)
        return 1
    world = StageWorld.from_stage(stage)
    weather = {r.time.hour: r for r in load_records(weather_path)}
    pets = PetTable(world, weather)

    # 判定は 8〜19 時（帰りが 19 時台にかかる）。PET に要る時刻が気象と舞台にそろっているか
    missing = sorted(set(range(8, 20)) - (set(weather) & world.hours))
    if missing:
        print(f"気象か舞台に、判定に要る時刻がありません: {missing}", file=sys.stderr)
        return 1

    workers = make_workers(stage, args.workers, seed=args.seed)
    carryover = CARRYOVER_MIN if args.carryover_min is None else args.carryover_min
    if carryover < 0:
        print("--carryover-min は 0 以上にしてください", file=sys.stderr)
        return 1
    result = run_jitai(
        workers,
        world,
        dispatch,
        pets.pet,
        corrected=not args.no_correction,
        recovery=not args.no_recovery,
        carryover_min=carryover,
    )
    summary = summarize(result)
    home = {w.id: w.home_spot_id for w in workers}
    zone_of = demand_zones(stage, DEMAND_ZONE_M)
    out = {
        "generator": f"{_git_version()} jitai",
        "stage_ref": {"area_id": stage.area.id, "date": stage.date.isoformat()},
        "dispatch_generator": dispatch.generator,
        "weather_scenario": scenario,
        "seed": args.seed,
        "rules": {
            "z_threshold": detect.Z_THRESHOLD,
            "sustain_minutes": detect.SUSTAIN_WINDOWS * 5,
            "circadian_correction": not args.no_correction,
            "min_free_minutes": MIN_FREE_MIN,
            "walk_radius_m": pods.WALK_RADIUS_M,
            "dwell_minutes": pods.DWELL_MIN,
            "pod_seats": pods.POD_SEATS,
            "destination_pet_max_c": safety.DEST_PET_MAX_C,
            "route_pet_max_c": safety.ROUTE_PET_MAX_C,
            "cooldown_after_intervention_minutes": COOLDOWN_AFTER_INTERVENTION_MIN,
            "retry_after_skip_minutes": RETRY_AFTER_SKIP_MIN,
            "recovery": not args.no_recovery,
            "recovery_carryover_minutes": carryover,
            "recovery_pod_green": POD_GREEN,
        },
        "records": [r.to_json() for r in result.records],
        "summary": summary,
        "demand": {
            "note": (
                "過緊張の判定点を、勤務地に近い地点の区画（一辺 zone_m）・時刻ごとに数え、"
                "min_workers 人以上の区画・時刻だけを、区画内の地点に等分した集計値"
            ),
            "min_workers": DEMAND_MIN_WORKERS,
            "zone_m": DEMAND_ZONE_M,
            "zones": [
                {
                    "zone": f"{z['zone'][0]},{z['zone'][1]}",
                    "hour": z["hour"],
                    "count": z["count"],
                    "spots": len(z["spot_ids"]),
                }
                for z in demand_zone_counts(result.records, home, zone_of, DEMAND_MIN_WORKERS)
            ],
            "cells": demand_cells(result.records, home, zone_of, DEMAND_MIN_WORKERS),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"書き出した: {args.out}（ワーカー {args.workers} 人・気象 {scenario}）")
    if args.summary_out is not None:
        from vmgs.jitai.demand import jitai_summary

        js = jitai_summary(stage, result, home, scenario, f"{_git_version()} jitai")
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(js.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"書き出した: {args.summary_out}（区画・時刻 {len(js.zones)} 件）")
    print(_markdown(summary))
    return 0
