"""`vmgs loop`: 配車 → JITAI → 需要の書き戻し を、指定の周回だけ繰り返す。

周 0 は入力の舞台のまま配車する（需要が空なら既定の一様の需要）。周 r の JITAI の需要の集計を
舞台に書き戻したものが周 r+1 の舞台になる。各周のファイルと、周ごとの表（summary.md・summary.json）を
書き出す。各段階は `vmgs dispatch`・`vmgs jitai` と同じ関数を呼ぶ（引数の既定も同じ）。

例: uv run vmgs loop --stage contracts/fixtures/stage.sample.json --scenario normal \\
        --workers 300 --rounds 3 --out-dir runs/loop
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def add_loop_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    lp = sub.add_parser("loop", help="配車 → JITAI → 需要の書き戻し を繰り返し、周ごとの表を出す")
    lp.add_argument("--stage", type=Path, required=True, help="舞台の stage.json（上書きしない）")
    lp.add_argument("--scenario", default="normal", choices=["heatwave", "normal", "rainy"])
    lp.add_argument("--workers", type=int, default=300, help="合成のワーカーの人数")
    lp.add_argument("--seed", type=int, default=0)
    lp.add_argument("--pods", type=int, default=3, help="ポッドの台数")
    lp.add_argument(
        "--policy",
        default="greedy",
        choices=["greedy", "coverage"],
        help="配車の方式（coverage: 徒歩圏で覆える需要を見る）",
    )
    lp.add_argument("--rounds", type=int, default=1, help="需要を書き戻して配車し直す回数")
    lp.add_argument("--no-recovery", action="store_true", help="介入後の回復を反映しない")
    lp.add_argument("--carryover-min", type=float, default=None, help="回復の持ち越しの時定数 [分]")
    lp.add_argument("--out-dir", type=Path, required=True)


def _demand_vector(stage_path: Path) -> dict[tuple[str, int], float]:
    """舞台の需要を、合計 1 の分布にする（需要が空なら空）。"""
    data = json.loads(stage_path.read_text(encoding="utf-8"))
    cells = {(s["id"], d["hour"]): float(d["weight"]) for s in data["spots"] for d in s["demand"]}
    total = sum(cells.values())
    return {k: v / total for k, v in cells.items()} if total > 0 else {}


def _l1(a: dict[tuple[str, int], float], b: dict[tuple[str, int], float]) -> float | None:
    if not a or not b:
        return None
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        (
            "| 周 | 需要 | 需要の地点 | 前の周との需要の差（L1） | 判定 | 介入 | 介入率 "
            "| 過緊張の窓 | エピソードの負担 |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        l1 = "—" if r["demand_l1_from_prev"] is None else f"{r['demand_l1_from_prev']:.3f}"
        lines.append(
            f"| {r['round']} | {r['demand']} | {r['demand_spots']} | {l1} | {r['decisions']} "
            f"| {r['interventions']} | {r['intervention_rate']:.1%} | {r['hyperstress_windows']} "
            f"| {r['episode_burden']:.0f} |"
        )
    return "\n".join(lines)


def cmd_loop(args: argparse.Namespace, weather_dir: Path) -> int:
    from vmgs.fleet.cli import cmd_dispatch
    from vmgs.jitai.cli import cmd_demand, cmd_jitai

    if args.rounds < 0:
        print("--rounds は 0 以上にしてください", file=sys.stderr)
        return 1
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stage_path = out / "stage.r0.json"
    stage_path.write_text(args.stage.read_text(encoding="utf-8"), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    prev_demand: dict[tuple[str, int], float] = {}
    for r in range(args.rounds + 1):
        dispatch_path = out / f"dispatch.r{r}.json"
        jitai_path = out / f"jitai.r{r}.json"
        summary_path = out / f"jitai-summary.r{r}.json"
        code = cmd_dispatch(
            argparse.Namespace(
                stage=stage_path,
                scenario=args.scenario,
                out=dispatch_path,
                policy=args.policy,
                pods=args.pods,
                max_move_m=None,
                compare=False,
            ),
            weather_dir,
        )
        if code != 0:
            print(f"周 {r} の配車に失敗しました", file=sys.stderr)
            return code
        code = cmd_jitai(
            argparse.Namespace(
                stage=stage_path,
                dispatch=dispatch_path,
                scenario=args.scenario,
                workers=args.workers,
                seed=args.seed,
                no_correction=False,
                no_recovery=args.no_recovery,
                carryover_min=args.carryover_min,
                out=jitai_path,
                summary_out=summary_path,
                messages=False,
                llm=None,
            ),
            weather_dir,
        )
        if code != 0:
            print(f"周 {r} の JITAI に失敗しました", file=sys.stderr)
            return code
        demand = _demand_vector(stage_path)
        s = json.loads(jitai_path.read_text(encoding="utf-8"))["summary"]
        dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "round": r,
                "demand": "一様（既定）" if "default-demand" in dispatch["generator"] else "集計",
                "demand_spots": len({k[0] for k in demand}),
                "demand_l1_from_prev": _l1(prev_demand, demand),
                "decisions": s["decisions"],
                "interventions": s["interventions"],
                "intervention_rate": s["intervention_rate"],
                "hyperstress_windows": s["recovery"]["hyperstress_windows"],
                "episode_burden": s["recovery"]["episode_burden"],
            }
        )
        prev_demand = demand
        if r == args.rounds:
            break
        next_stage = out / f"stage.r{r + 1}.json"
        code = cmd_demand(argparse.Namespace(stage=stage_path, jitai=jitai_path, out=next_stage))
        if code != 0:
            print(f"周 {r} の需要を書き戻せませんでした", file=sys.stderr)
            return code
        stage_path = next_stage

    meta = {
        "stage": str(args.stage),
        "scenario": args.scenario,
        "workers": args.workers,
        "seed": args.seed,
        "pods": args.pods,
        "policy": args.policy,
        "recovery": not args.no_recovery,
        "carryover_min": args.carryover_min,
        "rounds": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    table = _markdown(rows)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print()
    print(table)
    return 0
