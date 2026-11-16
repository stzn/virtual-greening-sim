"""評価の結果ファイル（runs/evals/*.json）を再集計し、失敗の型を表示する。

使い方:
  uv run python scripts/eval_report.py runs/evals/<結果>.json [...]       # 通過数と失敗の型
  uv run python scripts/eval_report.py --diff A.json B.json                # 2 つの結果の差（通過の入れ替わり）
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ACTION_TOOLS = {"irrigate", "move_to_spot"}
# 最終回答の要約で、散水・移動を宣言していると見なす語
DECLARED = re.compile("散水|irrigate|水を与|移動|B へ|Bへ")


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def called_action_tool(trace: dict[str, Any]) -> bool:
    return bool({c["name"] for c in trace["calls"]} & ACTION_TOOLS)


def declared_action(trace: dict[str, Any]) -> bool:
    summary = trace.get("action_summary") or ""
    return bool(DECLARED.search(summary)) and "何もしない" not in summary


def summarize(data: dict[str, Any]) -> str:
    scenarios = data["scenarios"]
    traces = [t for s in scenarios for t in s["traces"]]
    passed = sum(1 for s in scenarios if s["passed"])
    called = sum(1 for t in traces if called_action_tool(t))
    declared = [t for t in traces if declared_action(t)]
    not_run = sum(1 for t in declared if not called_action_tool(t))
    tool_calls = sum(len(t["calls"]) for t in traces)
    lines = [
        (
            f"{data['provider']} {data['model']} {data['prompt_version']}: "
            f"通過 {passed}/{len(scenarios)}（{len(traces)} 試行）"
        ),
        (
            f"  行動ツールを呼んだ試行: {called}、呼ばなかった試行: {len(traces) - called}"
            f"、ツール呼び出し総数: {tool_calls}"
        ),
        f"  行動を宣言した試行: {len(declared)}、うちツール未実行: {not_run}",
        "  失敗したシナリオ（成功回数 / 試行、行動の分布）:",
    ]
    for s in scenarios:
        if s["passed"]:
            continue
        dist = Counter((t.get("action_summary") or "なし")[:12] for t in s["traces"])
        lines.append(f"    {s['successes']}/{s['trials']} {s['name']} {dict(dist)}")
    return "\n".join(lines)


def diff(a: dict[str, Any], b: dict[str, Any]) -> str:
    ra = {s["name"]: s for s in a["scenarios"]}
    rb = {s["name"]: s for s in b["scenarios"]}
    gained = [n for n in rb if n in ra and not ra[n]["passed"] and rb[n]["passed"]]
    lost = [n for n in rb if n in ra and ra[n]["passed"] and not rb[n]["passed"]]
    lines = [f"A→B で通過に変わった {len(gained)} 件、不通過に変わった {len(lost)} 件"]
    lines += [f"  + {n}" for n in gained] + [f"  - {n}" for n in lost]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--diff", action="store_true", help="2 つの結果の差を表示する")
    args = parser.parse_args()
    if args.diff:
        if len(args.files) != 2:
            parser.error("--diff には結果ファイルを 2 つ指定する")
        print(diff(load(args.files[0]), load(args.files[1])))
        return
    for f in args.files:
        print(summarize(load(f)))


if __name__ == "__main__":
    main()
