"""CLI。例: uv run vmgs turn --scenario heatwave --hour 13 --spot C [--dry-run]"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel

from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.llm import MissingApiKeyError, make_chat_model
from vmgs.agent.runner import AgentRunner
from vmgs.config import Settings
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
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
