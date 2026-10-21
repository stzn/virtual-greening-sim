"""ローカルのテレメトリ（トークン・レイテンシ・LLM呼び出し数）とトレースログ。

steps.jsonl とは別ファイルに保存する。レイテンシは実行ごとに変わるため、ステップ記録に入れると
「再開しても結果が一致する」再現性が損なわれる。
"""

import math
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import BaseModel

from vmgs.agent.graph import message_text

# 公開価格の例（100万トークンあたり USD: 入力, 出力）。概算用で、価格は変わるため要確認。
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {"gpt-4o-mini": (0.15, 0.60)}


class Usage(BaseModel):
    """1ターンの使用量。トークン数が None なら、モデルが報告しなかった（0ではない）。"""

    llm_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class StepTelemetry(BaseModel):
    step: int
    latency_s: float
    llm_calls: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


class TelemetrySummary(BaseModel):
    steps: int
    total_latency_s: float
    mean_latency_s: float
    p95_latency_s: float
    total_tokens: int
    llm_calls: int


def usage_from_messages(messages: Sequence[BaseMessage]) -> Usage:
    """ターンのメッセージ列から使用量を合算する。AIMessage 1件 = LLM呼び出し1回。"""
    ais = [m for m in messages if isinstance(m, AIMessage)]
    if not ais:
        return Usage(llm_calls=0, input_tokens=0, output_tokens=0, total_tokens=0)
    reported = [m.usage_metadata for m in ais if m.usage_metadata]
    if not reported:
        return Usage(llm_calls=len(ais))
    inp = sum(u["input_tokens"] for u in reported)
    out = sum(u["output_tokens"] for u in reported)
    return Usage(
        llm_calls=len(ais),
        input_tokens=inp,
        output_tokens=out,
        total_tokens=sum(u["total_tokens"] for u in reported),
    )


def message_record(m: BaseMessage) -> dict[str, Any]:
    rec: dict[str, Any] = {"role": m.type, "content": message_text(m)}
    if isinstance(m, AIMessage):
        if m.tool_calls:
            rec["tool_calls"] = [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
        if m.usage_metadata:
            rec["usage"] = dict(m.usage_metadata)
    elif isinstance(m, ToolMessage):
        rec["tool_call_id"] = m.tool_call_id
        rec["name"] = m.name
    return rec


def trace_record(step: int, time: str, messages: Sequence[BaseMessage]) -> dict[str, Any]:
    """1ステップ分のLLMへの入出力。次ターンへ渡した記憶（直近の判断）もここから確認できる。"""
    return {"step": step, "time": time, "messages": [message_record(m) for m in messages]}


def summarize_telemetry(rows: Sequence[StepTelemetry]) -> TelemetrySummary:
    latencies = sorted(r.latency_s for r in rows)
    n = len(latencies)
    total = sum(latencies)
    p95 = latencies[max(0, math.ceil(0.95 * n) - 1)] if n else 0.0
    return TelemetrySummary(
        steps=n,
        total_latency_s=total,
        mean_latency_s=total / n if n else 0.0,
        p95_latency_s=p95,
        total_tokens=sum(r.total_tokens or 0 for r in rows),
        llm_calls=sum(r.llm_calls for r in rows),
    )


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    *,
    provider: str | None = None,
    prices: dict[str, tuple[float, float]] | None = None,
) -> float | None:
    """コストの概算。ローカル(ollama)は0、単価表に無いモデルは None（0ではなく「不明」）。"""
    if provider == "ollama":
        return 0.0
    table = PRICES_USD_PER_MTOK if prices is None else prices
    price = table.get(model)
    if price is None:
        return None
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]
