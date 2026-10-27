"""ローカルのテレメトリとトレースログ。"""

import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pytest_bdd import given, parsers, scenarios, then, when

from tests.step_defs.conftest import Ctx, SimCtx
from vmgs.agent.fake import ScriptedChatModel, tool_call_message
from vmgs.agent.runner import AgentRunner, TurnResult
from vmgs.obs.telemetry import (
    StepTelemetry,
    TelemetrySummary,
    Usage,
    estimate_cost_usd,
    summarize_telemetry,
    trace_record,
    usage_from_messages,
)
from vmgs.sim.store import RunStore

scenarios("obs/telemetry.feature")


class TelCtx:
    def __init__(self) -> None:
        self.usage: tuple[int, int] | None = None
        self.model: ScriptedChatModel | None = None
        self.result: TurnResult | None = None
        self.rows: list[StepTelemetry] = []
        self.summary: TelemetrySummary | None = None
        self.prices: dict[str, tuple[float, float]] = {}
        self.cost: float | None = None
        self.ai: AIMessage | None = None
        self.trace: dict[str, Any] = {}


@pytest.fixture
def tl() -> TelCtx:
    return TelCtx()


def _with_usage(msg: AIMessage, usage: tuple[int, int] | None) -> AIMessage:
    if usage is None:
        return msg
    i, o = usage
    return msg.model_copy(
        update={"usage_metadata": {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}}
    )


# ---------------- トークン数（ターン単位） ----------------
@given(parsers.parse("LLMは各応答で 入力 {i:d} トークン・出力 {o:d} トークンを報告する"))
def _g_usage(tl: TelCtx, i: int, o: int) -> None:
    tl.usage = (i, o)


@given("LLMはトークン数を報告しない")
def _g_no_usage(tl: TelCtx) -> None:
    tl.usage = None


@given("LLMは次の順に応答する")
def _g_script(tl: TelCtx, datatable: list[list[str]]) -> None:
    header, *rows = datatable
    responses: list[AIMessage] = []
    for i, row in enumerate(dict(zip(header, r, strict=True)) for r in rows):
        if row["応答"] == "ツール呼び出し":
            msg = tool_call_message(row["ツール"], json.loads(row["引数"] or "{}"), f"call_{i}")
        else:
            body = json.dumps(
                {"action_summary": row["要約"], "rationale": "理由"}, ensure_ascii=False
            )
            msg = AIMessage(content=body)
        responses.append(_with_usage(msg, tl.usage))
    tl.model = ScriptedChatModel(responses=responses)


@when("エージェントに1ターン判断させる")
def _w_turn(tl: TelCtx, ctx: Ctx) -> None:
    assert tl.model is not None
    tl.result = AgentRunner(tl.model, ctx.world).run_turn()


def _usage(tl: TelCtx) -> Usage:
    assert tl.result is not None
    return usage_from_messages(tl.result.messages)


@then(
    parsers.parse(
        "ターンの使用量は LLM呼び出し {c:d} 回・入力 {i:d}・出力 {o:d}・合計 {t:d} トークンである"
    )
)
def _t_usage(tl: TelCtx, c: int, i: int, o: int, t: int) -> None:
    u = _usage(tl)
    assert (u.llm_calls, u.input_tokens, u.output_tokens, u.total_tokens) == (c, i, o, t)


@then(parsers.parse("ターンの使用量は LLM呼び出し {c:d} 回である"))
def _t_calls(tl: TelCtx, c: int) -> None:
    assert _usage(tl).llm_calls == c


@then("ターンの使用量のトークン数は報告なしである")
def _t_not_reported(tl: TelCtx) -> None:
    u = _usage(tl)
    assert u.input_tokens is None
    assert u.output_tokens is None
    assert u.total_tokens is None


# ---------------- 実行単位のテレメトリ ----------------
@then(parsers.parse("テレメトリの各ステップは LLM呼び出し {n:d} 回である"))
def _t_step_calls(sc: SimCtx, n: int) -> None:
    rows = RunStore(sc.last.run_dir).read_telemetry()
    assert rows
    assert all(r.llm_calls == n for r in rows)


@then(parsers.parse("テレメトリの各ステップのレイテンシは {s:g} 秒である"))
def _t_latency(sc: SimCtx, s: float) -> None:
    rows = RunStore(sc.last.run_dir).read_telemetry()
    assert rows
    assert all(r.latency_s == pytest.approx(s) for r in rows)


@then("steps.jsonl にレイテンシやトークン数は含まれない")
def _t_steps_clean(sc: SimCtx) -> None:
    text = (sc.last.run_dir / "steps.jsonl").read_text(encoding="utf-8")
    assert "latency" not in text
    assert "tokens" not in text


@then("2回のテレメトリのレイテンシは一致しない")
def _t_latency_differs(sc: SimCtx) -> None:
    a, b = ([r.latency_s for r in RunStore(res.run_dir).read_telemetry()] for res in sc.results[:2])
    assert a
    assert a != b


@then(parsers.parse("テレメトリのステップ番号は 1 から {n:d} まで重複なく並ぶ"))
def _t_step_numbers(sc: SimCtx, n: int) -> None:
    rows = RunStore(sc.last.run_dir).read_telemetry()
    assert [r.step for r in rows] == list(range(1, n + 1))


@when(
    parsers.parse(
        "{k:d}時間目で中断した実行に、チェックポイントより先の{j:d}時間目のテレメトリとトレースだけが書かれている"
    )
)
def _w_crash_between_writes(sc: SimCtx, k: int, j: int) -> None:
    """ステップ記録・テレメトリ・トレースを書いた後、チェックポイントの前に落ちた状態を作る。"""
    sc.run(6, stop_after=k)
    store = RunStore(sc.last.run_dir)
    store.append_telemetry(
        StepTelemetry(
            step=j, latency_s=1.0, llm_calls=1, input_tokens=1, output_tokens=1, total_tokens=2
        )
    )
    store.append_trace({"step": j, "time": "stale", "messages": []})


@when(parsers.parse("その実行を {n:d}時間目まで再開する"))
def _w_resume_last(sc: SimCtx, n: int) -> None:
    sc.results.append(sc.resume(sc.last.run_id))


# ---------------- トレースログ ----------------
def _trace(sc: SimCtx, hour: int) -> dict[str, Any]:
    return RunStore(sc.last.run_dir).read_traces()[hour - 1]


@then("1時間目のトレースにシステムプロンプトと最後のLLM応答が含まれる")
def _t_trace_basic(sc: SimCtx) -> None:
    msgs = _trace(sc, 1)["messages"]
    assert msgs[0]["role"] == "system"
    assert "移動緑化ユニットの自律制御AI" in msgs[0]["content"]
    assert msgs[-1]["role"] == "ai"
    assert "要約1" in msgs[-1]["content"]


@then(parsers.parse('{h:d}時間目のトレースの最初の入力に "{a}" と "{b}" が含まれる'))
def _t_trace_memory(sc: SimCtx, h: int, a: str, b: str) -> None:
    first_input = _trace(sc, h)["messages"][1]
    assert first_input["role"] == "human"
    assert a in first_input["content"]
    assert b in first_input["content"]


@then(parsers.parse('{h:d}時間目のトレースの最初の入力には "要約" で始まる直近の判断は含まれない'))
def _t_trace_no_memory(sc: SimCtx, h: int) -> None:
    import re

    assert not re.search(r"要約\d", _trace(sc, h)["messages"][1]["content"])


@then(parsers.parse("{h:d}時間目のトレースのメッセージは空である"))
def _t_trace_empty(sc: SimCtx, h: int) -> None:
    assert _trace(sc, h)["messages"] == []


# ---------------- 思考モードの全文 ----------------
@given(parsers.parse('AIの応答に思考 "{text}" が付いている'))
def _g_ai_with_reasoning(tl: TelCtx, text: str) -> None:
    tl.ai = AIMessage(content="何もしない", additional_kwargs={"reasoning_content": text})


@given("AIの応答に思考は付いていない")
def _g_ai_plain(tl: TelCtx) -> None:
    tl.ai = AIMessage(content="何もしない")


@when("トレースの記録を作る")
def _w_trace(tl: TelCtx) -> None:
    assert tl.ai is not None
    tl.trace = trace_record(1, "2025-08-01T00:00", [SystemMessage("s"), HumanMessage("h"), tl.ai])


def _ai_record(tl: TelCtx) -> dict[str, Any]:
    return next(m for m in tl.trace["messages"] if m["role"] == "ai")


@then(parsers.parse('トレースのAIメッセージに reasoning "{text}" が含まれる'))
def _t_trace_reasoning(tl: TelCtx, text: str) -> None:
    assert _ai_record(tl)["reasoning"] == text


@then("トレースのAIメッセージに reasoning のキーは含まれない")
def _t_trace_no_reasoning(tl: TelCtx) -> None:
    assert "reasoning" not in _ai_record(tl)


# ---------------- 実行の設定 ----------------
@then(parsers.parse("実行の設定に prompt_version={v} と model={m} が記録されている"))
def _t_meta(sc: SimCtx, v: str, m: str) -> None:
    info = RunStore(sc.last.run_dir).read_meta()["agent_info"]
    assert info["prompt_version"] == v
    assert info["model"] == m


# ---------------- 集計 ----------------
@given("テレメトリが次のとおり記録されている")
def _g_rows(tl: TelCtx, datatable: list[list[str]]) -> None:
    header, *rows = datatable
    tl.rows = []
    for i, r in enumerate(dict(zip(header, row, strict=True)) for row in rows):
        inp, out = int(r["入力"]), int(r["出力"])
        tl.rows.append(
            StepTelemetry(
                step=i + 1,
                latency_s=float(r["レイテンシ"]),
                llm_calls=int(r["LLM呼び出し"]),
                input_tokens=inp,
                output_tokens=out,
                total_tokens=inp + out,
            )
        )


@when("テレメトリを要約する")
def _w_summarize(tl: TelCtx) -> None:
    tl.summary = summarize_telemetry(tl.rows)


def _summary(tl: TelCtx) -> TelemetrySummary:
    assert tl.summary is not None
    return tl.summary


@then(parsers.parse("要約の合計レイテンシは {v:g} 秒である"))
def _t_total_latency(tl: TelCtx, v: float) -> None:
    assert _summary(tl).total_latency_s == pytest.approx(v)


@then(parsers.parse("要約の平均レイテンシは {v:g} 秒である"))
def _t_mean_latency(tl: TelCtx, v: float) -> None:
    assert _summary(tl).mean_latency_s == pytest.approx(v)


@then(parsers.parse("要約のp95レイテンシは {v:g} 秒である"))
def _t_p95(tl: TelCtx, v: float) -> None:
    assert _summary(tl).p95_latency_s == pytest.approx(v)


@then(parsers.parse("要約の合計トークンは {n:d} である"))
def _t_total_tokens(tl: TelCtx, n: int) -> None:
    assert _summary(tl).total_tokens == n


@then(parsers.parse("要約のLLM呼び出しは {n:d} 回である"))
def _t_llm_calls(tl: TelCtx, n: int) -> None:
    assert _summary(tl).llm_calls == n


# ---------------- コスト ----------------
@given(
    parsers.parse(
        '単価表に "{m}" が 入力 {a:g} ドル・出力 {b:g} ドル(100万トークンあたり)で登録されている'
    )
)
def _g_price(tl: TelCtx, m: str, a: float, b: float) -> None:
    tl.prices[m] = (float(a), float(b))


@when(parsers.parse('入力 {i:d}・出力 {o:d} トークンのコストを "{m}" で見積もる'))
def _w_cost(tl: TelCtx, i: int, o: int, m: str) -> None:
    tl.cost = estimate_cost_usd(m, i, o, prices=tl.prices or None)


@when(parsers.parse('入力 {i:d}・出力 {o:d} トークンのコストを ollama の "{m}" で見積もる'))
def _w_cost_local(tl: TelCtx, i: int, o: int, m: str) -> None:
    tl.cost = estimate_cost_usd(m, i, o, provider="ollama", prices=tl.prices or None)


@when(parsers.parse('入力 {i:d} トークンのコストを "{m}" で見積もる'))
def _w_cost_input_only(tl: TelCtx, i: int, m: str) -> None:
    tl.cost = estimate_cost_usd(m, i, 0, prices=tl.prices or None)


@then(parsers.parse("コストは {v:g} ドルである"))
def _t_cost(tl: TelCtx, v: float) -> None:
    assert tl.cost == pytest.approx(v)


@then("コストは不明である")
def _t_cost_unknown(tl: TelCtx) -> None:
    assert tl.cost is None
