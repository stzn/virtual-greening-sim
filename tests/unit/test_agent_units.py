import pytest

from vmgs.agent.graph import parse_decision
from vmgs.agent.llm import MissingApiKeyError, make_chat_model
from vmgs.agent.prompts import SYSTEM_PROMPT_V1, TurnSummary, build_turn_input
from vmgs.config import Settings


def test_parse_decision_はJSONを構造化する() -> None:
    d = parse_decision('{"action_summary": "Bへ退避", "rationale": "猛暑", "risk_notes": "電池"}')
    assert (d.action_summary, d.rationale, d.risk_notes) == ("Bへ退避", "猛暑", "電池")


def test_parse_decision_はコードフェンス付きJSONも読める() -> None:
    d = parse_decision('```json\n{"action_summary": "維持"}\n```')
    assert d.action_summary == "維持"


@pytest.mark.parametrize("text", ["散水した", "[1, 2]", ""])
def test_parse_decision_はJSONでなければ本文を要約にする(text: str) -> None:
    assert parse_decision(text).action_summary == text.strip()


def test_ターン入力に履歴が入り_なければなしと書く() -> None:
    with_hist = build_turn_input(
        "13:00", "C", "歩道", [TurnSummary(time="12:00", action_summary="散水")]
    )
    assert "12:00 散水" in with_hist
    assert "直近の判断: なし" in build_turn_input("13:00", "C", "歩道", [])


def test_システムプロンプトは役割と最終回答の形式を含む() -> None:
    assert "移動緑化ユニットの自律制御AI" in SYSTEM_PROMPT_V1
    assert "action_summary" in SYSTEM_PROMPT_V1


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_APIキーが無ければ明確なエラー(provider: str) -> None:
    settings = Settings(_env_file=None, llm_provider=provider)  # type: ignore[call-arg, arg-type]
    with pytest.raises(MissingApiKeyError):
        make_chat_model(settings)


def test_ollamaはAPIキー無しでモデルを作れる() -> None:
    settings = Settings(_env_file=None, llm_provider="ollama", llm_model="qwen3:14b")  # type: ignore[call-arg]
    model = make_chat_model(settings)
    assert type(model).__name__ == "ChatOllama"


def test_プロンプトv7は移動予算の読み方を含み_v6から変わっている() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS

    assert PROMPT_VERSION in PROMPTS
    assert "move_budget_remaining" in PROMPTS["v7"]
    assert "move_budget_remaining" not in PROMPTS["v6"]
    assert PROMPTS["v7"].startswith(PROMPTS["v6"].split("## 最終回答の形式")[0].rstrip()[:200])


def test_プロンプトv8は移動予算をチェックリストの条件に含む() -> None:
    from vmgs.agent.prompts import PROMPTS

    row2 = next(line for line in PROMPTS["v8"].splitlines() if line.startswith("| 2 |"))
    assert "move_budget_remaining が 1 以上" in row2
    assert "move_budget_remaining が 1 以上" not in next(
        line for line in PROMPTS["v7"].splitlines() if line.startswith("| 2 |")
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('メモ\n{"action_summary": "維持", "rationale": "r"}', "維持"),
        ('```json\n{"action_summary": "維持"}\n```', "維持"),
        ('{"action_summary": "古い"}\n言い直す\n{"action_summary": "新しい"}', "新しい"),
        ('#2: {気温28 < 33} ×\n{"action_summary": "何もしない"}', "何もしない"),
        ('{"action_summary": "維持", "extra": {"a": 1}}', "維持"),
    ],
)
def test_parse_decision_は本文中の最後のJSONを抽出する(text: str, expected: str) -> None:
    d = parse_decision(text)
    assert d.action_summary == expected
    assert d.raw_fallback is False


@pytest.mark.parametrize("text", ['{"foo": 1}', "#1: 非該当", '{"action_summary": ', "{}"])
def test_parse_decision_はaction_summaryを持つJSONが無ければフォールバック(text: str) -> None:
    assert parse_decision(text).raw_fallback is True


def test_summarize_executed_は成功した行動だけを順に要約する() -> None:
    from vmgs.agent.graph import ToolCallRecord, summarize_executed

    def rec(name: str, args: dict[str, object], result: dict[str, object]) -> ToolCallRecord:
        return ToolCallRecord(name=name, args=args, result=result)

    calls = [
        rec("get_plant_state", {}, {"ok": True}),
        rec("move_to_spot", {"spot_id": "B"}, {"ok": True, "new_spot_id": "B"}),
        rec("irrigate", {"amount_ml": 800}, {"ok": False, "error": {"code": "X"}}),
        rec("irrigate", {"amount_ml": 500}, {"ok": True}),
    ]
    assert summarize_executed(calls) == "Bへ移動、500ml散水"
    assert summarize_executed(calls[:1]) == "何もしない"
    assert summarize_executed([]) == "何もしない"


def test_プロンプトv9は最終回答のJSONを必須にし_メモのあとに付ける形を示す() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS

    assert PROMPT_VERSION == "v9"
    assert "必ず1つ" in PROMPTS["v9"]
    assert "必ず1つ" not in PROMPTS["v8"]
    assert "次のJSONだけを返す" not in PROMPTS["v9"]


def test_プロンプトv10は雨の条件を1時間でも_降水量だけでも該当すると書く() -> None:
    from vmgs.agent.prompts import PROMPTS

    v10 = PROMPTS["v10"]
    assert "どれか1時間でも" in v10
    assert "降水量だけで該当" in v10
    assert "どれか1時間でも" not in PROMPTS["v9"]


def test_プロンプトv10は移動の条件を場所と暑さと水分に分けて含む含まないを書く() -> None:
    from vmgs.agent.prompts import PROMPTS

    v10 = PROMPTS["v10"]
    assert "B にいるなら当てはまらない" in v10
    assert "33 ちょうどを含む" in v10
    assert "日射が 700W/m² 以上なら当てはまる" in v10
    assert "40% 以上なら当てはまらない" in v10


def test_プロンプトv10はバッテリー不足でも雨なら散水しないと書く() -> None:
    from vmgs.agent.prompts import PROMPTS

    assert "散水もしない" in PROMPTS["v10"]
    assert "散水もしない" not in PROMPTS["v9"]


def test_プロンプトv10はv9の最終回答の形式と例を引き継ぎ_既定はv9のまま() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    assert PROMPTS["v10"] != PROMPTS["v9"]
    assert "例（何もしない場合の最終回答）" in PROMPTS["v10"]
    assert "## 最終回答の形式" in PROMPTS["v10"]
    assert PROMPT_VERSION == "v9"
    assert SYSTEM_PROMPT == PROMPTS["v9"]
