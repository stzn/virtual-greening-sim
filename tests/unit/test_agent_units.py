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

    assert PROMPT_VERSION == "v17"
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


def test_プロンプトv10はv9の最終回答の形式と例を引き継ぎ_既定はv15() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    assert PROMPTS["v10"] != PROMPTS["v9"]
    assert "例（何もしない場合の最終回答）" in PROMPTS["v10"]
    assert "## 最終回答の形式" in PROMPTS["v10"]
    assert PROMPT_VERSION == "v17"
    assert SYSTEM_PROMPT == PROMPTS["v17"]


def test_プロンプトv11は条件の判定キーを説明し_条件欄がそのキーを指す() -> None:
    from vmgs.agent.prompts import PROMPT_REQUIRES_COND_OBS, PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    v11 = PROMPTS["v11"]
    for key in ("rain_within_3h", "heat_trigger", "moisture_band", "battery_below_15"):
        assert key in v11
        assert key not in PROMPTS["v9"]
    assert "| 1 | `rain_within_3h` が true |" in v11
    assert PROMPT_REQUIRES_COND_OBS == frozenset({"v11", "v12", "v13", "v14", "v15", "v16", "v17"})
    assert PROMPT_VERSION == "v17" and SYSTEM_PROMPT == PROMPTS["v17"]


def test_プロンプトv11は判定を行動の指示でなく事実だと明記しv9の最終回答の形式を引き継ぐ() -> None:
    from vmgs.agent.prompts import PROMPTS

    v11 = PROMPTS["v11"]
    assert "事実の判定で、行動の指示ではない" in v11
    assert "## 最終回答の形式" in v11 and "例（何もしない場合の最終回答）" in v11


def test_構成の条件観測はv11のllmだけ有効() -> None:
    from vmgs.experiment.configs import parse_config_spec

    assert parse_config_spec("llm:ollama:qwen3:14b:v11").condition_observations is True
    assert parse_config_spec("llm:ollama:qwen3:14b:v9").condition_observations is False
    assert parse_config_spec("baseline").condition_observations is False
    assert parse_config_spec("llm:ollama:qwen3:14b:v11+think").condition_observations is True
    assert parse_config_spec("llm:ollama:qwen3:14b:v12").condition_observations is True
    assert parse_config_spec("llm:ollama:qwen3:14b:v13").condition_observations is True
    assert parse_config_spec("llm:ollama:qwen3:14b:v14").condition_observations is True


def test_プロンプトv14は行3の数値条件をキーで指し_他の行はv13のまま() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    v13, v14 = PROMPTS["v13"], PROMPTS["v14"]
    assert v14 != v13
    for key in ("temp_at_least_33", "moisture_50_to_70", "tank_at_least_4000"):
        assert key in v14 and key not in v13
    row3 = next(r for r in v14.splitlines() if r.startswith("| 3 |"))
    assert row3.startswith(
        "| 3 | C にいて、`temp_at_least_33` が true、`moisture_50_to_70` が true、"
        "`tank_at_least_4000` が true、`rain_within_3h` が false |"
    )
    assert "4000ml 以上" not in row3 and "50〜70%" not in row3
    # 他の行（#1・#2・#4・#5）は v13 と同じ
    for n in (1, 2, 4, 5):
        prefix = f"| {n} |"
        assert [r for r in v14.splitlines() if r.startswith(prefix)] == [
            r for r in v13.splitlines() if r.startswith(prefix)
        ]
    assert PROMPT_VERSION == "v17" and SYSTEM_PROMPT == PROMPTS["v17"]


def test_プロンプトv15は雨の禁止を雨だけで水分が足りるかのキーで指し_他はv13のまま() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    v13, v15 = PROMPTS["v13"], PROMPTS["v15"]
    assert v15 != v13
    assert "`rain_covers_irrigation`" in v15 and "rain_covers_irrigation" not in v13
    # 雨の判定は新しいキーに置き換わり、旧キー rain_within_3h は使わない
    assert "rain_within_3h" not in v15
    rule, table = v15.split("## 判断チェックリスト")
    assert "`rain_covers_irrigation` が true の間は irrigate を呼ばない" in rule
    row2 = next(r for r in table.splitlines() if r.startswith("| 2 |"))
    row3 = next(r for r in table.splitlines() if r.startswith("| 3 |"))
    assert "`rain_covers_irrigation` が false" in row2
    assert "`rain_covers_irrigation` が false" in row3
    # キー一覧は旧キーを外し、新しいキーを末尾（in_sunlight の次）に短く置く（先頭の長い説明は qwen3 が行 #1 を誤る）
    keys = [r for r in v15.splitlines() if r.startswith("- `") and "`: " in r]
    assert keys[-1].startswith("- `rain_covers_irrigation`:")
    assert keys[-2].startswith("- `in_sunlight`:")
    assert len(keys[-1]) < 110
    # 他の行（#1・#4・#5）は v13 と同じ
    for n in (1, 4, 5):
        prefix = f"| {n} |"
        assert [r for r in v15.splitlines() if r.startswith(prefix)] == [
            r for r in v13.splitlines() if r.startswith(prefix)
        ]
    assert PROMPT_VERSION == "v17" and SYSTEM_PROMPT == PROMPTS["v17"]


def test_プロンプトv15を選んだ構成は条件観測を有効にする() -> None:
    from vmgs.experiment.configs import parse_config_spec

    assert parse_config_spec("llm:ollama:qwen3:14b:v15").condition_observations is True


def test_プロンプトv13は雨を表から外した禁止ルールにし_日向をキーで指す() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    v13 = PROMPTS["v13"]
    assert v13 != PROMPTS["v12"]
    assert "`in_sunlight`" in v13 and "日向（A か C）で" not in v13
    # 雨は表の行ではなく、表の前の禁止ルール。移動の判断は続ける
    assert "| 1 | `rain_within_3h`" not in v13
    rule, table = v13.split("## 判断チェックリスト")
    assert "`rain_within_3h` が true の間は irrigate を呼ばない" in rule
    assert "| 1 | `in_sunlight` が true、かつ `heat_trigger` が true、" in table
    assert "すでに B にいる" not in table
    assert table.count("`rain_within_3h` が false") == 2  # 散水する行（旧 #3・#4）
    assert "| 5 | `moisture_above_70` が true |" in table and "| 6 |" not in table
    assert "#1 に当てはまらない" not in v13
    assert "#4: 土壌水分55%" in v13 and "#4 に該当（水分55%・健康度100）" in v13
    assert "#5 に該当（水分55%" not in v13
    assert "moisture_band" not in v13
    assert PROMPT_VERSION == "v17" and SYSTEM_PROMPT == PROMPTS["v17"]


def test_プロンプトv12は水分を真偽値のキーで指し_帯を使わない() -> None:
    from vmgs.agent.prompts import PROMPT_VERSION, PROMPTS, SYSTEM_PROMPT

    v12 = PROMPTS["v12"]
    for key in (
        "rain_within_3h",
        "heat_trigger",
        "moisture_below_30",
        "moisture_below_40",
        "moisture_above_70",
        "battery_below_15",
    ):
        assert key in v12
    assert "moisture_band" not in v12
    assert v12 != PROMPTS["v11"] != PROMPTS["v9"]
    assert "| 1 | `rain_within_3h` が true |" in v12
    assert "`moisture_below_40` が true" in v12  # #2
    assert "| 6 | `moisture_above_70` が true |" in v12
    assert "土壌水分 50〜70%" in v12  # #4 は現在値のまま
    assert "事実の判定で、行動の指示ではない" in v12
    assert "## 最終回答の形式" in v12 and "例（何もしない場合の最終回答）" in v12
    assert PROMPT_VERSION == "v17" and SYSTEM_PROMPT == PROMPTS["v17"]


def test_SimConfigの条件観測は既定でオフ() -> None:
    from vmgs.sim.simulate import SimConfig

    assert SimConfig(scenario="normal").condition_observations is False


def test_既定のプロンプトは条件観測を要する版なので_指定なしでも条件観測を有効にする() -> None:
    from vmgs.agent.prompts import prompt_requires_cond_obs

    assert prompt_requires_cond_obs(None) is True  # 既定は v15
    assert prompt_requires_cond_obs("v15") is True
    assert prompt_requires_cond_obs("v9") is False
    assert prompt_requires_cond_obs("free") is False


def test_プロンプトv16は雨の条件を全てtrueで満たす向きのキーにそろえ_他はv15と同じ() -> None:
    from vmgs.agent.prompts import PROMPT_REQUIRES_COND_OBS, PROMPT_VERSION, PROMPTS

    v13, v15, v16 = PROMPTS["v13"], PROMPTS["v15"], PROMPTS["v16"]
    assert v16 != v15 and "v16" in PROMPT_REQUIRES_COND_OBS
    # 旧キー（v15 の rain_covers_irrigation、v13 の rain_within_3h）は使わない
    assert "rain_covers_irrigation" not in v16 and "rain_within_3h" not in v16
    rule, table = v16.split("## 判断チェックリスト")
    assert "`rain_leaves_dry` が false の間は irrigate を呼ばない" in rule
    row2 = next(r for r in table.splitlines() if r.startswith("| 2 |"))
    row3 = next(r for r in table.splitlines() if r.startswith("| 3 |"))
    # 散水行の雨の条件は「が true」（false を条件にすると qwen3 が不成立と誤読する）
    assert "`rain_leaves_dry` が true" in row2 and "`rain_leaves_dry` が false" not in row2
    assert "`rain_leaves_dry` が true" in row3 and "`rain_leaves_dry` が false" not in row3
    # キー一覧は新しいキーを末尾（in_sunlight の次）に短く置く
    keys = [r for r in v16.splitlines() if r.startswith("- `") and "`: " in r]
    assert keys[-1].startswith("- `rain_leaves_dry`:") and keys[-2].startswith("- `in_sunlight`:")
    assert len(keys[-1]) < 110
    # 他の行（#1・#4・#5）は v13 と同じ
    for n in (1, 4, 5):
        prefix = f"| {n} |"
        assert [r for r in v16.splitlines() if r.startswith(prefix)] == [
            r for r in v13.splitlines() if r.startswith(prefix)
        ]
    # 既定は v17
    assert PROMPT_VERSION == "v17"


def test_プロンプトv17は水分キーの使う行を説明に添え_他はv16と同じ() -> None:
    from vmgs.agent.prompts import PROMPT_REQUIRES_COND_OBS, PROMPT_VERSION, PROMPTS

    v16, v17 = PROMPTS["v16"], PROMPTS["v17"]
    assert v17 != v16 and "v17" in PROMPT_REQUIRES_COND_OBS
    keys = {r.split("`")[1]: r for r in v17.splitlines() if r.startswith("- `") and "`: " in r}
    # moisture_below_30 は行 #2・#4、moisture_below_40 は行 #1 でだけ使う（取り違え防止）
    assert "（行 #2・#4 で使う）" in keys["moisture_below_30"]
    assert "（行 #1 でだけ使う）" in keys["moisture_below_40"]
    assert all(len(line) < 110 for line in keys.values())
    # キーの並びと、チェックリストの行は v16 と同じ
    assert list(keys) == [
        r.split("`")[1] for r in v16.splitlines() if r.startswith("- `") and "`: " in r
    ]
    assert v17.split("## 判断チェックリスト")[1] == v16.split("## 判断チェックリスト")[1]
    # 既定は v17
    assert PROMPT_VERSION == "v17"


def test_プロンプトv18は判断メモの手順に_1つでも_なら非該当_を足し_他はv17と同じ() -> None:
    from vmgs.agent.prompts import PROMPT_REQUIRES_COND_OBS, PROMPT_VERSION, PROMPTS

    v17, v18 = PROMPTS["v17"], PROMPTS["v18"]
    assert v18 != v17 and "v18" in PROMPT_REQUIRES_COND_OBS
    # 手順 4 に、× が 1 つでもあれば非該当にする規則がある
    step4 = v18.split("4. 判断メモを書く。")[1].split("5. ")[0]
    assert "× が 1 つでもあればその行は非該当" in step4
    assert "× を書いた行を「該当」と書いてはならない" in step4
    # 差分はその規則だけ（それを除くと v17 と同じ）
    added = "「かつ」で並ぶ条件は、**× が 1 つでもあればその行は非該当**（他が全部 ○ でも該当にしない）。× を書いた行を「該当」と書いてはならない。"
    assert v18.replace(added, "") == v17
    # チェックリストとキー一覧は v17 と同じ
    assert v18.split("## 判断チェックリスト")[1] == v17.split("## 判断チェックリスト")[1]
    # 既定はまだ v17
    assert PROMPT_VERSION == "v17"
