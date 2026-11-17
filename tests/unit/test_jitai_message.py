"""JITAI の介入の通知文（vmgs.jitai.message）の振る舞い。

判断は規則で済んでいるので、通知文は記録の事実を文章にするだけ。LLM の言い換えは偽の応答で検査を確かめ、
実際の ollama は呼ばない。
"""

import argparse

import pytest

from vmgs.jitai.calendar import Meeting
from vmgs.jitai.engine import Record
from vmgs.jitai.message import (
    Facts,
    check_message,
    compose,
    facts_for,
    parse_llm,
    template_message,
)

SPOTS = {
    "s1": "大手町 仲通り北の歩道",
    "s2": "丸の内 仲通りの広場",
    "s3": "和田倉 公園沿いの歩道",
}


def _intervened(time_min: int = 14 * 60 + 5, walk_m: float = 300.0, pet: float = 31.2) -> Record:
    return Record(time_min, "w01", "intervened", None, None, "pod-1", "s2", walk_m, pet, 39.0)


def _facts(**kw: object) -> Facts:
    base: dict[str, object] = {
        "time": "14:05",
        "free_min": 35,
        "until": "次の予定",
        "spot_name": SPOTS["s2"],
        "walk_min": 4,
        "dwell_min": 10,
        "pet_destination_c": 31,
    }
    base.update(kw)
    return Facts(**base)  # type: ignore[arg-type]


# --- 事実の取り出し ---


def test_介入の記録から_時刻_空き時間_地点名_徒歩分_休憩時間_体感温度を取り出す() -> None:
    # Given 14:05 に徒歩 300 m の広場のポッドへ介入した記録と、15:00 からの予定
    rec = _intervened()
    meetings = (Meeting(15 * 60, 16 * 60),)
    # When 事実を取り出す
    f = facts_for(rec, meetings, SPOTS)
    # Then 徒歩は 80 m/分で切り上げた 4 分、空きは次の予定まで 55 分、体感温度は整数に丸める
    assert f == _facts(free_min=55)


def test_次の予定が無ければ_空き時間は終業までと書く() -> None:
    f = facts_for(_intervened(), (), SPOTS)
    assert f is not None
    assert f.until == "終業"
    assert f.free_min == 19 * 60 - (14 * 60 + 5)


def test_見送った記録には通知を出さない() -> None:
    rec = Record(600, "w01", "skipped", "pet", "pet_route", "pod-1", "s2", 300.0, 31.0, 43.0)
    assert facts_for(rec, (), SPOTS) is None


# --- テンプレート ---


def test_テンプレートは事実だけで通知文を作る() -> None:
    text = template_message(_facts())
    assert text == (
        "14:05 次の予定まで 35 分あります。徒歩 4 分の丸の内 仲通りの広場にいる緑化ポッドで"
        " 10 分休みませんか。樹冠の下の体感温度は約 31℃ です。"
    )


def test_テンプレートの通知文は検査を通る() -> None:
    f = _facts()
    assert check_message(template_message(f), f, SPOTS.values()) == []


def test_通知文に個人の計測値は入らない() -> None:
    text = template_message(_facts())
    for word in ("HRV", "HF", "心拍", "ストレス値", "z"):
        assert word not in text


# --- LLM の出力の検査 ---


def test_事実と同じ数値と地点名だけの言い換えは通る() -> None:
    f = _facts()
    text = "14:05、次の予定まで35分空いています。徒歩4分の丸の内 仲通りの広場で10分ほど休憩しませんか（木陰は約31℃）。"
    assert check_message(text, f, SPOTS.values()) == []


def test_全角の数字も同じ数値として扱う() -> None:
    f = _facts()
    text = "１４:０５ 徒歩４分の丸の内 仲通りの広場で１０分休みませんか。"
    assert check_message(text, f, SPOTS.values()) == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        # 記録にない数値（徒歩 6 分と読み違えた）
        ("徒歩6分の丸の内 仲通りの広場で10分休みませんか。", "unknown_number:6"),
        # 時刻の分（05）と同じ数値でも、徒歩の分が抜けていれば外れる
        ("徒歩5分の丸の内 仲通りの広場で10分休みませんか。", "missing_walk"),
        # 記録にない時刻（数値はすべて記録にあるが、時と分を取り違えた）
        ("5:14 徒歩4分の丸の内 仲通りの広場で10分休みませんか。", "unknown_time:5:14"),
        # 記録にない地点名
        ("徒歩4分の大手町 仲通り北の歩道で10分休みませんか。", "missing_spot"),
        (
            "徒歩4分の丸の内 仲通りの広場か大手町 仲通り北の歩道で10分休みませんか。",
            "unknown_spot:大手町 仲通り北の歩道",
        ),
        # 必須の事実が抜けた
        ("丸の内 仲通りの広場で10分休みませんか。", "missing_walk"),
        ("徒歩4分の丸の内 仲通りの広場で休みませんか。", "missing_dwell"),
    ],
)
def test_記録にない数値_時刻_地点名や_必須の事実の抜けを見つける(text: str, problem: str) -> None:
    assert problem in check_message(text, _facts(), SPOTS.values())


# --- 言い換えと、テンプレートへの戻し ---


def test_LLMを使わなければテンプレートの通知文を返す() -> None:
    m = compose(_facts(), SPOTS.values())
    assert m.source == "template"
    assert m.text == template_message(_facts())
    assert m.llm_problems == []


def test_検査を通った言い換えはそのまま使う() -> None:
    good = "14:05、次の予定まで35分。徒歩4分の丸の内 仲通りの広場で10分ひと息つきませんか。"
    m = compose(_facts(), SPOTS.values(), paraphrase=lambda prompt: good)
    assert m.source == "llm"
    assert m.text == good


def test_検査に外れた言い換えはテンプレートに戻し_外れた理由を記録する() -> None:
    bad = "徒歩6分の丸の内 仲通りの広場で10分休みませんか。"
    m = compose(_facts(), SPOTS.values(), paraphrase=lambda prompt: bad)
    assert m.source == "template"
    assert m.text == template_message(_facts())
    assert "unknown_number:6" in m.llm_problems
    assert m.llm_text == bad


def test_LLMの呼び出しが失敗したらテンプレートに戻し_その旨を記録する() -> None:
    def boom(prompt: str) -> str:
        raise ConnectionError("ollama に接続できない")

    m = compose(_facts(), SPOTS.values(), paraphrase=boom)
    assert m.source == "template"
    assert m.llm_problems == ["llm_error:ConnectionError"]


def test_LLMの思考の部分は検査の前に取り除く() -> None:
    raw = "<think>徒歩は5分かな</think>\n徒歩4分の丸の内 仲通りの広場で10分休みませんか。"
    m = compose(_facts(), SPOTS.values(), paraphrase=lambda prompt: raw)
    assert m.source == "llm"
    assert m.text == "徒歩4分の丸の内 仲通りの広場で10分休みませんか。"


def test_LLMへの指示には事実だけを渡し_個人の計測値は渡さない() -> None:
    seen: list[str] = []

    def record(prompt: str) -> str:
        seen.append(prompt)
        return ""

    compose(_facts(), SPOTS.values(), paraphrase=record)
    prompt = seen[0]
    for fact in ("14:05", "35", "丸の内 仲通りの広場", "4", "10", "31"):
        assert fact in prompt
    for word in ("HRV", "ln_hf", "心拍"):
        assert word not in prompt


def test_JSONに書き出す形() -> None:
    bad = "徒歩6分の丸の内 仲通りの広場で10分休みませんか。"
    m = compose(_facts(), SPOTS.values(), paraphrase=lambda prompt: bad)
    assert m.to_json() == {
        "text": template_message(_facts()),
        "source": "template",
        "llm_text": bad,
        "llm_problems": ["unknown_number:6", "missing_walk"],
    }


# --- --llm の指定 ---


def test_LLMの指定は_ollama_のモデル名だけを受け付ける() -> None:
    assert parse_llm("ollama:qwen3:8b") == "qwen3:8b"
    for bad in ("qwen3:8b", "openai:gpt-5", "ollama:"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_llm(bad)
