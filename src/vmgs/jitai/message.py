"""介入の通知文。判断は規則で済んでいるので、記録の事実を文章にするだけ。

既定はテンプレート（LLM なし）。LLM（ローカルの ollama）に言い換えさせるときは、出力を事実と照合し、
記録にない数値・時刻・地点名が入った、必須の事実（地点名・徒歩の分・休憩の分）が抜けた、呼び出しが失敗した
ときはテンプレートに戻して、その旨を記録する。小さなモデルは数値を読み違えやすいので、検査を省かない。
個人の計測値（HRV など）は、事実にも指示にも入れない。
"""

import argparse
import logging
import math
import re
import unicodedata
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from vmgs.jitai.calendar import Meeting, free_minutes
from vmgs.jitai.engine import Record
from vmgs.jitai.pods import DWELL_MIN, walk_minutes

Paraphrase = Callable[[str], str]

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_TIME = re.compile(r"(\d{1,2}):(\d{2})")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Facts:
    """通知文に書いてよい事実のすべて。"""

    time: str  # "HH:MM"
    free_min: int
    until: str  # 空きの終わり（"次の予定" か "終業"）
    spot_name: str
    walk_min: int
    dwell_min: int
    pet_destination_c: int  # 樹冠の下の PET（整数に丸める）

    def allowed_numbers(self) -> set[float]:
        h, m = (int(x) for x in self.time.split(":"))
        values = (h, m, self.free_min, self.walk_min, self.dwell_min, self.pet_destination_c)
        return {float(v) for v in values}


@dataclass(frozen=True)
class Message:
    text: str
    source: str  # "template" か "llm"
    llm_text: str | None = None  # LLM の出力（思考の部分は除く）
    llm_problems: list[str] = field(default_factory=list)  # 検査で外れた理由（戻したとき）

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "llm_text": self.llm_text,
            "llm_problems": list(self.llm_problems),
        }


def facts_for(
    rec: Record, meetings: Sequence[Meeting], spot_names: Mapping[str, str]
) -> Facts | None:
    """介入の記録から事実を取り出す。見送った記録には通知を出さないので None。"""
    if rec.outcome != "intervened":
        return None
    assert rec.spot_id is not None and rec.walk_m is not None
    assert rec.pet_destination_c is not None
    free = free_minutes(tuple(meetings), rec.time_min)
    until = "次の予定" if any(m.start_min == rec.time_min + free for m in meetings) else "終業"
    h, m = divmod(rec.time_min, 60)
    return Facts(
        time=f"{h:02d}:{m:02d}",
        free_min=free,
        until=until,
        spot_name=spot_names[rec.spot_id],
        walk_min=walk_minutes(rec.walk_m),
        dwell_min=DWELL_MIN,
        pet_destination_c=math.floor(rec.pet_destination_c + 0.5),
    )


def template_message(f: Facts) -> str:
    return (
        f"{f.time} {f.until}まで {f.free_min} 分あります。"
        f"徒歩 {f.walk_min} 分の{f.spot_name}にいる緑化ポッドで {f.dwell_min} 分休みませんか。"
        f"樹冠の下の体感温度は約 {f.pet_destination_c}℃ です。"
    )


def check_message(text: str, f: Facts, spot_names: Iterable[str]) -> list[str]:
    """通知文が事実から外れていないか。外れた理由の一覧（空なら合格）。

    漢数字（「十分」など）は数値として読まない（「十分に」と区別できないため）。
    """
    norm = unicodedata.normalize("NFKC", text)
    problems: list[str] = []
    allowed = f.allowed_numbers()
    numbers = [float(x) for x in _NUMBER.findall(norm)]
    for x in numbers:
        if x not in allowed:
            problems.append(f"unknown_number:{x:g}")
    h, m = (int(x) for x in f.time.split(":"))
    for th, tm in _TIME.findall(norm):
        if (int(th), int(tm)) != (h, m):
            problems.append(f"unknown_time:{th}:{tm}")
    own = unicodedata.normalize("NFKC", f.spot_name)
    if own not in norm:
        problems.append("missing_spot")
    rest = norm.replace(own, "")
    for name in sorted(set(spot_names), key=len, reverse=True):
        other = unicodedata.normalize("NFKC", name)
        if other != own and other in rest:
            problems.append(f"unknown_spot:{name}")
            rest = rest.replace(other, "")
    if float(f.walk_min) not in numbers:
        problems.append("missing_walk")
    if float(f.dwell_min) not in numbers:
        problems.append("missing_dwell")
    return problems


def llm_prompt(f: Facts) -> str:
    return "\n".join(
        [
            "次の通知文を、意味を変えずに、働く人に送る自然な日本語の 1〜2 文に言い換えてください。",
            "- 数値・時刻・地点名は事実のとおりに書き、事実にない数値・時刻・地点名を足さない。",
            "- 地点名、徒歩の分、休憩の分は必ず入れる。",
            "- 言い換えた文だけを出力する。",
            "事実:",
            f"- 時刻: {f.time}",
            f"- {f.until}までの空き: {f.free_min} 分",
            f"- 地点名: {f.spot_name}",
            f"- 徒歩: {f.walk_min} 分",
            f"- 休憩: {f.dwell_min} 分",
            f"- 樹冠の下の体感温度: 約 {f.pet_destination_c}℃",
            f"元の文: {template_message(f)}",
        ]
    )


def compose(f: Facts, spot_names: Iterable[str], paraphrase: Paraphrase | None = None) -> Message:
    """通知文を作る。paraphrase（LLM）があれば言い換えさせ、検査に外れたらテンプレートに戻す。"""
    template = template_message(f)
    if paraphrase is None:
        return Message(template, "template")
    try:
        raw = paraphrase(llm_prompt(f))
    except Exception as e:  # 接続・タイムアウトなど。理由を記録してテンプレートに戻す
        logger.exception("通知文の言い換え（LLM）に失敗しました")
        return Message(template, "template", None, [f"llm_error:{type(e).__name__}"])
    text = _THINK.sub("", raw).strip()
    problems = check_message(text, f, spot_names)
    if problems:
        return Message(template, "template", text, problems)
    return Message(text, "llm", text)


def attach_messages(
    records_json: Sequence[MutableMapping[str, Any]],
    records: Sequence[Record],
    meetings: Mapping[str, Sequence[Meeting]],
    spot_names: Mapping[str, str],
    paraphrase: Paraphrase | None = None,
) -> int:
    """書き出す記録に message を付ける（見送りは None）。テンプレートに戻した件数を返す。"""
    fallbacks = 0
    for out, rec in zip(records_json, records, strict=True):
        f = facts_for(rec, meetings[rec.worker_id], spot_names)
        if f is None:
            out["message"] = None
            continue
        msg = compose(f, spot_names.values(), paraphrase)
        fallbacks += paraphrase is not None and msg.source == "template"
        out["message"] = msg.to_json()
    return fallbacks


def parse_llm(spec: str) -> str:
    """--llm の指定（ollama:<model>）からモデル名を取り出す。有料 API は受け付けない。"""
    provider, _, model = spec.partition(":")
    if provider != "ollama" or not model:
        raise argparse.ArgumentTypeError("--llm は ollama:<model> の形で指定してください")
    return model


def ollama_paraphraser(model: str) -> Paraphrase:
    """ローカルの ollama で言い換える関数（agent と同じ make_chat_model を使う。思考はオフ）。"""
    from vmgs.agent.llm import make_chat_model
    from vmgs.config import Settings

    chat = make_chat_model(Settings(llm_provider="ollama", llm_model=model, ollama_reasoning=False))

    def paraphrase(prompt: str) -> str:
        return str(chat.invoke(prompt).content)

    return paraphrase
