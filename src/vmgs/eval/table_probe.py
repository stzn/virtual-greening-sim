"""別のチェックリスト（倉庫の空調）で、表の読み取りと「世界が答えを渡す」効果が一般化するかを測る。

植物の表とは無関係な合成の表を使い、ツールなしの 1 回の質問で行動を答えさせる。
入力の与え方を 3 通り比べる:
  raw  : 数値の状態だけ
  cond : 状態 + 各条件の真偽（v21 に相当）
  row  : 状態 + 各条件の真偽 + 世界が計算した行番号（v24 に相当）

使い方: uv run python -m vmgs.eval.table_probe --model qwen3:8b --n 120
"""

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass

import httpx

ROW_ACTIONS = {
    1: "cool_strong",
    2: "dehumidify",
    3: "cool_weak",
    4: "heat",
    5: "idle",
    0: "idle",
}

TABLE_PROMPT = """\
あなたは倉庫の空調を制御する。状態を見て、下の判断チェックリストに従い、行動を 1 つ選ぶ。

## 判断チェックリスト（上から順。最初に当てはまる行に従う）
| # | 条件（すべて満たす） | 行動 |
| --- | --- | --- |
| 1 | 室温が 28℃ 以上、かつ 在室人数が 1 以上、かつ 扉が閉じている、かつ 電力単価が 30 円/kWh 未満 | cool_strong |
| 2 | 湿度が 70% 以上、かつ 扉が閉じている | dehumidify |
| 3 | 室温が 28℃ 以上、かつ 在室人数が 0、かつ 電力単価が 30 円/kWh 未満 | cool_weak |
| 4 | 室温が 18℃ 未満、かつ 在室人数が 1 以上 | heat |
| 5 | 室温が 18℃ 以上 28℃ 未満、かつ 湿度が 70% 未満 | idle |
どの行にも当てはまらなければ何もしない（idle）。

行動は cool_strong / dehumidify / cool_weak / heat / idle のいずれか。
次の JSON だけを出力する: {"action": "<行動>"}
"""

KEYS_NOTE = """\

## 条件の真偽（世界が計算済み）
`temp_ge_28` `temp_lt_18` `temp_18_to_28` `humidity_ge_70` `door_closed` `occupied` `empty` `price_lt_30`
"""

ROW_NOTE = """\
`checklist_row`: 世界が上の表を上から順に当てはめ、最初に当てはまった行の番号。その行の行動に従う。どの行にも当てはまらなければ 0 で、何もしない（idle）。
"""


@dataclass(frozen=True)
class State:
    temp: int
    humidity: int
    occupancy: int
    door_open: bool
    price: int


def conditions(s: State) -> dict[str, bool]:
    return {
        "temp_ge_28": s.temp >= 28,
        "temp_lt_18": s.temp < 18,
        "temp_18_to_28": 18 <= s.temp < 28,
        "humidity_ge_70": s.humidity >= 70,
        "door_closed": not s.door_open,
        "occupied": s.occupancy >= 1,
        "empty": s.occupancy == 0,
        "price_lt_30": s.price < 30,
    }


def checklist_row(s: State) -> int:
    c = conditions(s)
    if c["temp_ge_28"] and c["occupied"] and c["door_closed"] and c["price_lt_30"]:
        return 1
    if c["humidity_ge_70"] and c["door_closed"]:
        return 2
    if c["temp_ge_28"] and c["empty"] and c["price_lt_30"]:
        return 3
    if c["temp_lt_18"] and c["occupied"]:
        return 4
    if c["temp_18_to_28"] and not c["humidity_ge_70"]:
        return 5
    return 0


def sample_states(n_per_row: int, seed: int) -> list[State]:
    """行 0〜5 を同数ずつ、境界付近の値を多めに含めて作る（決定的）。"""
    rng = random.Random(seed)
    want = {r: n_per_row for r in range(6)}
    out: list[State] = []
    while any(v > 0 for v in want.values()):
        s = State(
            temp=rng.choice([rng.randint(10, 34), 17, 18, 27, 28]),
            humidity=rng.choice([rng.randint(30, 90), 69, 70]),
            occupancy=rng.choice([0, 0, 1, 2, 5]),
            door_open=rng.random() < 0.4,
            price=rng.choice([rng.randint(10, 50), 29, 30]),
        )
        r = checklist_row(s)
        if want[r] > 0:
            want[r] -= 1
            out.append(s)
    return out


def render_state(s: State, mode: str) -> str:
    lines = [
        "## 現在の状態",
        f"- 室温: {s.temp}℃",
        f"- 湿度: {s.humidity}%",
        f"- 在室人数: {s.occupancy}",
        f"- 扉: {'開いている' if s.door_open else '閉じている'}",
        f"- 電力単価: {s.price} 円/kWh",
    ]
    if mode in ("cond", "row"):
        lines.append("")
        lines.append("## 条件の真偽")
        lines += [f"- `{k}`: {str(v).lower()}" for k, v in conditions(s).items()]
    if mode == "row":
        lines.append(f"- `checklist_row`: {checklist_row(s)}")
    return "\n".join(lines)


def system_prompt(mode: str) -> str:
    text = TABLE_PROMPT
    if mode in ("cond", "row"):
        text += KEYS_NOTE
    if mode == "row":
        text += ROW_NOTE
    return text


def ask(base_url: str, model: str, mode: str, s: State, timeout: float) -> str:
    resp = httpx.post(
        f"{base_url}/api/chat",
        json={
            "model": model,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 0},
            "messages": [
                {"role": "system", "content": system_prompt(mode)},
                {"role": "user", "content": render_state(s, mode)},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        action = json.loads(content).get("action", "")
    except json.JSONDecodeError:
        return "invalid"
    return str(action)


def run(
    base_url: str, model: str, n_per_row: int, seed: int, timeout: float
) -> dict[str, dict[str, object]]:
    states = sample_states(n_per_row, seed)
    result: dict[str, dict[str, object]] = {}
    for mode in ("raw", "cond", "row"):
        correct = 0
        by_row: Counter[int] = Counter()
        for s in states:
            row = checklist_row(s)
            if ask(base_url, model, mode, s, timeout) == ROW_ACTIONS[row]:
                correct += 1
                by_row[row] += 1
        result[mode] = {
            "correct": correct,
            "total": len(states),
            "by_row": {r: by_row[r] for r in range(6)},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=20, help="行 0〜5 それぞれの状態数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    res = run(args.base_url, args.model, args.n, args.seed, args.timeout)
    print(f"{args.model} seed={args.seed} 各行 {args.n} 件（行 0〜5、計 {args.n * 6}）")
    for mode, r in res.items():
        print(f"  {mode:4s}: {r['correct']}/{r['total']}  行別 {r['by_row']}")


if __name__ == "__main__":
    main()
