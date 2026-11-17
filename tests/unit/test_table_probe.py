"""table_probe（別チェックリストでの一般化検証）の単体テスト。"""

from collections import Counter

from vmgs.eval.table_probe import (
    ROW_ACTIONS,
    State,
    checklist_row,
    render_state,
    sample_states,
    system_prompt,
)


def test_各行の代表状態で行番号が決まる() -> None:
    assert checklist_row(State(30, 50, 2, False, 20)) == 1
    assert checklist_row(State(20, 80, 0, False, 40)) == 2
    assert checklist_row(State(30, 50, 0, True, 20)) == 3
    assert checklist_row(State(10, 50, 1, True, 20)) == 4
    assert checklist_row(State(22, 50, 1, False, 40)) == 5
    # 暑いが扉が開いていて在室あり、湿度低 → どの行にも当てはまらない
    assert checklist_row(State(30, 50, 2, True, 20)) == 0


def test_境界値は以上と未満で分かれる() -> None:
    assert checklist_row(State(28, 50, 1, False, 29)) == 1
    assert checklist_row(State(28, 50, 1, False, 30)) == 0  # 単価 30 は未満ではない
    assert checklist_row(State(27, 69, 1, False, 40)) == 5
    assert checklist_row(State(27, 70, 1, False, 40)) == 2  # 湿度 70 は以上
    assert checklist_row(State(18, 50, 1, False, 40)) == 5
    assert checklist_row(State(17, 50, 1, False, 40)) == 4


def test_状態は行ごとに同数で決定的に作られる() -> None:
    a = sample_states(5, seed=0)
    assert a == sample_states(5, seed=0)
    assert Counter(checklist_row(s) for s in a) == {r: 5 for r in range(6)}


def test_入力モードごとに表示が増える() -> None:
    s = State(30, 50, 2, False, 20)
    assert "checklist_row" not in render_state(s, "cond")
    assert "temp_ge_28" not in render_state(s, "raw")
    assert "`checklist_row`: 1" in render_state(s, "row")
    assert "checklist_row" not in system_prompt("cond")
    assert "checklist_row" in system_prompt("row")


def test_行0は何もしないに対応する() -> None:
    assert ROW_ACTIONS[0] == "idle"
