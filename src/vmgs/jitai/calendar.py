"""段階 2: 予定の空き枠。"""

from dataclasses import dataclass

from vmgs.jitai.pods import DWELL_MIN, MAX_WALK_MIN

DAY_END_MIN = 19 * 60  # 終業（判定の対象は 8〜19 時）
# 介入に要る空き: ポッドでの滞在 + 徒歩圏の端までの往復
MIN_FREE_MIN = DWELL_MIN + 2 * MAX_WALK_MIN


@dataclass(frozen=True)
class Meeting:
    """予定（会議など）。[start_min, end_min) を分（0 時からの経過）で表す。"""

    start_min: int
    end_min: int


def free_minutes(meetings: tuple[Meeting, ...], t_min: int, day_end_min: int = DAY_END_MIN) -> int:
    """t_min から次の予定（無ければ終業）までの分数。予定の最中なら 0。"""
    end = day_end_min
    for m in meetings:
        if m.start_min <= t_min < m.end_min:
            return 0
        if m.start_min > t_min:
            end = min(end, m.start_min)
    return max(0, end - t_min)
