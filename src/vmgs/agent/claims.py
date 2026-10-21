"""行動要約が「実行した」と述べているツールを判定する。

再確認（graph.verify）と評価指標（eval.metrics）の両方が使うため、依存の少ないここに置く。
"""

# 要約が「行動していない」ことを述べている表現（宣言とはみなさない）
_NEGATIONS = (
    "しない",
    "せず",
    "しなかった",
    "しません",
    "行わない",
    "行わず",
    "試み",
    "不要",
    "必要ない",
    "不可",
    "なし",
    "できな",
    "拒否",
    "見送",
    "控え",
    "中止",
)
_MOVE_WORDS = ("移動", "退避")
_IRRIGATE_WORDS = ("散水", "給水")


def claimed_actions(summary: str) -> set[str]:
    """行動要約が「実行した」と述べているツール名。否定・試行の表現は宣言とみなさない。"""
    if any(n in summary for n in _NEGATIONS):
        return set()
    claims: set[str] = set()
    if any(w in summary for w in _MOVE_WORDS):
        claims.add("move_to_spot")
    if any(w in summary for w in _IRRIGATE_WORDS):
        claims.add("irrigate")
    return claims
