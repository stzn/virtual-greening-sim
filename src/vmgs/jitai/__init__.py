"""JITAI（Just-In-Time Adaptive Intervention）の模擬。

4 段階を規則で判断する: 日内変動を補正した過緊張の検知 → 予定の空き枠 → 徒歩圏のポッドの空き →
二段階の PET 安全判定。設計と閾値の根拠は docs/jitai.md。
"""
