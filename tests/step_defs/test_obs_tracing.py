"""LangSmith によるトレース（ステップ定義は実装時に追加）。"""

from pytest_bdd import scenarios

scenarios("obs/tracing.feature")
