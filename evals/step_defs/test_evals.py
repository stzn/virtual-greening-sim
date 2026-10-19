"""LLM判断品質の評価シナリオ（ステップ定義は 2-7 で実装）。実行: uv run pytest evals -m eval"""

from pathlib import Path

import pytest
from pytest_bdd import scenarios

pytestmark = pytest.mark.eval

FEATURES = Path(__file__).resolve().parent.parent / "features"

for name in ("rain", "heat", "resources", "loop_guard"):
    scenarios(str(FEATURES / f"{name}.feature"))
