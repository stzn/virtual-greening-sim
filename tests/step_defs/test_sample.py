"""BDD基盤の動作確認用ステップ定義（フェーズ1で本実装に置き換える）。"""

from dataclasses import dataclass

from pytest_bdd import given, parsers, scenarios, then, when

scenarios("world/sample.feature")


@dataclass
class Tank:
    remaining_ml: int


@given(parsers.parse("水タンクの初期残量が {ml:d} ml である"), target_fixture="tank")
def _tank(ml: int) -> Tank:
    return Tank(remaining_ml=ml)


@when(parsers.parse("{ml:d} ml 消費する"))
def _consume(tank: Tank, ml: int) -> None:
    tank.remaining_ml -= ml


@then(parsers.parse("水タンクの残量は {ml:d} ml である"))
def _check(tank: Tank, ml: int) -> None:
    assert tank.remaining_ml == ml
