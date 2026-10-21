"""初期条件のばらつき（バリアント）。"""

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from vmgs.experiment.variants import BaseConditions, Variant, make_variant

scenarios("experiment/variants.feature")


class VarCtx:
    def __init__(self) -> None:
        self.base = BaseConditions()
        self.variants: list[Variant] = []
        self.pair: list[Variant] = []


@pytest.fixture
def vc() -> VarCtx:
    return VarCtx()


@given(
    parsers.parse(
        '基準の初期条件が 土壌水分 {m:g}%・水タンク {t:d}ml・バッテリー {b:g}%・スポット "{s}" である'
    )
)
def _g_base(vc: VarCtx, m: float, t: int, b: float, s: str) -> None:
    vc.base = BaseConditions(moisture=float(m), tank_ml=t, battery_pct=float(b), spot=s)


@given(
    parsers.parse(
        '基準の初期条件が 土壌水分 {m:g}%・水タンク {t:d}ml・バッテリー {b:g}%・スポット "{s}" で {h:d}時開始である'
    )
)
def _g_base_start(vc: VarCtx, m: float, t: int, b: float, s: str, h: int) -> None:
    vc.base = BaseConditions(
        moisture=float(m), tank_ml=t, battery_pct=float(b), spot=s, start_hour=h
    )


@when(parsers.parse("シード {seed:d} でバリアント {i:d} を作る"))
def _w_one(vc: VarCtx, seed: int, i: int) -> None:
    vc.variants = [make_variant(vc.base, seed, i)]


@when(parsers.parse("シード {seed:d} でバリアント {i:d} を2回作る"))
def _w_twice(vc: VarCtx, seed: int, i: int) -> None:
    vc.pair = [make_variant(vc.base, seed, i), make_variant(vc.base, seed, i)]


@when(parsers.parse("シード {seed:d} でバリアント {a:d} から {b:d} までを作る"))
def _w_range(vc: VarCtx, seed: int, a: int, b: int) -> None:
    vc.variants = [make_variant(vc.base, seed, i) for i in range(a, b + 1)]


@when(parsers.parse("シード {s1:d} とシード {s2:d} でバリアント {i:d} を作る"))
def _w_two_seeds(vc: VarCtx, s1: int, s2: int, i: int) -> None:
    vc.pair = [make_variant(vc.base, s1, i), make_variant(vc.base, s2, i)]


@then("バリアントの初期条件は基準と同じである")
def _t_same_as_base(vc: VarCtx) -> None:
    assert vc.variants[0].conditions() == vc.base


@then("2回のバリアントの初期条件は一致する")
def _t_reproducible(vc: VarCtx) -> None:
    assert vc.pair[0] == vc.pair[1]


@then(parsers.parse("{n:d} 件のバリアントの土壌水分はすべて異なる"))
def _t_distinct(vc: VarCtx, n: int) -> None:
    assert len(vc.variants) == n
    assert len({v.moisture for v in vc.variants}) == n


@then("2つのバリアントの土壌水分は異なる")
def _t_seed_differs(vc: VarCtx) -> None:
    assert vc.pair[0].moisture != vc.pair[1].moisture


@then(parsers.parse("すべてのバリアントの土壌水分は {lo:g}% 以上 {hi:g}% 以下である"))
def _t_moisture_range(vc: VarCtx, lo: float, hi: float) -> None:
    assert all(lo <= v.moisture <= hi for v in vc.variants)


@then(parsers.parse("すべてのバリアントの水タンクは {lo:d}ml 以上 {hi:d}ml 以下である"))
def _t_tank_range(vc: VarCtx, lo: int, hi: int) -> None:
    assert all(lo <= v.tank_ml <= hi for v in vc.variants)


@then(parsers.parse("すべてのバリアントの水タンクは {lo:d}ml 以上である"))
def _t_tank_min(vc: VarCtx, lo: int) -> None:
    assert all(v.tank_ml >= lo for v in vc.variants)


@then(parsers.parse("すべてのバリアントのバッテリーは {lo:g}% 以上 {hi:g}% 以下である"))
def _t_battery_range(vc: VarCtx, lo: float, hi: float) -> None:
    assert all(lo <= v.battery_pct <= hi for v in vc.variants)


@then(parsers.parse('すべてのバリアントの開始スポットは "{s}" である'))
def _t_spot(vc: VarCtx, s: str) -> None:
    assert all(v.spot == s for v in vc.variants)


@then(parsers.parse("すべてのバリアントの開始時刻は {h:d}時である"))
def _t_start_hour(vc: VarCtx, h: int) -> None:
    assert all(v.start_hour == h for v in vc.variants)
