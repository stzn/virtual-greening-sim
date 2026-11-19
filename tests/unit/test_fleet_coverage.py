"""徒歩圏で覆える需要を見る配車（coverage）の振る舞い。"""

import pytest

from tests.unit.test_fleet import _spot, _stage, _weather
from vmgs.fleet.comfort import PetTable
from vmgs.fleet.dispatch import Coverage, FleetParams, Policy, place_initial, run_fleet
from vmgs.fleet.stage_world import StageWorld
from vmgs.jitai.pods import WALK_RADIUS_M
from vmgs.jitai.safety import DEST_PET_MAX_C, ROUTE_PET_MAX_C

# 緯度 0.001° ≈ 111 m


def _setup(*spots: dict[str, object], **weather: float):  # type: ignore[no-untyped-def]
    stage = _stage(*spots)
    world = StageWorld.from_stage(stage)
    # 既定は日向でも PET が JITAI の閾値に届かない気象（閾値を試すテストは閾値の側を動かす）
    w = _weather(**{"temperature_c": 29.0, "shortwave_w_m2": 500.0, **weather})
    pets = PetTable(world, w)
    fp = FleetParams()
    cov = Coverage(
        world,
        pets,
        walk_radius_m=fp.walk_radius_m,
        max_dest_pet_c=fp.max_dest_pet_c,
        max_route_pet_c=fp.max_route_pet_c,
    )
    return world, w, pets, cov


def test_歩ける距離とPETの閾値はJITAIと同じ値() -> None:
    fp = FleetParams()
    assert fp.walk_radius_m == WALK_RADIUS_M
    assert fp.max_dest_pet_c == DEST_PET_MAX_C
    assert fp.max_route_pet_c == ROUTE_PET_MAX_C


def test_同じワーカーを二重に数えない() -> None:
    # Given 100 m 離れた日向の 2 地点（徒歩圏が重なる）、需要は両方に
    _, _, _, cov = _setup(
        _spot("A", 35.680, 139.760, shade=0.0, capacity=2),
        _spot("B", 35.681, 139.760, shade=0.0, capacity=2),
    )
    one = cov.value(["A"], 12)
    assert one > 0
    # Then 同じ地点に 2 台、隣の地点にもう 1 台置いても、覆う需要は増えない（各ワーカーは 1 台へ行く）
    assert cov.value(["A", "A"], 12) == pytest.approx(one)
    assert cov.value(["A", "B"], 12) == pytest.approx(one)


def test_需要の無い隣の地点でも_徒歩圏の需要を覆えば価値がある() -> None:
    # Given 需要は日影の D だけ（そこに置いても PET はほぼ下がらない）。200 m 先に日向の N
    world, w, pets, cov = _setup(
        _spot("D", 35.680, 139.760, shade=1.0, demand=1.0),
        _spot("N", 35.682, 139.760, shade=0.0, demand=0.0),
        _spot("FAR", 35.688, 139.760, shade=0.0, demand=0.0),  # 約 890 m（徒歩圏の外）
    )
    # Then N に置くと D のワーカーが歩いて来られるので価値があり、徒歩圏の外の FAR には無い
    assert cov.value(["N"], 12) > cov.value(["D"], 12)
    assert cov.value(["FAR"], 12) == 0.0
    # When FAR から始めると / Then coverage は N へ動く。greedy は地点の需要だけを見るので、
    # 日影で PET がほとんど下がらない D そのものへ動く
    cases: list[tuple[Policy, str]] = [("coverage", "N"), ("greedy", "D")]
    for policy, expected in cases:
        r = run_fleet(
            world, pets, w, policy=policy, fleet=FleetParams(pods=1), initial_spots=["FAR"]
        )
        assert r.timelines["P1"][0].spot_id == expected, policy


def test_樹冠下のPETが閾値以上の地点には価値を付けない() -> None:
    world, _, pets, _ = _setup(_spot("SUN", 35.680, 139.760, shade=0.0))
    covered = pets.pet("SUN", 12, covered=True)
    hot = Coverage(
        world, pets, walk_radius_m=400.0, max_dest_pet_c=covered - 0.1, max_route_pet_c=99.0
    )
    ok = Coverage(
        world, pets, walk_radius_m=400.0, max_dest_pet_c=covered + 0.1, max_route_pet_c=99.0
    )
    assert hot.value(["SUN"], 12) == 0.0
    assert ok.value(["SUN"], 12) > 0.0


def test_離れた2か所の需要には_1台ずつ分けて置く() -> None:
    # Given 互いに徒歩圏の外（約 1.1 km）の A（需要 1.0）と B（需要 0.4）、どちらも 2 台まで置ける
    world, w, pets, cov = _setup(
        _spot("A", 35.680, 139.760, shade=0.0, capacity=2, demand=1.0),
        _spot("B", 35.690, 139.760, shade=0.0, capacity=2, demand=0.4),
    )
    hours = sorted(w)
    # Then greedy は 2 台目も A に置く（同じ地点の 2 台目の価値 0.5 > B の 0.4）
    assert sorted(place_initial(world, pets, hours, 2)) == ["A", "A"]
    # coverage は、A の 2 台目は同じワーカーを覆うだけなので B に置く
    assert sorted(place_initial(world, pets, hours, 2, coverage=cov)) == ["A", "B"]


def test_ポッドの地点の日向のPETが道中の閾値以上なら価値を付けない() -> None:
    # JITAI は道中の PET に、ポッドの地点の日向（樹冠の外）の PET も含めて見送るため
    world, _, pets, _ = _setup(_spot("SUN", 35.680, 139.760, shade=0.0))
    sun = pets.pet("SUN", 12, covered=False)
    hot = Coverage(world, pets, walk_radius_m=400.0, max_dest_pet_c=99.0, max_route_pet_c=sun - 0.1)
    ok = Coverage(world, pets, walk_radius_m=400.0, max_dest_pet_c=99.0, max_route_pet_c=sun + 0.1)
    assert hot.value(["SUN"], 12) == 0.0
    assert ok.value(["SUN"], 12) > 0.0


def _count_cov(world: StageWorld, pets: PetTable, **kw: float) -> Coverage:
    fp = FleetParams()
    args: dict[str, float] = {
        "walk_radius_m": fp.walk_radius_m,
        "max_dest_pet_c": fp.max_dest_pet_c,
        "max_route_pet_c": fp.max_route_pet_c,
        **kw,
    }
    return Coverage(world, pets, objective="count", **args)


def test_件数の目的では_勤務地が日向で暑すぎて届けられない需要を数えない() -> None:
    # Given 需要は日向の H だけ。100 m 先に日影の P（ポッドを置く地点）
    world, _, pets, _ = _setup(
        _spot("H", 35.680, 139.760, shade=0.0, demand=1.0),
        _spot("P", 35.681, 139.760, shade=1.0, demand=0.0),
    )
    sun_h = pets.pet("H", 12, covered=False)
    sun_p = pets.pet("P", 12, covered=False)
    assert sun_p < sun_h
    # 道中の閾値を、P の日向の PET と H の日向の PET の間に置く（JITAI は勤務地 H の暑さで見送る）
    limit = (sun_p + sun_h) / 2
    relief = Coverage(world, pets, walk_radius_m=400.0, max_dest_pet_c=99.0, max_route_pet_c=limit)
    count = _count_cov(world, pets, max_dest_pet_c=99.0, max_route_pet_c=limit)
    # Then PET の改善の目的は P に置く価値を数えるが、件数の目的は H のワーカーを届けられないので 0
    assert relief.value(["P"], 12) > 0.0
    assert count.value(["P"], 12) == 0.0


def test_件数の目的では_届けられれば_PETの改善の大きさを問わず同じ価値() -> None:
    # Given 需要は D だけ。D から 100 m の日向の S と、日影の C（どちらも閾値の下）
    world, _, pets, cov = _setup(
        _spot("D", 35.680, 139.760, shade=0.5, demand=1.0),
        _spot("S", 35.681, 139.760, shade=0.0, demand=0.0),
        _spot("C", 35.679, 139.760, shade=1.0, demand=0.0),
    )
    count = _count_cov(world, pets)
    # Then PET の改善の目的は日向の S を高く見るが、件数の目的は S と C を同じに見る
    assert cov.value(["S"], 12) > cov.value(["C"], 12)
    assert count.value(["S"], 12) == pytest.approx(count.value(["C"], 12))
    assert count.value(["S"], 12) > 0.0


def test_件数の目的の1件は_PETの改善の平均で換算する() -> None:
    # 移動の閾値・減点は「需要 × ℃」の単位なので、件数の目的も同じ大きさにそろえる
    world, w, pets, cov = _setup(
        _spot("A", 35.680, 139.760, shade=0.0, demand=1.0),
        _spot("B", 35.690, 139.760, shade=0.5, demand=1.0),
    )
    count = _count_cov(world, pets)
    hours = sorted(set(w) & world.hours)
    reliefs = [cov.relief(s, h) for s in ("A", "B") for h in hours if cov.relief(s, h) > 0]
    assert count.count_unit == pytest.approx(sum(reliefs) / len(reliefs))
    assert count.value(["A"], 12) == pytest.approx(1.0 * count.count_unit)


def test_配車の目的は指定しなければPETの改善_件数も選べる() -> None:
    assert FleetParams().objective == "relief"
    world, w, pets, _ = _setup(
        _spot("A", 35.680, 139.760, shade=0.0, demand=1.0),
        _spot("B", 35.690, 139.760, shade=0.0, demand=1.0),
    )
    r = run_fleet(world, pets, w, policy="coverage", fleet=FleetParams(pods=1, objective="count"))
    assert r.timelines["P1"]
