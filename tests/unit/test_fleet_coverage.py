"""徒歩圏で覆える需要を見る配車（coverage）の振る舞い。"""

import pytest

from tests.unit.test_fleet import _spot, _stage, _weather
from vmgs.fleet.comfort import PetTable
from vmgs.fleet.dispatch import Coverage, FleetParams, Policy, place_initial, run_fleet
from vmgs.fleet.stage_world import StageWorld
from vmgs.jitai.pods import WALK_RADIUS_M
from vmgs.jitai.safety import DEST_PET_MAX_C

# 緯度 0.001° ≈ 111 m


def _setup(*spots: dict[str, object], **weather: float):  # type: ignore[no-untyped-def]
    stage = _stage(*spots)
    world = StageWorld.from_stage(stage)
    w = _weather(**weather)
    pets = PetTable(world, w)
    fp = FleetParams()
    cov = Coverage(world, pets, walk_radius_m=fp.walk_radius_m, max_dest_pet_c=fp.max_dest_pet_c)
    return world, w, pets, cov


def test_歩ける距離とPETの閾値はJITAIと同じ値() -> None:
    fp = FleetParams()
    assert fp.walk_radius_m == WALK_RADIUS_M
    assert fp.max_dest_pet_c == DEST_PET_MAX_C


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
    hot = Coverage(world, pets, walk_radius_m=400.0, max_dest_pet_c=covered - 0.1)
    ok = Coverage(world, pets, walk_radius_m=400.0, max_dest_pet_c=covered + 0.1)
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
