"""stage.json から作る世界と、複数ポッドの配車（vmgs.fleet）。"""

import datetime as dt
from pathlib import Path

import pytest

from vmgs.contracts import PodStep, Stage, dispatch_errors_against_stage
from vmgs.fleet.comfort import PetTable
from vmgs.fleet.compare import summarize
from vmgs.fleet.dispatch import POLICIES, FleetParams, run_fleet, to_dispatch
from vmgs.fleet.stage_world import (
    DEFAULT_DEMAND_HOURS,
    KIND_DEFAULTS,
    StageWorld,
    haversine_m,
    load_stage,
)
from vmgs.weather.cache import load_records
from vmgs.weather.models import WeatherRecord

SAMPLE = Path("contracts/fixtures/stage.sample.json")
HEATWAVE = Path("data/weather/heatwave.json")
DAY = dt.date(2025, 8, 30)


def _weather(**kw: float) -> dict[int, WeatherRecord]:
    """毎時同じ気象（日中の猛暑）。"""
    base: dict[str, float] = {
        "temperature_c": 35.0,
        "humidity_pct": 50.0,
        "shortwave_w_m2": 800.0,
        "wind_ms": 2.0,
        "precipitation_mm": 0.0,
        "precipitation_prob_pct": 5.0,
    }
    base.update(kw)
    return {h: WeatherRecord(time=dt.datetime(2025, 8, 30, h), **base) for h in range(10, 16)}


def _spot(
    sid: str,
    lat: float,
    lon: float,
    *,
    shade: float,
    kind: str = "sidewalk",
    capacity: int = 1,
    demand: float = 1.0,
) -> dict[str, object]:
    return {
        "id": sid,
        "name": sid,
        "lat": lat,
        "lon": lon,
        "kind": kind,
        "capacity_pods": capacity,
        "sky_view_factor": 0.5,
        "green": {
            "radius_m": 100,
            "trees_all": 0,
            "trees_tall": 0,
            "parks_count": 0,
            "parks_area_m2": 0,
            "nearest_park_m": None,
        },
        "hourly": [{"hour": h, "shade_fraction": shade} for h in range(24)],
        "demand": [{"hour": h, "weight": demand} for h in range(24)] if demand else [],
    }


def _stage(*spots: dict[str, object]) -> Stage:
    return Stage.model_validate(
        {
            "schema_version": "1",
            "area": {"id": "t", "name": "t", "bbox": [139.7, 35.6, 139.8, 35.7]},
            "date": "2025-08-30",
            "timezone": "Asia/Tokyo",
            "generator": "test",
            "spots": list(spots),
        }
    )


def _run(stage: Stage, weather: dict[int, WeatherRecord], **kw: object) -> tuple:  # type: ignore[type-arg]
    world = StageWorld.from_stage(stage)
    pets = PetTable(world, weather)
    policy = kw.pop("policy", "greedy")
    fp = FleetParams(**kw)  # type: ignore[arg-type]
    return world, pets, run_fleet(world, pets, weather, policy=policy, fleet=fp)  # type: ignore[arg-type]


# --- 世界 ---
def test_haversine_matches_latitude_degree() -> None:
    # 緯度 0.006° はおよそ 667 m
    assert haversine_m(35.6872, 139.7652, 35.6812, 139.7652) == pytest.approx(667, abs=3)


def test_spot_from_stage_uses_shade_and_kind() -> None:
    world = StageWorld.from_stage(load_stage(SAMPLE))
    s = world.spot("OTM-001", 10)
    assert s.solar_factor == pytest.approx(1.0 - 0.3)
    assert s.surface_offset_c == KIND_DEFAULTS["sidewalk"].surface_offset_c
    assert KIND_DEFAULTS["sidewalk"].surface_offset_c > KIND_DEFAULTS["park_edge"].surface_offset_c
    assert world.distance_m("OTM-001", "OTM-002") == pytest.approx(667, abs=3)
    assert world.capacity("OTM-002") == 3


def test_demand_reads_stage_and_zero_when_missing() -> None:
    world = StageWorld.from_stage(load_stage(SAMPLE))
    assert world.demand("OTM-001", 12) == 1.0
    assert world.demand("OTM-001", 13) == 0.0
    assert not world.uses_default_demand


def test_default_demand_when_stage_has_none() -> None:
    world = StageWorld.from_stage(_stage(_spot("X", 35.68, 139.76, shade=0.0, demand=0)))
    assert world.uses_default_demand
    assert world.demand("X", DEFAULT_DEMAND_HOURS[0]) == 1.0
    assert world.demand("X", 3) == 0.0


# --- PET ---
def test_pod_lowers_pet_more_in_sun_than_in_shade() -> None:
    stage = _stage(
        _spot("SUN", 35.680, 139.760, shade=0.0), _spot("SHADE", 35.681, 139.760, shade=1.0)
    )
    pets = PetTable(StageWorld.from_stage(stage), _weather())
    assert pets.pet("SUN", 12, covered=True) < pets.pet("SUN", 12, covered=False)
    assert pets.relief("SUN", 12) > pets.relief("SHADE", 12) >= 0.0


# --- 配車 ---
def test_greedy_moves_pod_to_sunny_demand_spot() -> None:
    # 需要は同じ。日影の地点に置いたポッドは、PET の改善が大きい日向の地点へ移る
    stage = _stage(
        _spot("SHADE", 35.680, 139.760, shade=1.0, demand=1.0),
        _spot("SUN", 35.682, 139.760, shade=0.0, demand=1.0),
    )
    world = StageWorld.from_stage(stage)
    weather = _weather()
    result = run_fleet(
        world,
        PetTable(world, weather),
        weather,
        policy="greedy",
        fleet=FleetParams(pods=1),
        initial_spots=["SHADE"],
    )
    first = result.timelines["P1"][0]
    assert (first.action, first.spot_id) == ("move", "SUN")


def test_greedy_respects_capacity_and_move_limit() -> None:
    stage = _stage(
        _spot("NEAR", 35.680, 139.760, shade=1.0, demand=0.1, capacity=2),
        _spot("SUN", 35.681, 139.760, shade=0.0, demand=1.0, capacity=1),
        _spot("FAR", 35.700, 139.760, shade=0.0, demand=5.0, capacity=3),  # 約 2.2 km
    )
    world = StageWorld.from_stage(stage)
    weather = _weather()
    pets = PetTable(world, weather)
    result = run_fleet(
        world,
        pets,
        weather,
        policy="greedy",
        fleet=FleetParams(pods=3, max_move_m=500.0),
        initial_spots=["NEAR", "NEAR", "SUN"],
    )
    for steps in result.timelines.values():
        assert all(s.spot_id != "FAR" for s in steps)
    for h in weather:
        on_sun = sum(1 for steps in result.timelines.values() if steps_at(steps, h) == "SUN")
        assert on_sun <= 1


def steps_at(steps: list[PodStep], hour: int) -> str:
    return next(s.spot_id for s in steps if s.hour == hour)


def test_dry_pod_irrigates_and_stays() -> None:
    stage = _stage(_spot("A", 35.680, 139.760, shade=0.0), _spot("B", 35.681, 139.760, shade=0.0))
    _, _, result = _run(stage, _weather(), pods=1, initial_moisture=20.0)
    first = result.timelines["P1"][0]
    assert first.action == "irrigate"


def test_hot_sun_and_low_moisture_retreats_to_shade() -> None:
    # 日向が猛暑で水分 35%（散水の閾値 30% より上）: 既存の規則と同じく日影へ退避する
    stage = _stage(
        _spot("SUN", 35.680, 139.760, shade=0.0, demand=5.0),
        _spot("SHADE", 35.681, 139.760, shade=1.0, demand=0.0),
    )
    world = StageWorld.from_stage(stage)
    weather = _weather()
    result = run_fleet(
        world,
        PetTable(world, weather),
        weather,
        policy="greedy",
        fleet=FleetParams(pods=1, initial_moisture=35.0),
        initial_spots=["SUN"],
    )
    first = result.timelines["P1"][0]
    assert (first.action, first.spot_id) == ("move", "SHADE")


def test_fixed_never_moves_and_none_never_acts() -> None:
    stage = _stage(_spot("A", 35.680, 139.760, shade=0.0), _spot("B", 35.681, 139.760, shade=1.0))
    for policy, allowed in (("fixed", {"stay", "irrigate"}), ("none", {"stay"})):
        _, _, result = _run(stage, _weather(), pods=1, initial_moisture=20.0, policy=policy)
        steps = result.timelines["P1"]
        assert {s.action for s in steps} <= allowed
        assert len({s.spot_id for s in steps}) == 1
    _, _, none = _run(stage, _weather(), pods=1, initial_moisture=20.0, policy="none")
    assert all(s.action == "stay" for s in none.timelines["P1"])


def test_sample_dispatch_fits_contract_and_keeps_plants_healthier_than_none() -> None:
    stage = load_stage(SAMPLE)
    world = StageWorld.from_stage(stage)
    weather = {r.time.hour: r for r in load_records(HEATWAVE)}
    pets = PetTable(world, weather)
    results = {
        p: run_fleet(world, pets, weather, policy=p, fleet=FleetParams(pods=3)) for p in POLICIES
    }
    dispatch = to_dispatch(results["greedy"], stage, generator="test", scenario="heatwave")
    assert dispatch_errors_against_stage(dispatch, stage) == []
    assert all(len(p.timeline) == 24 for p in dispatch.pods)
    assert len(dispatch.comfort) == 3 * 24
    # 貪欲法と固定配置の快適度の差は、植物の健康の損失との重みづけで決まる（docs/dispatch.md）。
    # ここでは植物の世話の効果だけを見る
    greedy = summarize(results["greedy"], world, pets)
    none = summarize(results["none"], world, pets)
    assert greedy.min_health is not None and none.min_health is not None
    assert greedy.min_health >= none.min_health


def test_pod_in_hot_sun_irrigates_before_it_must_retreat() -> None:
    # 日向・猛暑で水分 45%（退避の閾値 40% の手前）・タンクに余裕 → 退避せず、先に散水してその場に残る
    stage = _stage(
        _spot("SUN", 35.680, 139.760, shade=0.0, demand=5.0),
        _spot("SHADE", 35.681, 139.760, shade=1.0, demand=0.0),
    )
    world = StageWorld.from_stage(stage)
    weather = _weather()
    for policy in ("greedy", "fixed"):
        result = run_fleet(
            world,
            PetTable(world, weather),
            weather,
            policy=policy,
            fleet=FleetParams(pods=1, initial_moisture=45.0),
            initial_spots=["SUN"],
        )
        steps = result.timelines["P1"]
        assert (steps[0].action, steps[0].spot_id) == ("irrigate", "SUN")
        assert {s.spot_id for s in steps} == {"SUN"}


def test_pod_retreats_when_water_is_short_for_preemptive_irrigation() -> None:
    # タンクが予備の量を下回っていれば、先回りの散水はせず、既存の規則どおり日影へ退避する
    stage = _stage(
        _spot("SUN", 35.680, 139.760, shade=0.0, demand=5.0),
        _spot("SHADE", 35.681, 139.760, shade=1.0, demand=0.0),
    )
    world = StageWorld.from_stage(stage)
    weather = _weather()
    result = run_fleet(
        world,
        PetTable(world, weather),
        weather,
        policy="greedy",
        fleet=FleetParams(pods=1, initial_moisture=35.0, initial_tank_ml=500),
        initial_spots=["SUN"],
    )
    first = result.timelines["P1"][0]
    assert (first.action, first.spot_id) == ("move", "SHADE")


def _spot_with_demand(sid: str, lat: float, hours: range) -> dict[str, object]:
    spot = _spot(sid, lat, 139.760, shade=0.0, demand=0)
    spot["demand"] = [{"hour": h, "weight": 1.0} for h in hours]
    return spot


def test_greedy_follows_moving_demand_and_beats_fixed() -> None:
    # 午前は MORNING、午後は AFTERNOON に需要がある（約 330 m 離れた日向の地点）。固定配置は片方しか覆えない
    stage = _stage(
        _spot_with_demand("MORNING", 35.680, range(10, 13)),
        _spot_with_demand("AFTERNOON", 35.683, range(13, 16)),
    )
    world = StageWorld.from_stage(stage)
    weather = _weather(temperature_c=30.0)
    pets = PetTable(world, weather)
    results = {
        p: run_fleet(
            world, pets, weather, policy=p, fleet=FleetParams(pods=1), initial_spots=["MORNING"]
        )
        for p in ("greedy", "fixed")
    }
    greedy = summarize(results["greedy"], world, pets)
    fixed = summarize(results["fixed"], world, pets)
    assert greedy.demand_covered_pct == 100.0
    assert fixed.demand_covered_pct == 50.0
    assert greedy.demand_pet_c < fixed.demand_pet_c
