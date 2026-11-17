"""JITAI の 4 段階（vmgs.jitai）の振る舞い。"""

import itertools
from pathlib import Path

import numpy as np
import pytest

from vmgs.contracts import Dispatch, Pod, PodStep, StageRef
from vmgs.fleet.stage_world import StageWorld, load_stage
from vmgs.jitai.calendar import MIN_FREE_MIN, Meeting, free_minutes
from vmgs.jitai.detect import (
    Z_THRESHOLD,
    detect,
    detection_scores,
    fit_circadian,
    fit_flat,
)
from vmgs.jitai.engine import (
    COOLDOWN_AFTER_INTERVENTION_MIN,
    Record,
    demand_cells,
    run_jitai,
)
from vmgs.jitai.pods import POD_SEATS, WALK_RADIUS_M, Occupancy, PodSchedule
from vmgs.jitai.safety import DEST_PET_MAX_C, ROUTE_PET_MAX_C, pet_check
from vmgs.jitai.workers import (
    Episode,
    Worker,
    WorkerModel,
    circadian,
    make_workers,
    time_grid_min,
)

SAMPLE_STAGE = Path("contracts/fixtures/stage.sample.json")
T = time_grid_min()


def _const_pet(dest: float, route: float):  # type: ignore[no-untyped-def]
    def pet(spot_id: str, hour: int, *, covered: bool) -> float:
        return dest if covered else route

    return pet


# --- 段階 1: 日内変動を補正した過緊張の検知 ---


def test_日内変動の当てはめは埋め込んだ振幅と位相を取り戻す() -> None:
    # Given 振幅 0.5・頂点 4 時の日内変動に雑音を足した 5 日分の ln(HF)
    rng = np.random.default_rng(0)
    days = np.array([6.0 + circadian(T, 0.5, 4.0) + rng.normal(0, 0.1, T.size) for _ in range(5)])
    # When 24 時間周期の余弦で当てはめる
    fit = fit_circadian(T, days)
    # Then 日中の予測は真の曲線から 0.1 以内、残差の SD は雑音に近い
    assert np.max(np.abs(fit.predict(T) - (6.0 + circadian(T, 0.5, 4.0)))) < 0.1
    assert fit.sd == pytest.approx(0.1, rel=0.3)


def test_持続した低下を検知し_一瞬の低下では検知しない() -> None:
    # Given 平らな基準（平均 6、SD 0.2）
    fit = fit_flat(
        T, np.full((3, T.size), 6.0) + np.random.default_rng(1).normal(0, 0.2, (3, T.size))
    )
    x = np.full(T.size, 6.0)
    # When 1 窓だけ大きく下がる / 4 窓続けて下がる
    x_spike = x.copy()
    x_spike[50] = 4.0
    x_sustained = x.copy()
    x_sustained[60:64] = 6.0 + 2.5 * Z_THRESHOLD * fit.sd
    # Then 一瞬の低下では検知せず、続いた低下は 3 窓目から検知する
    assert not detect(T, x_spike, fit).any()
    flags = detect(T, x_sustained, fit)
    assert not flags[:62].any() and flags[62]


def test_計算できない窓NaNは欠測として扱い_検知の根拠にしない() -> None:
    fit = fit_flat(T, np.random.default_rng(2).normal(6.0, 0.2, (3, T.size)))
    x = np.full(T.size, np.nan)
    assert not detect(T, x, fit).any()


def test_日内変動を補正しないと_誤検知が日内変動の谷の午後に集中する() -> None:
    # Given 過緊張のエピソードが無いワーカー 40 人（誤検知だけが起きうる）
    stage = load_stage(SAMPLE_STAGE)
    workers = make_workers(stage, 40, seed=3, model=WorkerModel(episodes_per_day=0.0))
    # When 補正なし・ありで検知する
    naive = detection_scores(workers, corrected=False)
    corrected = detection_scores(workers, corrected=True)
    # Then 補正なしの誤検知はすべて 13 時以降、補正ありは午前にも散らばる（時刻に偏らない）
    assert naive.false_alarm_windows > 0
    assert naive.false_alarm_windows_afternoon == naive.false_alarm_windows
    assert corrected.false_alarm_windows_afternoon < corrected.false_alarm_windows


def test_日内変動を補正すると_エピソードの検知が増える() -> None:
    # Given エピソードのあるワーカー 40 人（seed を変えても同じ向き）
    stage = load_stage(SAMPLE_STAGE)
    for seed in range(3):
        workers = make_workers(stage, 40, seed=seed)
        naive = detection_scores(workers, corrected=False)
        corrected = detection_scores(workers, corrected=True)
        # Then 補正したほうが多くのエピソードを検知する
        assert corrected.detected_episodes > naive.detected_episodes


def test_合成のエピソードの大半を検知する() -> None:
    stage = load_stage(SAMPLE_STAGE)
    workers = make_workers(stage, 30, seed=4)
    scores = detection_scores(workers, corrected=True)
    assert scores.episodes > 0
    assert scores.detected_episodes / scores.episodes >= 0.8


# --- 段階 2: 予定の空き枠 ---


def test_会議中は空きが0_次の会議までの分数が空き() -> None:
    meetings = (Meeting(10 * 60, 11 * 60), Meeting(14 * 60, 14 * 60 + 30))
    assert free_minutes(meetings, 10 * 60 + 15) == 0
    assert free_minutes(meetings, 13 * 60 + 30) == 30
    assert free_minutes(meetings, 15 * 60) == 19 * 60 - 15 * 60  # 終業まで


def test_必要な空きは滞在と往復の徒歩の和() -> None:
    assert MIN_FREE_MIN == 10 + 2 * 5


# --- 段階 3: 徒歩圏のポッドの空き ---


def _dispatch(steps: dict[str, list[tuple[int, str]]]) -> Dispatch:
    return Dispatch(
        schema_version="1",
        stage_ref=StageRef(area_id="otemachi", date="2025-08-30"),  # type: ignore[arg-type]
        generator="test",
        weather_scenario="normal",
        pods=[
            Pod(
                id=p,
                timeline=[
                    PodStep(hour=h, spot_id=s, action="stay", plant_health=90) for h, s in tl
                ],
            )
            for p, tl in steps.items()
        ],
    )


def test_徒歩圏は400m_時刻の地点だけを見る() -> None:
    world = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    sched = PodSchedule(_dispatch({"P1": [(12, "OTM-001")], "P2": [(12, "OTM-002")]}))
    s1 = world.stage_spot("OTM-001")
    near = sched.nearby(world, s1.lat, s1.lon, hour=12)
    assert [c.pod_id for c in near] == ["P1"]  # OTM-002 は約 670 m 先
    assert WALK_RADIUS_M == 400
    assert sched.nearby(world, s1.lat, s1.lon, hour=13) == []


def test_滞在の終わりまで同じ地点にいるポッドだけを使う() -> None:
    sched = PodSchedule(_dispatch({"P1": [(12, "OTM-001"), (13, "OTM-002")]}))
    assert sched.stays("P1", "OTM-001", 12 * 60 + 40, 12 * 60 + 55)
    assert not sched.stays("P1", "OTM-001", 12 * 60 + 50, 13 * 60 + 5)


def test_席の数を超えて予約できない() -> None:
    occ = Occupancy()
    for _ in range(POD_SEATS):
        assert occ.free("P1", 600, 610)
        occ.book("P1", 600, 610)
    assert not occ.free("P1", 605, 615)
    assert occ.free("P1", 610, 620)  # 時間が重ならなければ空く


# --- 段階 4: 二段階の PET 安全判定 ---


def test_目的地の樹冠下PETが閾値以上なら見送る() -> None:
    reason, dest, _ = pet_check(_const_pet(DEST_PET_MAX_C, 30.0), "S", ["S"], [12])
    assert reason == "pet_destination" and dest == DEST_PET_MAX_C


def test_道中の日向PETが閾値以上なら見送る() -> None:
    reason, _, route = pet_check(_const_pet(30.0, ROUTE_PET_MAX_C), "S", ["S"], [12])
    assert reason == "pet_route" and route == ROUTE_PET_MAX_C


def test_両方が閾値未満なら通す() -> None:
    reason, _, _ = pet_check(
        _const_pet(DEST_PET_MAX_C - 0.1, ROUTE_PET_MAX_C - 0.1), "S", ["S"], [12]
    )
    assert reason is None


def test_閾値は高温多湿の区分に合わせる() -> None:
    assert (DEST_PET_MAX_C, ROUTE_PET_MAX_C) == (38.0, 42.0)


# --- 全体 ---


def _one_worker(spot_lat: float, spot_lon: float) -> Worker:
    # 9 時から 19 時まで深い低下が続くワーカー（判定点が繰り返し来る）
    cal = np.random.default_rng(5).normal(6.0, 0.1, (3, T.size))
    x = np.full(T.size, 6.0)
    x[T >= 9 * 60] = 4.0
    return Worker(
        id="W01",
        lat=spot_lat,
        lon=spot_lon,
        home_spot_id="OTM-001",
        calibration_ln_hf=cal,
        ln_hf=x,
        episodes=(Episode(9 * 60, 19 * 60, -2.0),),
        meetings=(),
    )


def test_介入の後は一定時間判定しない_見送りの後は15分後に再判定する() -> None:
    world = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    s1 = world.stage_spot("OTM-001")
    w = _one_worker(s1.lat, s1.lon)
    sched = _dispatch({"P1": [(h, "OTM-001") for h in range(24)]})
    result = run_jitai([w], world, sched, _const_pet(30.0, 30.0))
    times = [r.time_min for r in result.records]
    assert result.records[0].outcome == "intervened"
    assert all(b - a >= COOLDOWN_AFTER_INTERVENTION_MIN for a, b in itertools.pairwise(times))
    # PET で見送り続けると 15 分ごとに再判定する
    blocked = run_jitai([w], world, sched, _const_pet(40.0, 30.0))
    t2 = [r.time_min for r in blocked.records]
    # 終業間際（空きが 20 分未満）は予定の段階で見送る
    for r in blocked.records:
        late = r.time_min > 19 * 60 - MIN_FREE_MIN
        assert r.reason == ("short_gap" if late else "pet_destination")
    assert set(np.diff(t2)) == {15}


def test_需要の集計は3人未満の地点時刻を除く() -> None:
    recs = [
        Record(
            12 * 60 + i, f"W{i}", "skipped", "pods", "no_pod_nearby", None, None, None, None, None
        )
        for i in range(3)
    ] + [Record(13 * 60, "W9", "skipped", "pods", "no_pod_nearby", None, None, None, None, None)]
    home = {"W0": "A", "W1": "A", "W2": "A", "W9": "A"}
    cells = demand_cells(recs, home, min_workers=3)
    assert cells == [{"spot_id": "A", "hour": 12, "weight": 3}]
