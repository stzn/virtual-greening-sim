"""JITAI の 4 段階（vmgs.jitai）の振る舞い。"""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from vmgs.contracts import Dispatch, Pod, PodStep, Stage, StageRef
from vmgs.fleet.stage_world import StageWorld, load_stage
from vmgs.jitai.calendar import MIN_FREE_MIN, Meeting, free_minutes
from vmgs.jitai.demand import demand_zones
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
    demand_residual_counts,
    run_jitai,
)
from vmgs.jitai.pods import POD_SEATS, WALK_RADIUS_M, Occupancy, PodSchedule
from vmgs.jitai.safety import (
    DEST_PET_MAX_C,
    RAIN_COVER_MAX_MM_H,
    RAIN_MAX_MM_H,
    ROUTE_PET_MAX_C,
    pet_check,
    rain_check,
    rain_limit,
)
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


def test_雨は行きから帰りまでの各時の最大の降水量で見る() -> None:
    rain = {9: 0.2, 10: 3.0, 11: 0.0}
    assert rain_check(rain, [9]) == 0.2
    assert rain_check(rain, [9, 10]) == 3.0
    # 気象に無い時刻は雨なしとみなす
    assert rain_check(rain, [20]) == 0.0


def test_雨の閾値は傘が要る1mm毎時() -> None:
    assert RAIN_MAX_MM_H == 1.0


def _rain_run(rain: dict[int, float] | None, *, rain_cover: bool = False):  # type: ignore[no-untyped-def]
    # 勤務地にポッドが終日いて、PET は低い（暑さでは見送らない）
    world = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    s1 = world.stage_spot("OTM-001")
    w = _one_worker(s1.lat, s1.lon)
    sched = _dispatch({"P1": [(h, "OTM-001") for h in range(24)]})
    return run_jitai([w], world, sched, _const_pet(30.0, 30.0), rain_mm=rain, rain_cover=rain_cover)


def test_雨が1mm毎時以上の時刻は_天候の段階で理由rainとして見送り_15分後に再判定する() -> None:
    # Given 9 時だけ 1 mm/h の雨
    result = _rain_run({h: (1.0 if h == 9 else 0.0) for h in range(24)})
    rained = [r for r in result.records if r.reason == "rain"]
    # Then 9 時台の判定は天候の段階（契約の pet）で見送り、ポッドの情報は残す
    assert rained and all(r.time_min // 60 == 9 for r in rained)
    assert all(r.stage == "pet" and r.pod_id == "P1" for r in rained)
    assert set(np.diff([r.time_min for r in rained])) <= {15}
    # 雨が止んだ後は介入する
    assert any(r.outcome == "intervened" for r in result.records)


def test_雨が1mm毎時未満なら見送らない() -> None:
    result = _rain_run({h: 0.9 for h in range(24)})
    assert result.records[0].outcome == "intervened"
    assert not any(r.reason == "rain" for r in result.records)


def test_雨の気象を渡さなければ雨を見ない() -> None:
    assert not any(r.reason == "rain" for r in _rain_run(None).records)


def test_雨の基準は_雨よけなしは1mm毎時_雨よけありは傘で歩ける5mm毎時() -> None:
    assert rain_limit(rain_cover=False) == RAIN_MAX_MM_H == 1.0
    assert rain_limit(rain_cover=True) == RAIN_COVER_MAX_MM_H == 5.0


def test_ポッドに雨よけがあれば_5mm毎時未満の雨では介入する() -> None:
    # Given 終日 3 mm/h の雨
    rain = {h: 3.0 for h in range(24)}
    # Then 雨よけなしは雨で見送り、雨よけありは座る間は雨よけの下・行き帰りは傘で歩けるので介入する
    assert _rain_run(rain).records[0].reason == "rain"
    covered = _rain_run(rain, rain_cover=True)
    assert covered.records[0].outcome == "intervened"
    assert not any(r.reason == "rain" for r in covered.records)


def test_雨よけを指定しなければ雨よけ有りとして_5mm毎時未満の雨では介入する() -> None:
    # Given 終日 3 mm/h の雨、rain_cover を渡さない
    world = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    s1 = world.stage_spot("OTM-001")
    w = _one_worker(s1.lat, s1.lon)
    sched = _dispatch({"P1": [(h, "OTM-001") for h in range(24)]})
    result = run_jitai(
        [w], world, sched, _const_pet(30.0, 30.0), rain_mm={h: 3.0 for h in range(24)}
    )
    # Then 既定は雨よけ有りなので介入する
    assert result.records[0].outcome == "intervened"


def test_ポッドに雨よけがあっても_5mm毎時以上の雨では見送る() -> None:
    result = _rain_run({h: 5.0 for h in range(24)}, rain_cover=True)
    # 予定で見送る判定もあるので、天候の段階まで進んだ判定がすべて雨で見送られることを見る
    weather = [r for r in result.records if r.stage == "pet"]
    assert weather and all(r.reason == "rain" for r in weather)
    assert not any(r.outcome == "intervened" for r in result.records)


# --- 屋根付きの網（地点データの版 2） ---

SAMPLE_STAGE_V2 = Path("contracts/fixtures/stage.v2.sample.json")


def _sheltered_world(home: str | None, pod: str | None) -> StageWorld:
    """版 2 のサンプルで、OTM-002 を勤務地 OTM-001 の約 110 m 北に寄せ、2 地点の屋根の網を指定する。"""
    data = json.loads(SAMPLE_STAGE_V2.read_text(encoding="utf-8"))
    s1, s2 = data["spots"][0], data["spots"][1]
    s2["lat"] = s1["lat"] + 0.001
    s2["lon"] = s1["lon"]
    for spot, net in ((s1, home), (s2, pod)):
        spot["shelter"] = "open" if net is None else "covered"
        spot["shelter_network"] = net
        spot["shelter_access"] = None  # サンプルの入口は使わない（入口は _access_world で指定する）
    return StageWorld.from_stage(Stage.model_validate(data))


def _sheltered_run(world: StageWorld, rain_mm: float):  # type: ignore[no-untyped-def]
    # 勤務地 OTM-001 のワーカー、ポッドは終日 OTM-002（雨よけあり）、PET は低い
    s1 = world.stage_spot("OTM-001")
    w = _one_worker(s1.lat, s1.lon)
    sched = _dispatch({"P1": [(h, "OTM-002") for h in range(24)]})
    return run_jitai(
        [w], world, sched, _const_pet(30.0, 30.0), rain_mm={h: rain_mm for h in range(24)}
    )


def test_勤務地とポッドが同じ屋根付きの網の上なら_強い雨でも介入する() -> None:
    # Given 勤務地とポッドの地点が同じ網 N1、終日 20 mm/h の雨
    result = _sheltered_run(_sheltered_world("N1", "N1"), 20.0)
    # Then 濡れずに行き帰りできるので、雨では見送らない
    assert result.records[0].outcome == "intervened"
    assert not any(r.reason == "rain" for r in result.records)


@pytest.mark.parametrize(("home", "pod"), [("N1", "N2"), (None, "N1"), ("N1", None), (None, None)])
def test_網が違うか片方が屋根なしなら_今までどおり雨で見送る(
    home: str | None, pod: str | None
) -> None:
    result = _sheltered_run(_sheltered_world(home, pod), 20.0)
    weather = [r for r in result.records if r.stage == "pet"]
    assert weather and all(r.reason == "rain" for r in weather)


def test_濡れずに行けるかは_ポッドが屋根付きで_勤務地の網か入口の網と同じときだけ() -> None:
    # 勤務地も屋根付き
    assert _sheltered_world("N1", "N1").sheltered_route("OTM-001", "OTM-002")
    assert not _sheltered_world("N1", "N2").sheltered_route("OTM-001", "OTM-002")
    assert not _sheltered_world(None, None).sheltered_route("OTM-001", "OTM-002")
    # 勤務地は屋根なしで、出入口から網へ入れる
    assert _access_world("N1", "N1").sheltered_route("OTM-001", "OTM-002")
    assert not _access_world("N2", "N1").sheltered_route("OTM-001", "OTM-002")
    # ポッドは屋根の下にいる必要がある（入口のある屋根なしの地点のポッドへは、外を歩く）
    assert not _access_world("N1", "N1").sheltered_route("OTM-002", "OTM-001")
    # 版 1 の地点データには屋根付きの地点が無い
    v1 = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    assert not v1.sheltered_route("OTM-001", "OTM-001")


def _access_world(access: str | None, pod: str | None) -> StageWorld:
    """勤務地 OTM-001 は屋根なしで、出入口から網 access へ入れる。OTM-002 は網 pod の屋根付きの地点（None なら屋根なし）。"""
    world = _sheltered_world(None, pod)
    data = world.stage.model_dump(mode="json")
    data["spots"][0]["shelter_access"] = access
    return StageWorld.from_stage(Stage.model_validate(data))


def test_屋根なしの勤務地でも_出入口から入れる網の屋根の下のポッドなら_強い雨でも介入する() -> None:
    # Given 勤務地は屋根なしだが、50 m 以内の出入口から網 N1 へ入れる。ポッドは N1 の屋根の下。終日 20 mm/h
    result = _sheltered_run(_access_world("N1", "N1"), 20.0)
    # Then 雨では見送らない
    assert result.records[0].outcome == "intervened"
    assert not any(r.reason == "rain" for r in result.records)


@pytest.mark.parametrize(("access", "pod"), [("N2", "N1"), (None, "N1"), ("N1", None)])
def test_入口の網が違うか_入口が無いか_ポッドが屋根なしなら_雨で見送る(
    access: str | None, pod: str | None
) -> None:
    result = _sheltered_run(_access_world(access, pod), 20.0)
    weather = [r for r in result.records if r.stage == "pet"]
    assert weather and all(r.reason == "rain" for r in weather)


def _rec(t_min: int, worker: str) -> Record:
    return Record(t_min, worker, "skipped", "pods", "no_pod_nearby", None, None, None, None, None)


def test_需要は徒歩圏の区画と時刻で数え_3人未満の区画を除き_区画内の地点に等分する() -> None:
    # Given 区画 Z1 に地点 A・B・D（D に勤務地のワーカーはいない）、区画 Z2 に地点 C
    zone_of = {"A": (0, 0), "B": (0, 0), "D": (0, 0), "C": (5, 5)}
    home = {"W0": "A", "W1": "A", "W2": "B", "W9": "C"}
    # 12 時: Z1 は 3 人（地点単位なら A が 2 人・B が 1 人で両方消える）、Z2 は 1 人
    recs = [_rec(12 * 60, "W0"), _rec(12 * 60 + 30, "W0"), _rec(12 * 60, "W1")]
    recs += [_rec(12 * 60 + 5, "W2"), _rec(12 * 60, "W9")]
    # When 数える
    cells = demand_cells(recs, home, zone_of, min_workers=3)
    # Then Z1 の 4 件を A・B・D に等分し、1 人だけの Z2 は出さない
    assert sorted((c["spot_id"], c["hour"]) for c in cells) == [("A", 12), ("B", 12), ("D", 12)]
    assert sum(c["weight"] for c in cells) == pytest.approx(4)
    assert [c["weight"] for c in cells] == pytest.approx([4 / 3] * 3)


def test_人数で落とした区画は時刻ごとに合算し_3人以上なら全地点に等分して足す() -> None:
    # Given 区画 Z1（A・B）に 3 人、Z2（C）・Z3（D）・Z4（E）に 1 人ずつ（12 時）
    zone_of = {"A": (0, 0), "B": (0, 0), "C": (5, 5), "D": (9, 9), "E": (2, 7)}
    home = {"W0": "A", "W1": "A", "W2": "B", "W3": "C", "W4": "D", "W5": "E"}
    recs = [_rec(12 * 60, w) for w in ("W0", "W1", "W2", "W3", "W4")]
    recs += [_rec(12 * 60, "W5"), _rec(12 * 60 + 15, "W5")]
    # When 数える
    cells = demand_cells(recs, home, zone_of, min_workers=3)
    # Then Z1 の 3 件は A・B に等分し、落とした Z2〜Z4 の 4 件（3 人）は全 5 地点に等分して足す
    by_spot = {(c["spot_id"], c["hour"]): c["weight"] for c in cells}
    assert by_spot == pytest.approx(
        {("A", 12): 1.5 + 0.8, ("B", 12): 1.5 + 0.8, ("C", 12): 0.8, ("D", 12): 0.8, ("E", 12): 0.8}
    )
    assert sum(by_spot.values()) == pytest.approx(7)


def test_落とした区画の合算も3人未満なら出さない() -> None:
    zone_of = {"A": (0, 0), "C": (5, 5)}
    home = {"W0": "A", "W1": "C"}
    recs = [_rec(12 * 60, "W0"), _rec(12 * 60, "W1")]
    assert demand_cells(recs, home, zone_of, min_workers=3) == []


def test_落とした区画の合算は時刻と件数と人数だけを返す() -> None:
    zone_of = {"A": (0, 0), "C": (5, 5), "D": (9, 9)}
    home = {"W0": "A", "W1": "C", "W2": "D", "W3": "A"}
    recs = [_rec(12 * 60, w) for w in ("W0", "W1", "W2")] + [_rec(13 * 60, "W3")]
    assert demand_residual_counts(recs, home, zone_of, min_workers=3) == [
        {"hour": 12, "count": 3, "workers": 3, "spots": 3}
    ]


def test_同じ区画でも時刻が違えば別に数える() -> None:
    zone_of = {"A": (0, 0)}
    home = {f"W{i}": "A" for i in range(3)}
    recs = [_rec(12 * 60, "W0"), _rec(12 * 60, "W1"), _rec(13 * 60, "W2")]
    assert demand_cells(recs, home, zone_of, min_workers=3) == []


def test_区画は区域の南西の角から一辺の長さで切る() -> None:
    stage = load_stage(SAMPLE_STAGE)
    # サンプルの 3 地点は互いに 400 m 以上離れていて、別々の区画になる
    assert len(set(demand_zones(stage, 400.0).values())) == 3
    # 区域より大きい区画なら 1 つにまとまる
    assert len(set(demand_zones(stage, 5000.0).values())) == 1
    assert set(demand_zones(stage, 400.0)) == {s.id for s in stage.spots}
