"""介入後の回復（R(t)）を JITAI の次の判定に反映する（vmgs.jitai.recovery・engine）振る舞い。"""

from pathlib import Path

import numpy as np
import pytest

from vmgs.analysis.recording import CONDITIONS
from vmgs.analysis.synthetic import PET_REF_C, TrueParams, response_amplitude, simulate_study
from vmgs.contracts import Dispatch, Pod, PodStep, StageRef
from vmgs.fleet.stage_world import StageWorld, load_stage
from vmgs.jitai.detect import worker_flags
from vmgs.jitai.engine import run_jitai
from vmgs.jitai.recovery import POD_GREEN, recovery_amplitude, recovery_curve
from vmgs.jitai.workers import Episode, Worker, time_grid_min

SAMPLE_STAGE = Path("contracts/fixtures/stage.sample.json")
T = time_grid_min()
P = TrueParams()


# --- 振幅（フェーズ 5 の合成モデルと共有する） ---


def test_振幅の式はフェーズ5の合成データの真の振幅を再現する() -> None:
    # Given フェーズ 5 の合成の研究（被験者・セッションの揺らぎを 0 にする）
    p = TrueParams(subject_response_sd=0.0, session_response_sd=0.0)
    study = simulate_study(params=p, seed=0)
    # When 各セッションの環境から振幅を計算する / Then 埋め込んだ真の振幅と一致する
    for rec in study.recordings:
        leave = float(CONDITIONS[rec.condition][1])
        got = response_amplitude(p, leave, rec.green, rec.pet_c, rec.noise_db)
        assert got == pytest.approx(study.session_truth[rec.session].amplitude, abs=1e-9)


def test_ポッドでの回復の振幅は_PETが高いほど小さい() -> None:
    a30 = recovery_amplitude(30.0)
    a37 = recovery_amplitude(37.0)
    assert a30 == pytest.approx(
        P.beta_leave + P.beta_green * POD_GREEN + P.beta_pet * (30.0 - PET_REF_C)
    )
    assert a37 < a30
    assert a30 == pytest.approx(0.52, abs=0.01)
    assert a37 == pytest.approx(0.24, abs=0.01)


# --- 回復の曲線（5 分ごとの窓） ---


def test_回復は到着前は0_滞在中は立ち上がり_帰った後は持ち越しの時定数で減る() -> None:
    arrive, leave = 10 * 60, 10 * 60 + 10
    r = recovery_curve(T, arrive, leave, amplitude=0.5, tau_s=P.tau_s, carryover_min=30.0)
    assert np.all(r[arrive > T] == 0.0)
    at = {int(t): float(v) for t, v in zip(T, r, strict=True)}
    assert at[arrive] == 0.0
    assert at[arrive + 5] == pytest.approx(0.5 * (1 - np.exp(-300 / P.tau_s)))
    peak = 0.5 * (1 - np.exp(-600 / P.tau_s))
    assert at[leave] == pytest.approx(peak)
    assert at[leave + 30] == pytest.approx(peak * np.exp(-1.0))
    assert at[leave + 60] < at[leave + 30]


def test_持ち越しが0なら帰った後は回復が残らない() -> None:
    r = recovery_curve(T, 600, 610, amplitude=0.5, tau_s=P.tau_s, carryover_min=0.0)
    assert np.all(r[T > 610] == 0.0)


# --- 次の判定への反映 ---


def _dispatch_all_day() -> Dispatch:
    return Dispatch(
        schema_version="1",
        generator="test",
        stage_ref=StageRef(area_id="otemachi", date=load_stage(SAMPLE_STAGE).date),
        weather_scenario="normal",
        pods=[
            Pod(
                id="P1",
                timeline=[
                    PodStep(hour=h, spot_id="OTM-001", action="stay", plant_health=1.0)
                    for h in range(24)
                ],
            )
        ],
        comfort=[],
    )


def _stressed_worker(world: StageWorld) -> Worker:
    # 個人の揺らぎ SD 0.25、9〜12 時に 0.9 下がる（z ≈ −3.6）。勤務地はポッドのいる地点
    s = world.stage_spot("OTM-001")
    cal = np.random.default_rng(7).normal(6.0, 0.25, (5, T.size))
    x = np.full(T.size, 6.0)
    x[(T >= 9 * 60) & (T < 12 * 60)] -= 0.9
    return Worker("W01", s.lat, s.lon, "OTM-001", cal, x, (Episode(9 * 60, 12 * 60, -0.9),), ())


def _run(pet_c: float, *, recovery: bool, carryover_min: float = 30.0):  # type: ignore[no-untyped-def]
    world = StageWorld.from_stage(load_stage(SAMPLE_STAGE))
    w = _stressed_worker(world)

    def pet(spot_id: str, hour: int, *, covered: bool) -> float:
        return pet_c if covered else 30.0

    return w, run_jitai(
        [w], world, _dispatch_all_day(), pet, recovery=recovery, carryover_min=carryover_min
    )


def test_回復を反映しないと_過緊張の窓は介入前の検知と同じ() -> None:
    w, result = _run(30.0, recovery=False)
    assert result.hyperstress_windows == int(worker_flags(w).sum())


def test_介入して回復すると_その後の過緊張の窓とエピソードの負担が減る() -> None:
    _, without = _run(30.0, recovery=False)
    _, with_recovery = _run(30.0, recovery=True)
    assert with_recovery.records[0].outcome == "intervened"
    assert with_recovery.hyperstress_windows < without.hyperstress_windows
    assert with_recovery.episode_burden < without.episode_burden


def test_目的地のPETが高いと_回復が小さく_過緊張の窓は減りにくい() -> None:
    # PET 37℃（目的地の閾値 38℃ の手前）は通るが、回復の振幅は 0.24 で z は −2 を上回らない
    _, without = _run(37.0, recovery=False)
    _, hot = _run(37.0, recovery=True)
    _, mild = _run(30.0, recovery=True)
    assert hot.records[0].outcome == "intervened"
    assert mild.hyperstress_windows < hot.hyperstress_windows <= without.hyperstress_windows
    assert mild.episode_burden < hot.episode_burden < without.episode_burden


def test_持ち越しが長いほど_過緊張の窓が減る() -> None:
    _, none = _run(30.0, recovery=True, carryover_min=0.0)
    _, long = _run(30.0, recovery=True, carryover_min=60.0)
    assert long.hyperstress_windows < none.hyperstress_windows
