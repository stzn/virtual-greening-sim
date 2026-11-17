"""合成のワーカー集団（勤務日の ln(HF) の時系列・過緊張のエピソード・予定・勤務地）。

ln(HF) は 5 分ごとの窓の値を直接作る（実データでは RR から vmgs.analysis.hrv.windowed_hrv で求める
値に当たる。1 日 × 多人数の拍を作ると重いため、窓の値から始める）。

    lnHF(t) = b_s + c_s(t) + ou(t) + e(t) + Σ depth_k · 1[t ∈ エピソード k]
    c_s(t) = amp_s · cos(2π (t − acrophase_s) / 24 h)          （個人の日内変動）

b_s の平均・SD はフェーズ 5 の合成データ（TrueParams）と同じ値を使う。ゆっくりした揺らぎ ou も
同じ OU 過程（vmgs.analysis.synthetic.ou_process）で作る。数値の根拠は docs/jitai.md。
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vmgs.analysis.recording import FloatArray
from vmgs.analysis.synthetic import TrueParams, ou_process
from vmgs.contracts import Stage
from vmgs.fleet.stage_world import EARTH_RADIUS_M, haversine_m
from vmgs.jitai.calendar import Meeting

IntArray = npt.NDArray[np.int64]

DAY_START_MIN = 8 * 60
DAY_END_MIN = 19 * 60
STEP_MIN = 5
LUNCH_MIN = (12 * 60, 13 * 60)  # 昼休みには予定を入れない


@dataclass(frozen=True)
class Episode:
    """過緊張のエピソード。[start_min, end_min) の間 ln(HF) が depth（負）だけ下がる。"""

    start_min: int
    end_min: int
    depth: float


@dataclass(frozen=True)
class WorkerModel:
    """合成の前提。"""

    circadian_amp: float = 0.5  # 日内変動の振幅（ln 単位。山と谷の差は 1.0）
    circadian_amp_sd: float = 0.1
    acrophase_h: float = 4.0  # 迷走神経の活動が最も高い時刻（夜間）
    acrophase_sd_h: float = 1.0
    ou_sd: float = 0.2
    ou_time_min: float = 30.0
    noise_sd: float = 0.1  # 窓ごとの推定の誤差
    episodes_per_day: float = 1.5
    episode_depth: tuple[float, float] = (0.8, 1.2)  # 下がる幅（ln 単位）
    episode_minutes: tuple[int, int] = (20, 60)
    meetings_per_day: float = 4.0
    calibration_days: int = 5  # 個人の基準を当てはめる過去の勤務日
    office_radius_m: float = 300.0  # 勤務地は、舞台の地点からこの範囲に置く


@dataclass(frozen=True, eq=False)
class Worker:
    id: str
    lat: float
    lon: float
    home_spot_id: str  # 勤務地に最も近い舞台の地点（需要の集計と道中の PET に使う）
    calibration_ln_hf: FloatArray  # (calibration_days, 窓の数)。過去の勤務日
    ln_hf: FloatArray  # (窓の数,)。当日
    episodes: tuple[Episode, ...]  # 当日のエピソード（検知の評価に使う真の値）
    meetings: tuple[Meeting, ...]


def time_grid_min() -> IntArray:
    """判定する時刻（8:00〜18:55 の 5 分ごと。0 時からの分）。"""
    return np.arange(DAY_START_MIN, DAY_END_MIN, STEP_MIN, dtype=np.int64)


def circadian(
    t_min: npt.NDArray[np.int64] | FloatArray, amp: float, acrophase_h: float
) -> FloatArray:
    out: FloatArray = amp * np.cos(2 * np.pi * (t_min / 60.0 - acrophase_h) / 24.0)
    return out


def _episodes(model: WorkerModel, rng: np.random.Generator) -> tuple[Episode, ...]:
    eps = []
    for _ in range(int(rng.poisson(model.episodes_per_day))):
        start = int(rng.integers(9 * 60 // STEP_MIN, 18 * 60 // STEP_MIN)) * STEP_MIN
        length = int(rng.integers(model.episode_minutes[0], model.episode_minutes[1] + 1))
        depth = -float(rng.uniform(*model.episode_depth))
        eps.append(Episode(start, min(DAY_END_MIN, start + length), depth))
    return tuple(sorted(eps, key=lambda e: e.start_min))


def _day(
    t: IntArray,
    level: float,
    amp: float,
    phase: float,
    episodes: tuple[Episode, ...],
    model: WorkerModel,
    rng: np.random.Generator,
) -> FloatArray:
    x = (
        level
        + circadian(t, amp, phase)
        + ou_process(t.size, STEP_MIN, model.ou_sd, model.ou_time_min, rng)
        + rng.normal(0.0, model.noise_sd, t.size)
    )
    for e in episodes:
        x[(t >= e.start_min) & (t < e.end_min)] += e.depth
    return x


def _meetings(model: WorkerModel, rng: np.random.Generator) -> tuple[Meeting, ...]:
    """9:00〜18:00 の 30 分刻みに、30 分か 60 分の予定を重ならないように置く（昼休みを除く）。"""
    taken: list[Meeting] = []
    for _ in range(int(rng.poisson(model.meetings_per_day))):
        start = int(rng.integers(18, 36)) * 30  # 9:00〜17:30
        m = Meeting(start, start + int(rng.choice([30, 60])))
        if m.end_min > 18 * 60 or (m.start_min < LUNCH_MIN[1] and LUNCH_MIN[0] < m.end_min):
            continue
        if any(o.start_min < m.end_min and m.start_min < o.end_min for o in taken):
            continue
        taken.append(m)
    return tuple(sorted(taken, key=lambda m: m.start_min))


def _offset(lat: float, lon: float, dist_m: float, bearing: float) -> tuple[float, float]:
    dlat = dist_m * math.cos(bearing) / EARTH_RADIUS_M
    dlon = dist_m * math.sin(bearing) / (EARTH_RADIUS_M * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def make_workers(
    stage: Stage, n: int, seed: int = 0, model: WorkerModel | None = None
) -> list[Worker]:
    """n 人の合成ワーカー。乱数は seed で固定する。"""
    model = model or WorkerModel()
    p = TrueParams()
    rng = np.random.default_rng(seed)
    t = time_grid_min()
    workers = []
    for i in range(n):
        anchor = stage.spots[int(rng.integers(len(stage.spots)))]
        lat, lon = _offset(
            anchor.lat,
            anchor.lon,
            float(rng.uniform(0.0, model.office_radius_m)),
            float(rng.uniform(0.0, 2 * math.pi)),
        )
        home = min(stage.spots, key=lambda s: haversine_m(lat, lon, s.lat, s.lon))
        level = p.baseline_lnhf + float(rng.normal(0.0, p.subject_baseline_sd))
        amp = model.circadian_amp + float(rng.normal(0.0, model.circadian_amp_sd))
        phase = model.acrophase_h + float(rng.normal(0.0, model.acrophase_sd_h))
        calibration = np.array(
            [
                _day(t, level, amp, phase, _episodes(model, rng), model, rng)
                for _ in range(model.calibration_days)
            ]
        )
        episodes = _episodes(model, rng)
        workers.append(
            Worker(
                id=f"W{i + 1:02d}",
                lat=lat,
                lon=lon,
                home_spot_id=home.id,
                calibration_ln_hf=calibration,
                ln_hf=_day(t, level, amp, phase, episodes, model, rng),
                episodes=episodes,
                meetings=_meetings(model, rng),
            )
        )
    return workers
