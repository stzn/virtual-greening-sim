"""合成の RR 間隔データ（真のパラメータを埋め込み、解析で取り戻せるかを確かめるため）。

1 セッション = オフィスでの安静（基準、既定 180 秒）→ 条件の切り替え（t=0）→ 接触（600 秒）。
追跡（Design.track_s > 0）を付けると、接触の後に帰り（既定 300 秒、全条件で歩く）と、オフィスで
座っての追跡を、接触の終わりから track_s 秒まで作る。接触の終わりの到達分は時定数 tau_c で減る:

    lnHF(t) = b_s + ou(t) + slow(t) + beta_walk_step * 帰り中 + A_end * exp(-(t - T) / tau_c)  (t >= T)

T は接触の終わり、A_end = A * (1 - exp(-T / tau))。slow は 1 時間の単位のゆっくりした揺らぎ（日内変動の
代わり）で、追跡を付けたときだけ基準から終わりまで足す。追跡なしのときの拍は今までと同じになる。
瞬時の ln(HF)（ln ms²）を次のように作る:

    lnHF(t) = b_s + ou(t)                                          (t < 0)
    lnHF(t) = b_s + ou(t) + beta_walk_step * walk + A * (1 - exp(-t / tau))  (t >= 0)
    A = beta_leave * leave + beta_green * green
        + beta_pet * (PET - PET_REF) + beta_noise * (noise - NOISE_REF) + u_s + e

b_s・u_s は被験者のランダム効果、e はセッションの揺らぎ、ou は ln(HF) のゆっくりした揺らぎ
（Ornstein–Uhlenbeck 過程）。RR は、平均 RR に呼吸性の正弦（パワーが exp(lnHF) になる振幅）と
0.1 Hz の LF 成分、白色雑音を足して、拍ごとに作る。

環境は、オフィス（離脱なし）では緑が少なく PET・騒音は室内の値。離脱した先では緑の量を一様に引き、
緑が多い場所ほど PET・騒音も高くなるようにする（予備計測で見えた、緑の多い強風・騒音の屋上で
指標が下がる逆転を作る交絡）。
"""

from dataclasses import dataclass, field

import numpy as np

from vmgs.analysis.recording import CONDITIONS, FloatArray, Recording

PET_REF_C = 26.0
NOISE_REF_DB = 55.0


@dataclass(frozen=True)
class TrueParams:
    """埋め込む真の値。"""

    tau_s: float = 120.0  # 立ち上がりの時定数（95% 到達は約 6 分）
    beta_walk_step: float = -0.5  # 歩行開始による ln(HF) の即時の低下
    beta_leave: float = 0.2  # オフィスから離れること自体の効果
    beta_green: float = 0.6  # 緑の量 1 あたりの到達値（緑の純粋な効果）
    beta_pet: float = -0.04  # PET 1°C あたり
    beta_noise: float = -0.02  # 騒音 1 dB あたり
    baseline_lnhf: float = 6.0
    subject_baseline_sd: float = 0.5
    subject_response_sd: float = 0.1
    session_response_sd: float = 0.08
    ou_sd: float = 0.12  # ln(HF) のゆっくりした揺らぎの標準偏差
    ou_time_s: float = 20.0
    mean_rr_ms: float = 850.0
    subject_rr_sd_ms: float = 60.0
    walk_rr_drop_ms: float = 150.0
    ln_lf: float = 6.3
    rr_noise_ms: float = 4.0
    tau_c_s: float = 1800.0  # 接触の後の持ち越しの時定数（JITAI の仮の値 30 分と同じ）
    slow_ou_sd: float = 0.2  # 追跡のときのゆっくりした揺らぎ（JITAI の合成のワーカーと同じ仮定）
    slow_ou_time_s: float = 1800.0


@dataclass(frozen=True)
class Design:
    n_subjects: int = 24
    repeats: int = 2  # 被験者 × 条件ごとの繰り返し（環境は毎回引き直す）
    baseline_s: float = 180.0
    contact_s: float = 600.0
    return_s: float = 300.0  # 接触の後、オフィスに戻るまでの歩行（追跡のときだけ使う）
    track_s: float = 0.0  # 接触の終わりから追跡の終わりまで（0 は追跡なし）


@dataclass(frozen=True)
class SessionTruth:
    offset: float  # 即時の段差
    amplitude: float  # 立ち上がりの到達値 A


@dataclass(frozen=True)
class SyntheticStudy:
    recordings: list[Recording]
    truth: TrueParams
    session_truth: dict[str, SessionTruth] = field(default_factory=dict)


def response_amplitude(
    p: TrueParams, leave: float, green: float, pet: float, noise: float
) -> float:
    """立ち上がりの到達値 A の固定効果（被験者・セッションの揺らぎを除く）。JITAI の回復でも使う。"""
    return (
        p.beta_leave * leave
        + p.beta_green * green
        + p.beta_pet * (pet - PET_REF_C)
        + p.beta_noise * (noise - NOISE_REF_DB)
    )


def _environment(leave: bool, rng: np.random.Generator) -> tuple[float, float, float]:
    """(緑の量, PET, 騒音)。離脱先では緑と PET・騒音が正に相関する。"""
    if not leave:
        return float(rng.uniform(0.0, 0.1)), float(rng.normal(24.0, 1.0)), float(rng.normal(55, 3))
    green = float(rng.uniform(0.2, 1.0))
    pet = 27.0 + 8.0 * (green - 0.6) + float(rng.normal(0.0, 2.0))
    noise = 58.0 + 20.0 * (green - 0.6) + float(rng.normal(0.0, 4.0))
    return green, pet, noise


def ou_process(n: int, dt: float, sd: float, time_s: float, rng: np.random.Generator) -> FloatArray:
    """定常 SD が sd、時定数 time_s の OU 過程を dt 刻みで n 点（dt と time_s は同じ単位）。"""
    a = np.exp(-dt / time_s)
    x = np.empty(n)
    x[0] = rng.normal(0.0, sd)
    shocks = rng.normal(0.0, sd * np.sqrt(1 - a * a), n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + shocks[i]
    return x


def _beats(
    t0: float,
    t1: float,
    ln_hf: FloatArray,
    mean_rr: FloatArray,
    grid_t: FloatArray,
    p: TrueParams,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray]:
    """瞬時の ln(HF)・平均 RR（grid_t 上）から拍の時刻と RR を作る。"""
    f_resp = rng.uniform(0.2, 0.3)
    phase_r, phase_l = rng.uniform(0, 2 * np.pi, 2)
    amp_lf = np.sqrt(2.0 * np.exp(p.ln_lf))
    times: list[float] = []
    rrs: list[float] = []
    t = t0
    while t < t1:
        i = min(int(t - grid_t[0]), grid_t.size - 1)  # grid_t は 1 秒刻み
        amp_hf = np.sqrt(2.0 * np.exp(ln_hf[i]))
        rr = (
            mean_rr[i]
            + amp_hf * np.sin(2 * np.pi * f_resp * t + phase_r)
            + amp_lf * np.sin(2 * np.pi * 0.1 * t + phase_l)
            + rng.normal(0.0, p.rr_noise_ms)
        )
        t += rr / 1000.0
        times.append(t)
        rrs.append(rr)
    return np.array(times), np.array(rrs)


def simulate_study(
    design: Design | None = None, params: TrueParams | None = None, seed: int = 0
) -> SyntheticStudy:
    """被験者 × 2×2 条件 × 繰り返しのセッションを作る。乱数は seed で固定する。"""
    design = design or Design()
    p = params or TrueParams()
    rng = np.random.default_rng(seed)
    recordings: list[Recording] = []
    truth: dict[str, SessionTruth] = {}
    lead_s = 5.0  # 最初の窓の手前から拍を作る
    tracked = design.track_s > 0
    end_s = design.contact_s + design.track_s if tracked else design.contact_s
    grid_t = np.arange(-design.baseline_s - lead_s, end_s + 5.0, 1.0)
    after = grid_t >= 0
    post = grid_t >= design.contact_s
    returning = post & (grid_t < design.contact_s + design.return_s)
    for s in range(design.n_subjects):
        subject = f"P{s + 1:02d}"
        b_s = p.baseline_lnhf + rng.normal(0.0, p.subject_baseline_sd)
        u_s = rng.normal(0.0, p.subject_response_sd)
        rr_s = p.mean_rr_ms + rng.normal(0.0, p.subject_rr_sd_ms)
        for condition, (walk, leave) in CONDITIONS.items():
            for r in range(design.repeats):
                green, pet, noise = _environment(leave, rng)
                amplitude = (
                    response_amplitude(p, leave, green, pet, noise)
                    + u_s
                    + rng.normal(0.0, p.session_response_sd)
                )
                offset = p.beta_walk_step * walk
                rise = np.where(after, 1.0 - np.exp(-np.clip(grid_t, 0, None) / p.tau_s), 0.0)
                ou = ou_process(grid_t.size, 1.0, p.ou_sd, p.ou_time_s, rng)
                ln_hf = b_s + ou + after * offset + amplitude * rise
                mean_rr = np.full(grid_t.size, rr_s) - after * walk * p.walk_rr_drop_ms
                track: tuple[float | None, float | None, float | None] = (None, None, None)
                if tracked:
                    # 追跡なしのときの乱数の順を変えないよう、ゆっくりした揺らぎは後から引く
                    slow = ou_process(grid_t.size, 1.0, p.slow_ou_sd, p.slow_ou_time_s, rng)
                    end_level = amplitude * (1.0 - np.exp(-design.contact_s / p.tau_s))
                    since = np.clip(grid_t - design.contact_s, 0, None)
                    carry = end_level * np.exp(-since / p.tau_c_s)
                    ln_hf = (
                        np.where(post, b_s + ou + returning * p.beta_walk_step + carry, ln_hf)
                        + slow
                    )
                    mean_rr = np.where(post, rr_s - returning * p.walk_rr_drop_ms, mean_rr)
                    track = (design.contact_s, design.contact_s + design.return_s, end_s)
                times, rr = _beats(grid_t[0], end_s, ln_hf, mean_rr, grid_t, p, rng)
                session = f"{subject}-{condition}-{r + 1}"
                recordings.append(
                    Recording(
                        session,
                        subject,
                        condition,
                        green,
                        pet,
                        noise,
                        times,
                        rr,
                        contact_end_s=track[0],
                        track_start_s=track[1],
                        track_end_s=track[2],
                    )
                )
                truth[session] = SessionTruth(offset=offset, amplitude=float(amplitude))
    return SyntheticStudy(recordings, p, truth)
