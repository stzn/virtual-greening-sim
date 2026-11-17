"""介入後の回復（フェーズ 5 の R(t)）を、JITAI の合成ワーカーの ln(HF) に足す。

ポッドの樹冠の下での滞在中は、フェーズ 5 の合成モデルと同じ 1 次の立ち上がり
A·(1 − e^{−s/τ}) で ln(HF) が戻る（A・τ は vmgs.analysis.synthetic.TrueParams と共有する）。
帰った後は、滞在の終わりの値が CARRYOVER_MIN の時定数で減る。持ち越しの長さには根拠がなく
（フェーズ 5 は接触中の 10 分しか扱っていない）、仮の値である。根拠と前提は docs/jitai.md。
"""

import numpy as np
import numpy.typing as npt

from vmgs.analysis.recording import FloatArray
from vmgs.analysis.synthetic import NOISE_REF_DB, TrueParams, response_amplitude

POD_GREEN = 0.8  # 移動緑化ポッドの樹冠の下の「緑の量」（0〜1。仮定）
CARRYOVER_MIN = 30.0  # 帰った後に回復が減る時定数 [分]（根拠のない仮の値）


def recovery_amplitude(pet_destination_c: float, params: TrueParams | None = None) -> float:
    """ポッドでの回復の到達値 A（オフィスから離れる・緑 POD_GREEN・目的地の PET・騒音は基準値）。"""
    p = params or TrueParams()
    return response_amplitude(p, 1.0, POD_GREEN, pet_destination_c, NOISE_REF_DB)


def recovery_curve(
    t_min: npt.NDArray[np.int64],
    arrive_min: int,
    leave_min: int,
    *,
    amplitude: float,
    tau_s: float,
    carryover_min: float,
) -> FloatArray:
    """各窓の時刻での回復分（ln 単位）。到着前は 0。"""
    t = t_min.astype(float)
    out = np.zeros(t.size)
    during = (t >= arrive_min) & (t < leave_min)
    out[during] = amplitude * (1.0 - np.exp(-(t[during] - arrive_min) * 60.0 / tau_s))
    peak = amplitude * (1.0 - np.exp(-(leave_min - arrive_min) * 60.0 / tau_s))
    after = t >= leave_min
    if carryover_min > 0:
        out[after] = peak * np.exp(-(t[after] - leave_min) / carryover_min)
    return out
