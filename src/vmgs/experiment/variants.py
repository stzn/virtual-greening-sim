"""初期条件のばらつき（バリアント）。

temperature=0 のLLMは同じ入力に同じ出力を返すため、同じ設定を何回回しても試行は同一で、標本数は実質1になる。
基準の初期条件をシード付きで少しずつ変えて、複数回の比較に使う。気象・開始スポット・開始時刻は変えない。
"""

import random
from dataclasses import asdict, dataclass

MOISTURE_JITTER = 10.0  # 土壌水分 ± [%]
TANK_MIN_RATIO = 0.7  # 水タンクは 基準の 70〜100%
BATTERY_MAX_DROP = 15.0  # バッテリーは 基準から 0〜15% 下げる


@dataclass(frozen=True)
class BaseConditions:
    moisture: float = 60.0
    tank_ml: int = 10_000
    battery_pct: float = 100.0
    spot: str = "A"
    start_hour: int = 0


@dataclass(frozen=True)
class Variant:
    index: int
    moisture: float
    tank_ml: int
    battery_pct: float
    spot: str
    start_hour: int

    def conditions(self) -> BaseConditions:
        d = asdict(self)
        d.pop("index")
        return BaseConditions(**d)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def make_variant(base: BaseConditions, seed: int, index: int) -> Variant:
    """index=0 は基準そのまま。それ以外は (seed, index) から決定論的に作る。"""
    if index == 0:
        return Variant(0, base.moisture, base.tank_ml, base.battery_pct, base.spot, base.start_hour)
    rng = random.Random(f"{seed}:{index}")  # 文字列シードは実行をまたいで同じ列になる
    moisture = _clamp(
        round(base.moisture + rng.uniform(-MOISTURE_JITTER, MOISTURE_JITTER), 2), 0, 100
    )
    tank = max(0, round(base.tank_ml * rng.uniform(TANK_MIN_RATIO, 1.0)))
    battery = _clamp(round(base.battery_pct - rng.uniform(0, BATTERY_MAX_DROP), 2), 0, 100)
    return Variant(index, moisture, tank, battery, base.spot, base.start_hour)
