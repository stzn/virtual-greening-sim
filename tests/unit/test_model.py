import pytest

from vmgs.weather.models import WeatherRecord
from vmgs.world.params import Params
from vmgs.world.plant import evapotranspiration, health_delta
from vmgs.world.spots import SPOTS, move_cost_pct

P = Params()


def _w(**kw: float) -> WeatherRecord:
    base = {
        "time": "2025-08-01T13:00",
        "temperature_c": 30.0,
        "humidity_pct": 50.0,
        "shortwave_w_m2": 500.0,
        "wind_ms": 1.0,
        "precipitation_mm": 0.0,
    }
    return WeatherRecord.model_validate({**base, **kw})


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [("shortwave_w_m2", 100.0, 900.0), ("temperature_c", 22.0, 35.0), ("wind_ms", 0.5, 5.0)],
)
def test_蒸散は日射_気温_風速に対して単調増加(field: str, low: float, high: float) -> None:
    spot = SPOTS["A"]
    lo, hi = _w(**{field: low}), _w(**{field: high})
    et_lo = evapotranspiration(
        lo, spot, spot.effective_temp(lo.temperature_c, lo.shortwave_w_m2, P), P
    )
    et_hi = evapotranspiration(
        hi, spot, spot.effective_temp(hi.temperature_c, hi.shortwave_w_m2, P), P
    )
    assert et_lo < et_hi


def test_蒸散は負にならない() -> None:
    w = _w(shortwave_w_m2=0.0, temperature_c=5.0, wind_ms=0.0, humidity_pct=100.0)
    assert evapotranspiration(w, SPOTS["A"], 5.0, P) == 0.0


@pytest.mark.parametrize(
    ("moisture", "temp", "sign"),
    [(50, 25, 1), (10, 25, -1), (95, 25, -1), (75, 25, 0), (50, 40, -1)],
)
def test_健康度の変化の向き(moisture: float, temp: float, sign: int) -> None:
    d = health_delta(moisture, temp, P)
    assert (d > 0) - (d < 0) == sign


@pytest.mark.parametrize(("a", "b", "cost"), [("A", "C", 5.0), ("B", "C", 10.0), ("A", "B", 6.667)])
def test_移動コストは距離に比例して5から10パーセント(a: str, b: str, cost: float) -> None:
    assert move_cost_pct(a, b, P) == pytest.approx(cost, abs=0.01)
    assert move_cost_pct(a, b, P) == move_cost_pct(b, a, P)
