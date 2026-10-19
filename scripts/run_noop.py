"""何もしない（放置）で24時間回し、水分・健康度・バッテリーの推移をグラフ化する。

使い方: uv run python scripts/run_noop.py [開始スポット(A|B|C)]
"""

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vmgs.weather.cache import load_records
from vmgs.weather.providers import SeriesWeather
from vmgs.world.clock import SimClock
from vmgs.world.world import World

SCENARIOS = ("heatwave", "normal", "rainy")


def run(name: str, spot: str, out_dir: Path) -> dict[str, float]:
    records = load_records(Path(f"data/weather/{name}.json"))
    weather = SeriesWeather(records)
    start = records[0].time
    world = World(
        weather, SimClock(start), notes_path=Path(tempfile.mkdtemp()) / "n.md", start_spot=spot
    )
    hours, moist, health, batt, temp = [], [], [], [], []
    for h in range(24):
        w = world.current_weather()
        hours.append(h)
        temp.append(w.temperature_c)
        moist.append(world.plant.moisture)
        health.append(world.plant.health)
        batt.append(world.plant.battery_pct)
        world.advance(1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hours, moist, label="soil moisture %")
    ax.plot(hours, health, label="health")
    ax.plot(hours, batt, label="battery %")
    ax.plot(hours, temp, "--", color="gray", label="air temp C")
    ax.set_xlabel("hour")
    ax.set_title(f"noop 24h: {name}, spot {spot}")
    ax.legend()
    ax.grid(alpha=0.3)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}_{spot}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"min_moisture": min(moist), "final_health": health[-1], "final_battery": batt[-1]}


if __name__ == "__main__":
    spot = sys.argv[1] if len(sys.argv) > 1 else "C"
    for sc in SCENARIOS:
        r = run(sc, spot, Path("runs/noop"))
        print(
            f"{sc:9s} spot={spot} 最低水分={r['min_moisture']:5.1f}% 最終健康度={r['final_health']:5.1f} 最終バッテリー={r['final_battery']:5.1f}%"
        )
