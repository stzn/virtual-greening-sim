from pathlib import Path

import pytest

from vmgs import cli
from vmgs.agent.baseline import BaselineAgent
from vmgs.eval.report import plot_run
from vmgs.sim.presets import PRESETS
from vmgs.sim.simulate import SimConfig, run_simulation
from vmgs.sim.store import RunStore
from vmgs.weather.cache import load_records
from vmgs.weather.providers import SeriesWeather

DATA = Path(__file__).resolve().parents[2] / "data" / "weather"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "WEATHER_DIR", DATA)
    return tmp_path


def test_tradeoffプリセットは14時開始で猛暑のCから始まる() -> None:
    p = PRESETS["tradeoff"]
    assert (p.scenario, p.start_hour, p.spot, p.initial_moisture) == ("heatwave", 14, "C", 25.0)
    weather = SeriesWeather(load_records(DATA / "heatwave.json"))
    assert weather.first_time.replace(hour=p.start_hour).hour == 14


def test_気象データより長い時間を指定しても落ちずに理由を返す(tmp_path: Path) -> None:
    weather = SeriesWeather(load_records(DATA / "heatwave.json"))
    cfg = SimConfig(scenario="heatwave", hours=24, start_hour=14)  # 14時開始なので10時間分しかない
    res = run_simulation(cfg, weather, BaselineAgent, tmp_path, "r")
    assert (res.reason, len(res.steps)) == ("WEATHER_EXHAUSTED", 10)


def test_plot_runはPNGを書き出す(tmp_path: Path) -> None:
    weather = SeriesWeather(load_records(DATA / "heatwave.json"))
    res = run_simulation(
        SimConfig(scenario="heatwave", hours=6), weather, BaselineAgent, tmp_path, "r"
    )
    out = tmp_path / "x" / "t.png"
    plot_run(res.steps, out, "t")
    assert out.stat().st_size > 1000


def test_CLI_runは完走して成果物を出し_中断と再開もできる(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(
            ["run", "--scenario", "normal", "--hours", "6", "--run-id", "a", "--stop-after", "3"]
        )
        == 0
    )
    assert "STOPPED" in capsys.readouterr().out
    assert cli.main(["run", "--resume", "a"]) == 0
    assert "COMPLETED" in capsys.readouterr().out
    assert len(RunStore(runs / "a").read_steps()) == 6
    assert (runs / "a" / "timeseries.png").exists()


def test_CLI_再開で存在しない実行は終了コード2(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["run", "--resume", "none"]) == 2
    assert "見つかりません" in capsys.readouterr().err


def test_CLI_compareはレポートを表示して保存する(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["compare", "--scenario", "heatwave", "--agents", "baseline,noop"]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "noop" in out
    assert list(runs.glob("*_cmp.md"))


def test_CLI_llmはAPIキーが無ければエラーで終了する(
    runs: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(runs)  # .env を読まない場所へ
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert cli.main(["run", "--scenario", "normal", "--agent", "llm", "--provider", "openai"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err
