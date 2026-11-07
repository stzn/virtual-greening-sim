from pathlib import Path

import pytest

from vmgs import cli
from vmgs.experiment.configs import ConfigSpec, parse_config_spec
from vmgs.sim.store import RunStore

DATA = Path(__file__).resolve().parents[2] / "data" / "weather"


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cli, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cli, "WEATHER_DIR", DATA)
    monkeypatch.chdir(tmp_path)  # 実際の .env を読まない
    for name in ("LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("baseline", ConfigSpec("baseline")),
        ("noop", ConfigSpec("noop")),
        ("llm", ConfigSpec("llm")),
        ("llm:ollama:qwen3:14b:v9", ConfigSpec("llm", "ollama", "qwen3:14b", "v9")),
        ("llm:ollama:qwen3:14b", ConfigSpec("llm", "ollama", "qwen3:14b", None)),
        ("llm:openai:gpt-4o-mini:v8", ConfigSpec("llm", "openai", "gpt-4o-mini", "v8")),
        ("llm:ollama:gemma3:27b:v9", ConfigSpec("llm", "ollama", "gemma3:27b", "v9")),
    ],
)
def test_構成の指定を解釈する(text: str, expected: ConfigSpec) -> None:
    assert parse_config_spec(text) == expected


@pytest.mark.parametrize("text", ["gpt", "llm:foo:bar", "llm:ollama", "baseline:v9", ""])
def test_不正な構成の指定はエラー(text: str) -> None:
    with pytest.raises(ValueError, match="構成の指定が不正"):
        parse_config_spec(text)


def test_実行IDに使える名前になる() -> None:
    assert parse_config_spec("llm:ollama:qwen3:14b:v9").name == "llm_ollama_qwen3_14b_v9"
    assert parse_config_spec("baseline").name == "baseline"
    assert "/" not in parse_config_spec("llm:openai:org/model:v1").name


def test_CLI_experimentは比較して保存し_同名で再実行すると再利用する(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "experiment",
        "--name",
        "x",
        "--scenario",
        "heatwave",
        "--hours",
        "6",
        "--variants",
        "2",
        "--configs",
        "baseline,noop",
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "標本数が少ない" in out
    assert (runs / "x" / "experiment.json").exists()
    assert (runs / "x" / "report.md").exists()
    assert (runs / "x_baseline_v1" / "summary.json").exists()
    stamp = (runs / "x_noop_v0" / "summary.json").stat().st_mtime_ns
    assert cli.main(argv) == 0  # 完了済みは再利用され、書き換わらない
    assert (runs / "x_noop_v0" / "summary.json").stat().st_mtime_ns == stamp


def test_CLI_experimentは不正な構成で終了コード2(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["experiment", "--configs", "baseline,wat"]) == 2
    assert "構成の指定が不正" in capsys.readouterr().err


def test_CLI_experimentのAPIキー未設定のLLM構成は失敗として記録され他は続行する(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "experiment",
        "--name",
        "k",
        "--scenario",
        "heatwave",
        "--hours",
        "3",
        "--variants",
        "2",
        "--configs",
        "baseline,llm:openai:gpt-4o-mini:v9",
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "OPENAI_API_KEY" in out
    assert "baseline" in out


def test_CLI_telemetryは使用量を表示する(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--scenario", "normal", "--hours", "3", "--run-id", "t"]) == 0
    capsys.readouterr()
    assert cli.main(["telemetry", "t"]) == 0
    out = capsys.readouterr().out
    assert "3ステップ" in out
    assert "LLM呼び出し 0" in out
    assert RunStore(runs / "t").read_telemetry()


def test_CLI_telemetryは存在しない実行で終了コード2(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["telemetry", "none"]) == 2
    assert "見つかりません" in capsys.readouterr().err


def test_CLI_traceはAPIキーが無ければ無効の理由を表示して続行する(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        cli.main(["run", "--scenario", "normal", "--hours", "2", "--run-id", "tr", "--trace"]) == 0
    )
    err = capsys.readouterr().err
    assert "無効" in err
    assert "APIキー" in err
    assert "LANGSMITH_TRACING" not in __import__("os").environ


def test_CLI_traceはキーがあれば外部送信の警告を表示する(
    runs: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.setenv("LANGSMITH_API_KEY", "ls-test-key")
    monkeypatch.setattr(os, "environ", dict(os.environ))  # 実際のプロセス環境を汚さない
    assert (
        cli.main(["run", "--scenario", "normal", "--hours", "1", "--run-id", "tw", "--trace"]) == 0
    )
    err = capsys.readouterr().err
    assert "プロンプトと応答" in err
    assert "ls-test-key" not in err


def test_CLI_suiteは状況ごとに集計して保存し_状況の絞り込みもできる(
    runs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "experiment",
        "--suite",
        "stress",
        "--name",
        "ss",
        "--conditions",
        "wet_rain,tradeoff",
        "--variants",
        "2",
        "--configs",
        "baseline,noop",
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "### wet_rain（" in out
    assert "### tradeoff（" in out
    assert "### scarce_water" not in out
    assert "全状況の合計" in out
    assert (runs / "ss" / "suite.json").exists()
    assert (runs / "ss_wet_rain_baseline_v1" / "summary.json").exists()


def test_CLI_suiteは不明な状況で終了コード2(runs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["experiment", "--suite", "stress", "--conditions", "nope"]) == 2
    assert "不明な状況" in capsys.readouterr().err


def test_CLI_promptにfreeを選べる(runs: Path) -> None:
    from langchain_core.messages import AIMessage

    from vmgs.agent.fake import ScriptedChatModel
    from vmgs.agent.prompts import PROMPTS

    model = ScriptedChatModel(
        responses=[AIMessage(content='{"action_summary": "何もしない"}')], cycle=True
    )
    import pytest as _pytest

    mp = _pytest.MonkeyPatch()
    mp.setattr(cli, "make_chat_model", lambda settings: model)
    try:
        argv = [
            "run",
            "--scenario",
            "normal",
            "--hours",
            "1",
            "--agent",
            "llm",
            "--prompt",
            "free",
            "--run-id",
            "pf",
        ]
        assert cli.main(argv) == 0
    finally:
        mp.undo()
    assert model.inputs[0][0].content == PROMPTS["free"]


def test_プロンプト指定のないllm構成は既定のv15なので条件観測を有効にする() -> None:
    assert ConfigSpec("llm", "ollama", "qwen3:14b", None).condition_observations is True
    assert ConfigSpec("llm", "ollama", "qwen3:14b", "v9").condition_observations is False
    assert ConfigSpec("baseline", None, None, None).condition_observations is False
