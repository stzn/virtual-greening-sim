"""CLI のヘルプ。"""

import pytest
from pytest_bdd import scenarios, then, when

from vmgs.cli import main

scenarios("cli/help.feature")

COMMANDS = ["turn", "run", "telemetry", "experiment", "metrics", "compare", "dispatch"]


@pytest.fixture
def helps() -> dict[str, tuple[int | str | None, str]]:
    return {}


@when("各サブコマンドを --help 付きで実行する")
def _w_help(
    helps: dict[str, tuple[int | str | None, str]], capsys: pytest.CaptureFixture[str]
) -> None:
    for c in COMMANDS:
        try:
            main([c, "--help"])
            code: int | str | None = 0
        except SystemExit as e:
            code = e.code
        helps[c] = (code, capsys.readouterr().out)


@then("全てのサブコマンドが正常終了し、使い方が表示される")
def _t_ok(helps: dict[str, tuple[int | str | None, str]]) -> None:
    assert set(helps) == set(COMMANDS)
    assert all(code == 0 and "usage" in out for code, out in helps.values())
