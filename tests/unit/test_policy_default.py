"""配車の方式の既定値（coverage）の振る舞い。"""

import argparse

import pytest

from vmgs.fleet.cli import add_dispatch_parser
from vmgs.jitai.loop import add_loop_parser
from vmgs.jitai.sweep import add_sweep_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    add_dispatch_parser(sub)
    add_loop_parser(sub)
    add_sweep_parser(sub)
    return parser.parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [["dispatch", "--stage", "stage.json"], ["loop", "--stage", "stage.json", "--out-dir", "out"]],
)
def test_方式を指定しなければ_配車と通し道は_coverage_で配車する(argv: list[str]) -> None:
    assert _parse(argv).policy == "coverage"


def test_方式を指定しなければ_sweep_は_coverage_だけを回す() -> None:
    assert _parse(["sweep", "--stage", "stage.json", "--out-dir", "out"]).policies == ["coverage"]


def test_greedy_は指定すれば今までどおり選べる() -> None:
    assert _parse(["dispatch", "--stage", "s.json", "--policy", "greedy"]).policy == "greedy"
    assert (
        _parse(["loop", "--stage", "s.json", "--out-dir", "o", "--policy", "greedy"]).policy
        == "greedy"
    )
