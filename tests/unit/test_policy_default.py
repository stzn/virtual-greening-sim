"""配車の方式の既定値（coverage）の振る舞い。"""

import argparse

import pytest

from vmgs.fleet.cli import add_dispatch_parser
from vmgs.jitai.cli import add_jitai_parser
from vmgs.jitai.loop import add_loop_parser
from vmgs.jitai.sweep import add_sweep_parser


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    add_dispatch_parser(sub)
    add_jitai_parser(sub)
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


@pytest.mark.parametrize(
    "argv",
    [
        ["jitai", "--stage", "s.json", "--dispatch", "d.json"],
        ["loop", "--stage", "s.json", "--out-dir", "o"],
        ["sweep", "--stage", "s.json", "--out-dir", "o"],
    ],
)
def test_雨よけは指定しなければ有り_no_rain_coverで無しにできる(argv: list[str]) -> None:
    # Given ポッドに雨よけを付ける設計を既定にした / When 何も指定しない / Then 雨よけ有り
    assert _parse(argv).rain_cover is True
    # When --no-rain-cover / Then 雨よけ無し（比べるとき）
    assert _parse([*argv, "--no-rain-cover"]).rain_cover is False
    assert _parse([*argv, "--rain-cover"]).rain_cover is True


@pytest.mark.parametrize(
    "argv",
    [
        ["jitai", "--stage", "s.json", "--dispatch", "d.json"],
        ["loop", "--stage", "s.json", "--out-dir", "o"],
        ["sweep", "--stage", "s.json", "--out-dir", "o"],
    ],
)
def test_匿名にする人数と区画は指定しなければ3人と400m_指定すれば変えられる(
    argv: list[str],
) -> None:
    args = _parse(argv)
    assert (args.min_workers, args.zone_m) == (3, 400.0)
    args = _parse([*argv, "--min-workers", "2", "--zone-m", "800"])
    assert (args.min_workers, args.zone_m) == (2, 800.0)


@pytest.mark.parametrize(
    "argv",
    [
        ["dispatch", "--stage", "s.json"],
        ["loop", "--stage", "s.json", "--out-dir", "o"],
        ["sweep", "--stage", "s.json", "--out-dir", "o"],
    ],
)
def test_配車の目的は指定しなければPETの改善_countを選べる(argv: list[str]) -> None:
    assert _parse(argv).objective == "relief"
    assert _parse([*argv, "--objective", "count"]).objective == "count"
