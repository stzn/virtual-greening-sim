"""舞台（tokyo-green-simulator）と役者（本リポジトリ）のデータ契約の振る舞い。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from vmgs.contracts import (
    SCHEMA_FILES,
    Dispatch,
    Stage,
    dispatch_errors_against_stage,
    generated_schemas,
)

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def _load(name: str) -> dict:  # type: ignore[type-arg]
    data: dict = json.loads((CONTRACTS / "fixtures" / name).read_text(encoding="utf-8"))  # type: ignore[type-arg]
    return data


def test_正本のスキーマはモデルから生成したものと一致する() -> None:
    # Given モデルから生成したスキーマ / When 正本のファイルと比べる / Then 一致する（更新漏れを防ぐ）
    for name, schema in generated_schemas().items():
        committed = json.loads((CONTRACTS / SCHEMA_FILES[name]).read_text(encoding="utf-8"))
        assert committed == schema, (
            f"{SCHEMA_FILES[name]} を `uv run python -m vmgs.contracts` で更新する"
        )


def test_舞台のサンプルを読み込める() -> None:
    stage = Stage.model_validate(_load("stage.sample.json"))
    assert stage.schema_version == "1"
    assert stage.area.id == "otemachi"
    assert len(stage.spots) >= 3
    assert all(len(s.hourly) == 24 for s in stage.spots)


def test_配車のサンプルは舞台のサンプルと整合する() -> None:
    stage = Stage.model_validate(_load("stage.sample.json"))
    dispatch = Dispatch.model_validate(_load("dispatch.sample.json"))
    assert dispatch_errors_against_stage(dispatch, stage) == []


def test_日射遮蔽率が0から1の外なら拒否する() -> None:
    data = _load("stage.sample.json")
    data["spots"][0]["hourly"][12]["shade_fraction"] = 1.2
    with pytest.raises(ValidationError):
        Stage.model_validate(data)


def test_同じ時刻が重なる地点は拒否する() -> None:
    data = _load("stage.sample.json")
    data["spots"][0]["hourly"][1]["hour"] = 0
    with pytest.raises(ValidationError):
        Stage.model_validate(data)


def test_地点IDの重複は拒否する() -> None:
    data = _load("stage.sample.json")
    data["spots"][1]["id"] = data["spots"][0]["id"]
    with pytest.raises(ValidationError):
        Stage.model_validate(data)


def test_未知の版は拒否する() -> None:
    data = _load("stage.sample.json")
    data["schema_version"] = "2"
    with pytest.raises(ValidationError):
        Stage.model_validate(data)


def test_舞台にない地点や別の区域を指す配車は誤りとして挙げる() -> None:
    stage = Stage.model_validate(_load("stage.sample.json"))
    data = _load("dispatch.sample.json")
    data["pods"][0]["timeline"][0]["spot_id"] = "NOPE"
    data["stage_ref"]["area_id"] = "shibuya"
    errors = dispatch_errors_against_stage(Dispatch.model_validate(data), stage)
    assert any("NOPE" in e for e in errors)
    assert any("shibuya" in e for e in errors)
