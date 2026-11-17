"""舞台（tokyo-green-simulator）と役者（本リポジトリ）のデータ契約 v1。

- stage.json   : 舞台 → 役者。候補地点ごとの、時間別の日射遮蔽率・天空率・緑の文脈・需要
- dispatch.json: 役者 → 舞台。ポッドの時刻別の位置・行動と、地点ごとの快適度（PET）

正本はこのモジュール。`uv run python -m vmgs.contracts` で contracts/*.schema.json を書き出し、
tokyo-green-simulator の contracts/ へ同じファイルを写す（両リポジトリのテストが同じサンプルで検証する）。
"""

import datetime as dt
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"
SCHEMA_FILES = {"stage": "stage.schema.json", "dispatch": "dispatch.schema.json"}

Hour = Annotated[int, Field(ge=0, le=23)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
Lat = Annotated[float, Field(ge=-90.0, le=90.0)]
Lon = Annotated[float, Field(ge=-180.0, le=180.0)]


class _Strict(BaseModel):
    # 知らないキーは拒否する（綴りの誤りを黙って読み飛ばさない）
    model_config = ConfigDict(extra="forbid", frozen=True)


def _unique(values: list[Any], what: str) -> None:
    dup = sorted({v for v in values if values.count(v) > 1})
    if dup:
        raise ValueError(f"{what}が重複している: {dup}")


class Area(_Strict):
    id: str = Field(min_length=1, description="区域の ID（例: otemachi）")
    name: str
    bbox: tuple[Lon, Lat, Lon, Lat] = Field(description="[西経度, 南緯度, 東経度, 北緯度]")


class GreenContext(_Strict):
    """地点まわりの緑（舞台の computeGreenContext と同じ数え方）。"""

    radius_m: float = Field(gt=0)
    trees_all: int = Field(ge=0)
    trees_tall: int = Field(ge=0)
    parks_count: int = Field(ge=0)
    parks_area_m2: float = Field(ge=0)
    nearest_park_m: float | None = Field(ge=0, description="公園がなければ null")


class SpotHour(_Strict):
    hour: Hour
    shade_fraction: Fraction = Field(description="建物による日射遮蔽率（0=日向、1=完全な日影）")


class DemandHour(_Strict):
    hour: Hour
    weight: float = Field(
        ge=0, description="過緊張の需要の相対値（集計値のみ。個人の計測値は入れない）"
    )


SpotKind = Literal["sidewalk", "plaza", "park_edge", "rooftop"]


class StageSpot(_Strict):
    id: str = Field(min_length=1)
    name: str
    lat: Lat
    lon: Lon
    kind: SpotKind
    capacity_pods: int = Field(ge=1, description="同時に置けるポッドの数")
    sky_view_factor: Fraction = Field(description="天空率（平均放射温度の計算に使う）")
    green: GreenContext
    hourly: list[SpotHour] = Field(min_length=1)
    demand: list[DemandHour] = Field(default_factory=list)

    @model_validator(mode="after")
    def _hours_unique(self) -> "StageSpot":
        _unique([h.hour for h in self.hourly], f"地点 {self.id} の hourly の時刻")
        _unique([d.hour for d in self.demand], f"地点 {self.id} の demand の時刻")
        return self


class Stage(_Strict):
    schema_version: Literal["1"]
    area: Area
    date: dt.date = Field(description="日影を計算した日（太陽位置の基準）")
    timezone: Literal["Asia/Tokyo"]
    generator: str = Field(
        description="書き出した舞台の版など（例: tokyo-green-simulator@abc1234）"
    )
    spots: list[StageSpot] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "Stage":
        _unique([s.id for s in self.spots], "地点 ID")
        return self


PodAction = Literal["stay", "move", "irrigate"]


class PodStep(_Strict):
    hour: Hour
    spot_id: str
    action: PodAction
    plant_health: float = Field(ge=0, le=100)


class Pod(_Strict):
    id: str = Field(min_length=1)
    timeline: list[PodStep] = Field(min_length=1)

    @model_validator(mode="after")
    def _hours_unique(self) -> "Pod":
        _unique([s.hour for s in self.timeline], f"ポッド {self.id} の時刻")
        return self


class SpotComfort(_Strict):
    spot_id: str
    hour: Hour
    pet_c: float = Field(description="生理学的等価温度 PET [℃]")
    pods_present: int = Field(ge=0)


class StageRef(_Strict):
    area_id: str
    date: dt.date


class Dispatch(_Strict):
    schema_version: Literal["1"]
    stage_ref: StageRef
    generator: str = Field(description="書き出した役者の版と配車方式（例: vmgs@abc1234 greedy）")
    weather_scenario: str = Field(description="使った気象（例: heatwave）")
    pods: list[Pod] = Field(min_length=1)
    comfort: list[SpotComfort] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique(self) -> "Dispatch":
        _unique([p.id for p in self.pods], "ポッド ID")
        return self


def dispatch_errors_against_stage(dispatch: Dispatch, stage: Stage) -> list[str]:
    """配車が舞台と食い違う点を挙げる（区域・日付・未知の地点・地点の容量超過）。"""
    errors: list[str] = []
    if dispatch.stage_ref.area_id != stage.area.id:
        errors.append(f"区域が違う: 配車 {dispatch.stage_ref.area_id}、舞台 {stage.area.id}")
    if dispatch.stage_ref.date != stage.date:
        errors.append(f"日付が違う: 配車 {dispatch.stage_ref.date}、舞台 {stage.date}")
    capacity = {s.id: s.capacity_pods for s in stage.spots}
    used: dict[tuple[str, int], int] = {}
    for pod in dispatch.pods:
        for step in pod.timeline:
            if step.spot_id not in capacity:
                errors.append(f"舞台にない地点: ポッド {pod.id} の {step.hour} 時 {step.spot_id}")
                continue
            key = (step.spot_id, step.hour)
            used[key] = used.get(key, 0) + 1
    for (spot_id, hour), n in sorted(used.items()):
        if n > capacity[spot_id]:
            errors.append(f"容量超過: {spot_id} の {hour} 時に {n} 台（容量 {capacity[spot_id]}）")
    for c in dispatch.comfort:
        if c.spot_id not in capacity:
            errors.append(f"舞台にない地点の快適度: {c.spot_id}")
    return errors


def generated_schemas() -> dict[str, dict[str, Any]]:
    return {"stage": Stage.model_json_schema(), "dispatch": Dispatch.model_json_schema()}


def write_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, schema in generated_schemas().items():
        path = out_dir / SCHEMA_FILES[name]
        path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for p in write_schemas(Path("contracts")):
        print(f"書き出した: {p}")
