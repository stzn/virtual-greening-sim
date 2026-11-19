"""地図アプリ（tokyo-green-simulator）とシミュレーター（本リポジトリ）のデータ契約 v1。

- stage.json   : 地図アプリ → シミュレーター。候補地点ごとの、時間別の日射遮蔽率・天空率・緑の文脈・需要
- dispatch.json: シミュレーター → 地図アプリ。ポッドの時刻別の位置・行動と、地点ごとの快適度（PET）
- jitai-summary.json: シミュレーター → 地図アプリ。JITAI の判定・介入・見送りを、徒歩圏の区画 × 時刻で数えた集計値

正本はこのモジュール。`uv run python -m vmgs.contracts` で contracts/*.schema.json を書き出し、
tokyo-green-simulator の contracts/ へ同じファイルを写す（両リポジトリのテストが同じサンプルで検証する）。
"""

import datetime as dt
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1"
SCHEMA_FILES = {
    "stage": "stage.schema.json",
    "dispatch": "dispatch.schema.json",
    "jitai_summary": "jitai-summary.schema.json",
}

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
    """地点まわりの緑（地点データの computeGreenContext と同じ数え方）。"""

    radius_m: float = Field(gt=0)
    trees_all: int = Field(ge=0)
    trees_tall: int = Field(ge=0)
    parks_count: int = Field(ge=0)
    parks_area_m2: float = Field(ge=0)
    nearest_park_m: float | None = Field(ge=0, description="公園がなければ null")


class SpotHour(_Strict):
    hour: Hour
    shade_fraction: Fraction = Field(
        description="建物と街路樹の樹冠による日射遮蔽率（0=日向、1=完全な日影）"
    )


class DemandHour(_Strict):
    hour: Hour
    weight: float = Field(
        ge=0, description="過緊張の需要の相対値（集計値のみ。個人の計測値は入れない）"
    )


SpotKind = Literal["sidewalk", "plaza", "park_edge", "rooftop"]


Shelter = Literal["open", "covered"]


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
    shelter: Shelter = Field(
        default="open",
        description="屋根の有無（版 2 から）。covered は地下通路・屋内の通路・アーケードの上の地点",
    )
    shelter_network: str | None = Field(
        default=None,
        description="屋根付きの通路の網の番号（版 2 から）。同じ番号の地点どうしは濡れずに行き来できる",
    )
    shelter_access: str | None = Field(
        default=None,
        description=(
            "屋根なしの地点から近く（50 m 以内）の出入口で入れる、屋根付きの通路の網の番号（版 2 から）"
        ),
    )

    @model_validator(mode="after")
    def _hours_unique(self) -> "StageSpot":
        _unique([h.hour for h in self.hourly], f"地点 {self.id} の hourly の時刻")
        _unique([d.hour for d in self.demand], f"地点 {self.id} の demand の時刻")
        return self

    @model_validator(mode="after")
    def _network_iff_covered(self) -> "StageSpot":
        if (self.shelter == "covered") != (self.shelter_network is not None):
            raise ValueError(
                f"地点 {self.id}: 屋根付き（covered）の地点だけが shelter_network を持つ"
            )
        if self.shelter == "covered" and self.shelter_access is not None:
            raise ValueError(
                f"地点 {self.id}: 屋根付き（covered）の地点は網の上にいるので shelter_access を持てない"
            )
        return self


class Stage(_Strict):
    schema_version: Literal["1", "2"] = Field(
        description=(
            "1: 屋根の情報なし。2: 地点に shelter・shelter_network・shelter_access を足した版"
        )
    )
    area: Area
    date: dt.date = Field(description="日影を計算した日（太陽位置の基準）")
    timezone: Literal["Asia/Tokyo"]
    generator: str = Field(
        description="書き出した地図アプリの版など（例: tokyo-green-simulator@abc1234）"
    )
    spots: list[StageSpot] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_unique(self) -> "Stage":
        _unique([s.id for s in self.spots], "地点 ID")
        return self

    @model_validator(mode="after")
    def _v1_has_no_shelter(self) -> "Stage":
        if self.schema_version == "1":
            bad = [s.id for s in self.spots if s.shelter != "open" or s.shelter_access]
            if bad:
                raise ValueError(
                    f"版 1 の地点データは屋根付きの地点・網への入口を持てない（版 2 にする）: {bad}"
                )
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
    generator: str = Field(
        description="書き出したシミュレーターの版と配車方式（例: vmgs@abc1234 greedy）"
    )
    weather_scenario: str = Field(description="使った気象（例: heatwave）")
    pods: list[Pod] = Field(min_length=1)
    comfort: list[SpotComfort] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique(self) -> "Dispatch":
        _unique([p.id for p in self.pods], "ポッド ID")
        return self


def dispatch_errors_against_stage(dispatch: Dispatch, stage: Stage) -> list[str]:
    """配車が地点データと食い違う点を挙げる（区域・日付・未知の地点・地点の容量超過）。"""
    errors: list[str] = []
    if dispatch.stage_ref.area_id != stage.area.id:
        errors.append(f"区域が違う: 配車 {dispatch.stage_ref.area_id}、地点データ {stage.area.id}")
    if dispatch.stage_ref.date != stage.date:
        errors.append(f"日付が違う: 配車 {dispatch.stage_ref.date}、地点データ {stage.date}")
    capacity = {s.id: s.capacity_pods for s in stage.spots}
    used: dict[tuple[str, int], int] = {}
    for pod in dispatch.pods:
        for step in pod.timeline:
            if step.spot_id not in capacity:
                errors.append(
                    f"地点データにない地点: ポッド {pod.id} の {step.hour} 時 {step.spot_id}"
                )
                continue
            key = (step.spot_id, step.hour)
            used[key] = used.get(key, 0) + 1
    for (spot_id, hour), n in sorted(used.items()):
        if n > capacity[spot_id]:
            errors.append(f"容量超過: {spot_id} の {hour} 時に {n} 台（容量 {capacity[spot_id]}）")
    for c in dispatch.comfort:
        if c.spot_id not in capacity:
            errors.append(f"地点データにない地点の快適度: {c.spot_id}")
    return errors


class ZoneGrid(_Strict):
    """区画の切り方。区域の南西の角を原点に、東西・南北に zone_m ごとの升目（番号は 0 から）。"""

    origin_lon: Lon
    origin_lat: Lat
    zone_m: float = Field(gt=0, description="区画の一辺 [m]（徒歩圏に合わせる）")


class ZoneHour(_Strict):
    zone: tuple[int, int] = Field(description="区画の番号 [東西, 南北]")
    hour: Hour
    decisions: int = Field(ge=1, description="判定の件数（過緊張を検知した判定点）")
    interventions: int = Field(ge=0)
    skipped_calendar: int = Field(ge=0, description="予定の段階で見送った件数")
    skipped_pods: int = Field(ge=0, description="ポッドの段階で見送った件数")
    skipped_pet: int = Field(ge=0, description="天候（雨・PET）の段階で見送った件数")

    @model_validator(mode="after")
    def _sum(self) -> "ZoneHour":
        parts = self.interventions + self.skipped_calendar + self.skipped_pods + self.skipped_pet
        if parts != self.decisions:
            raise ValueError(f"判定 {self.decisions} 件が、介入と見送りの和 {parts} 件と違う")
        return self


class JitaiTotals(_Strict):
    decisions: int = Field(ge=0)
    interventions: int = Field(ge=0)


class RecoveryInfo(_Strict):
    enabled: bool
    carryover_minutes: float = Field(ge=0)


class JitaiSummary(_Strict):
    """JITAI の集計（シミュレーター → 地図アプリ）。min_workers 人未満の区画・時刻は入れない。個人の記録は持たない。"""

    schema_version: Literal["1"]
    stage_ref: StageRef
    generator: str
    weather_scenario: str
    workers: int = Field(ge=1, description="合成のワーカーの人数")
    min_workers: int = Field(ge=1)
    grid: ZoneGrid
    totals: JitaiTotals
    recovery: RecoveryInfo
    zones: list[ZoneHour] = Field(default_factory=list)

    @model_validator(mode="after")
    def _zones_unique(self) -> "JitaiSummary":
        _unique([(z.zone, z.hour) for z in self.zones], "区画と時刻")
        return self


def jitai_summary_errors_against_stage(summary: JitaiSummary, stage: Stage) -> list[str]:
    """JITAI の集計が地点データと食い違う点を挙げる（区域・日付・区画の原点）。"""
    errors: list[str] = []
    if summary.stage_ref.area_id != stage.area.id:
        errors.append(f"区域が違う: 集計 {summary.stage_ref.area_id}、地点データ {stage.area.id}")
    if summary.stage_ref.date != stage.date:
        errors.append(f"日付が違う: 集計 {summary.stage_ref.date}、地点データ {stage.date}")
    west, south, _east, _north = stage.area.bbox
    if (summary.grid.origin_lon, summary.grid.origin_lat) != (west, south):
        errors.append(
            f"区画の原点が区域の南西の角と違う: 集計 ({summary.grid.origin_lon}, "
            f"{summary.grid.origin_lat})、地点データ ({west}, {south})"
        )
    return errors


def generated_schemas() -> dict[str, dict[str, Any]]:
    return {
        "stage": Stage.model_json_schema(),
        "dispatch": Dispatch.model_json_schema(),
        "jitai_summary": JitaiSummary.model_json_schema(),
    }


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
