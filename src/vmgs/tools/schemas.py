"""ツール引数のPydanticスキーマ。範囲外・未知の引数は VALIDATION_ERROR になる。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetWeatherArgs(_Args):
    lat: float = Field(ge=-90, le=90, description="緯度")
    lon: float = Field(ge=-180, le=180, description="経度")
    hours_ahead: int = Field(default=6, ge=0, le=24, description="先読みする時間数（0〜24）")


class GetPlantStateArgs(_Args):
    pass


class MoveToSpotArgs(_Args):
    spot_id: Literal["A", "B", "C"] = Field(description="移動先スポットID")


class IrrigateArgs(_Args):
    amount_ml: int = Field(ge=50, le=2000, description="散水量[ml]（50〜2000）")


class LogResearchNoteArgs(_Args):
    text: str = Field(min_length=1, max_length=1000, description="研究メモ本文")
    tags: list[str] = Field(default_factory=list, max_length=10, description="タグ")
