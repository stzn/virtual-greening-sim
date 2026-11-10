"""エージェント向け5ツール。World を操作する薄いアダプタで、ロジックは持たない。

どのツールも例外を送出せず、失敗は {"ok": False, "error": {...}} で返す。
description はそのままLLMへのプロンプトになるため、フェーズ4で磨く。
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from vmgs.tools.schemas import (
    GetPlantStateArgs,
    GetWeatherArgs,
    IrrigateArgs,
    LogResearchNoteArgs,
    MoveToSpotArgs,
)
from vmgs.weather.models import WeatherRecord
from vmgs.weather.providers import WeatherUnavailableError
from vmgs.world.world import World, WorldError

Handler = Callable[[Any], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Handler


def _error(code: str, message: str, **details: object) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message, "details": details}}


# 散水の失敗の多くは「50ml 未満の要求」だった。残量が少なくても 50 を指定すれば残量まで散水できるので、
# 失敗は残したまま（再判断のきっかけを消さずに）、次の一手をエラーの message に添える（details では読まれにくかった）。
IRRIGATE_MIN_HINT = (
    "散水量は50ml以上で指定します。タンク残量が50ml未満のときは、50を指定すると残量まで散水します"
)


# タンクが空のときは「50 を指定すれば残量まで散水できる」は誤り（50 でも INSUFFICIENT_WATER になる）。
# 空のタンクで 0ml を要求した失敗のあと「散水した（0ml）」と要約する例があったため、空だと伝える。
IRRIGATE_EMPTY_HINT = (
    "タンクが空なので散水できません。この時間は散水せず、ほかの行動の要否を判断してください"
)


def _below_irrigate_min(name: str, errors: list[Any]) -> bool:
    return name == "irrigate" and any(
        e["loc"] == ("amount_ml",) and e["type"] == "greater_than_equal" for e in errors
    )


def _weather_dict(w: WeatherRecord) -> dict[str, Any]:
    return {
        "time": w.time.isoformat(timespec="minutes"),
        "temperature_c": w.temperature_c,
        "humidity_pct": w.humidity_pct,
        "shortwave_w_m2": w.shortwave_w_m2,
        "wind_ms": w.wind_ms,
        "precipitation_mm": w.precipitation_mm,
        "precipitation_prob_pct": w.precipitation_prob_pct,
    }


def _r(v: float) -> float:
    return round(v, 2)


class ToolBox:
    def __init__(self, world: World) -> None:
        self.world = world
        specs = [
            ToolSpec(
                "get_weather",
                "指定座標の現在時刻の気象と、先の数時間の予報（気温・日射・風速・湿度・降水量・降水確率）を取得する。"
                "移動や散水を決める前に、雨の予報や日射のピークを確認するために使う。",
                GetWeatherArgs,
                self._get_weather,
            ),
            ToolSpec(
                "get_plant_state",
                "植物ユニットの現在の状態（土壌水分・健康度・現在スポット・水タンク残量・バッテリー残量・ステータス）を取得する。"
                "副作用はない。判断の前に必ず確認する。",
                GetPlantStateArgs,
                self._get_plant_state,
            ),
            ToolSpec(
                "move_to_spot",
                "ユニットを別スポット（A:芝生広場 / B:ビルの谷間 / C:アスファルト歩道）へ移動する。"
                "距離に応じてバッテリーを5〜10%消費する。すでにいるスポットへは移動できない。"
                "直前の移動元へ短時間（2時間以内）で戻る往復は拒否される。さらに、直近12時間に移動できるのは2回までで、超えると MOVE_BUDGET_EXCEEDED で拒否される（どちらも健康度が低い緊急時を除く）。残りの移動予算は get_plant_state で分かる。",
                MoveToSpotArgs,
                self._move_to_spot,
            ),
            ToolSpec(
                "irrigate",
                "現在のスポットで散水する（50〜2000ml）。土壌水分が回復し、気化冷却で周囲を冷やす。"
                "タンク残量を消費する。タンク残量が50ml未満のときも、50を指定すれば残量まで散水する。"
                "降雨が直近に予報されているときは使わない。",
                IrrigateArgs,
                self._irrigate,
            ),
            ToolSpec(
                "log_research_note",
                "研究メモをMarkdownに保存する。気づき・仮説・次の方針を記録するために使う（1000字以内）。",
                LogResearchNoteArgs,
                self._log_note,
            ),
        ]
        self._specs = {s.name: s for s in specs}

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        spec = self._specs.get(name)
        if spec is None:
            return _error("UNKNOWN_TOOL", f"ツール {name!r} は存在しません")
        try:
            args = spec.args_model(**kwargs)
        except ValidationError as e:
            errors = e.errors(include_url=False, include_context=False, include_input=False)
            message = "引数が不正です"
            if _below_irrigate_min(name, errors):
                tank_empty = self.world.plant.water_tank_ml <= 0
                message += f"。{IRRIGATE_EMPTY_HINT if tank_empty else IRRIGATE_MIN_HINT}"
            return _error("VALIDATION_ERROR", message, errors=errors)
        try:
            return {"ok": True, **spec.handler(args)}
        except WorldError as e:
            return _error(e.code, e.message, **e.details)
        except WeatherUnavailableError as e:
            return _error("WEATHER_UNAVAILABLE", str(e))

    # --- ハンドラ ---
    def _get_weather(self, a: GetWeatherArgs) -> dict[str, Any]:
        w = self.world
        now = w.clock.now
        return {
            "location": {"lat": a.lat, "lon": a.lon},
            "time": now.strftime("%H:%M"),
            "datetime": now.isoformat(timespec="minutes"),
            "current": _weather_dict(w.current_weather()),
            "forecast": [_weather_dict(f) for f in w.weather.forecast(now, a.hours_ahead)],
        }

    def _get_plant_state(self, _: GetPlantStateArgs) -> dict[str, Any]:
        s = self.world.state()
        out: dict[str, Any] = {
            "sim_time": s.sim_time.isoformat(timespec="minutes"),
            "spot_id": s.spot_id,
            "spot_name": s.spot_name,
            "soil_moisture_pct": _r(s.soil_moisture_pct),
            "health": _r(s.health),
            "water_tank_ml": s.water_tank_ml,
            "battery_pct": _r(s.battery_pct),
            "status": s.status,
            "heat_mitigation_score": _r(s.heat_mitigation_score),
            "moves_in_window": s.moves_in_window,
            "move_budget_remaining": s.move_budget_remaining,
            "last_move": (
                {
                    "from": s.last_move.from_spot,
                    "to": s.last_move.to_spot,
                    "hours_ago": s.last_move.hours_ago,
                }
                if s.last_move
                else None
            ),
        }
        planning = self.world.planning_observations()
        if planning is not None:
            out.update(planning)
        conditions = self.world.condition_observations()
        if conditions is not None:
            out.update(conditions)
        return out

    def _move_to_spot(self, a: MoveToSpotArgs) -> dict[str, Any]:
        r = self.world.move_to(a.spot_id)
        return {
            "new_spot_id": r.new_spot_id,
            "battery_used_pct": _r(r.battery_used_pct),
            "battery_pct": _r(r.battery_pct),
        }

    def _irrigate(self, a: IrrigateArgs) -> dict[str, Any]:
        r = self.world.irrigate(a.amount_ml)
        out: dict[str, Any] = {
            "soil_moisture_pct": _r(r.soil_moisture_pct),
            "water_tank_ml": r.water_tank_ml,
            "cooling_c": _r(r.cooling_c),
            "heat_mitigation_gain": _r(r.heat_mitigation_gain),
        }
        if r.truncated:
            # 切り詰めたときだけキーを足す（切り詰めが無い結果は従来と同一。過去ログ・比較との互換のため）
            out["truncated"] = True
            out["requested_ml"] = r.requested_ml
            out["applied_ml"] = r.applied_ml
            out["message"] = (
                f"要求 {r.requested_ml}ml はタンク残量を超えたため、残量の {r.applied_ml}ml に切り詰めて散水しました"
                + ("。タンクは空です" if r.water_tank_ml == 0 else "")
            )
        return out

    def _log_note(self, a: LogResearchNoteArgs) -> dict[str, Any]:
        path = self.world.log_note(a.text, a.tags)
        return {"saved_to": str(path)}
