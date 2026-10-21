"""余裕の少ない状況（ストレス状況）。

猛暑日24時間では、ベースラインの最終健康度が 100.0±0.0 の天井に張り付き、LLM との差を測れなかった。
ここでは、ベースラインが天井に届かず、「ルールに従うだけ」と「状況を読む」を分けられる状況を集めた。
雨天の2状況は、同じ雨でも初期の水分によって正解が逆になる対になっている。
"""

from dataclasses import dataclass

from vmgs.experiment.variants import BaseConditions


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    scenario: str  # 気象シナリオ
    hours: int
    base: BaseConditions


STRESS_SUITE: dict[str, Condition] = {
    c.name: c
    for c in (
        Condition(
            "scarce_water",
            "猛暑の一日に、水が少ない。早めの退避と節水が要る",
            "heatwave",
            24,
            BaseConditions(moisture=40, tank_ml=1500),
        ),
        Condition(
            "dry_before_rain",
            "雨の予報があるが、いま既に乾燥している。「雨前は散水しない」規則は誤りになる",
            "rainy",
            24,
            BaseConditions(moisture=20),
        ),
        Condition(
            "wet_rain",
            "雨の前から土壌が湿っている。水を足すと過湿になる（わな）",
            "rainy",
            24,
            BaseConditions(moisture=75),
        ),
        Condition(
            "tradeoff",
            "14時・猛暑・歩道C・水分25%・タンク少。水と電池が足りない中での判断",
            "heatwave",
            10,
            BaseConditions(moisture=25, tank_ml=1500, spot="C", start_hour=14),
        ),
    )
}
