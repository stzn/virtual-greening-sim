"""トレードオフ等の再現用プリセット（気象シナリオ + 初期条件）。"""

from vmgs.sim.simulate import SimConfig

PRESETS: dict[str, SimConfig] = {
    # 14時・猛暑・歩道C・水分25%・タンク残りわずか。
    # 「Cで散水を続ける」か「日影Bへ退避する」か、水と電池が足りない中での判断を迫る。
    "tradeoff": SimConfig(
        scenario="heatwave",
        hours=10,
        start_hour=14,
        spot="C",
        initial_moisture=25.0,
        initial_tank_ml=1500,
    ),
}
