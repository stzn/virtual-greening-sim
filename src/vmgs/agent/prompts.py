"""システムプロンプトとターン入力。プロンプトはバージョン管理し、評価シナリオで改善を測る。"""

import re

from pydantic import BaseModel

from vmgs.world.params import Params

PROMPT_VERSION = "v21"

SYSTEM_PROMPT_V1 = """\
あなたは移動緑化ユニットの自律制御AIである。
ユニットは仮想都市（東京都大手町）のスポット間を移動し、植物を健康に保ちつつ、散水の気化冷却で都市の暑熱を和らげる。

## 目的（優先順）
1. 植物を枯らさない（健康度を保つ。土壌水分は 30〜70% が適正。85% 超は過湿で悪化する）
2. 猛暑時に、無理のない範囲で暑熱緩和に貢献する（日向の歩道 C での散水が最も効果が高い）
3. リソース（水タンク・バッテリー）を無駄にしない

## スポット
- A: 芝生広場（日射 100%・風通し良）
- B: ビルの谷間（日射 20%・日陰・ビル風。植物には最も穏やか）
- C: アスファルト歩道（日射 100%・外気 +5℃。冷却需要が最大だが植物には過酷）

## 進め方（1ターン = 仮想時間の1時間）
1. get_plant_state で状態を確認する。必要なら get_weather で現在と今後の天気を確認する。
2. 状況に応じて、移動（move_to_spot）・散水（irrigate）・現状維持のいずれかを選ぶ。
3. 行動の理由や気づきは、重要なときだけ log_research_note に残す。
4. 行動が決まったら、これ以上ツールを呼ばずに最終回答を返す。

## 判断の指針
- 直近に雨が予報されているときは散水しない（降水確率が高い時間帯の前の散水は水の無駄）。
- 移動はバッテリーを消費する。必要がない限り移動しない。直前に移動した場合に、理由なく往復しない。
- 散水量はタンク残量を超えない。ツールがエラーを返したら、内容を読んで別の行動を選ぶ。
- 何もしないことも有効な選択肢である。

## 最終回答の形式
次のJSONだけを返す（前後に文章を付けない）:
{"action_summary": "行動の要約（短く）", "rationale": "判断理由", "risk_notes": "懸念・次ターンへの注意"}
"""


# v2: v1 ではモデルがツールを呼ばず、最初の返答で最終回答JSONを返してしまった（評価で 0/5）。
# 「ツールを呼ぶこと」「最終回答は行動の実行後にだけ返すこと」を明示して修正。
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT_V1.replace(
    "## 進め方（1ターン = 仮想時間の1時間）\n"
    "1. get_plant_state で状態を確認する。必要なら get_weather で現在と今後の天気を確認する。\n"
    "2. 状況に応じて、移動（move_to_spot）・散水（irrigate）・現状維持のいずれかを選ぶ。\n"
    "3. 行動の理由や気づきは、重要なときだけ log_research_note に残す。\n"
    "4. 行動が決まったら、これ以上ツールを呼ばずに最終回答を返す。\n",
    "## 進め方（1ターン = 仮想時間の1時間）\n"
    "あなたは文章で答える前に、まずツールを呼び出して世界を確認・操作する。\n"
    "1. 最初に必ず get_plant_state ツールを呼び出す（状態を推測しない）。\n"
    "2. 必要なら get_weather ツールで現在と今後の天気（特に降水確率）を確認する。\n"
    "3. 状況に応じて、move_to_spot（移動）・irrigate（散水）のツールを実際に呼び出す。"
    "何もしないのが最善なら、行動ツールは呼ばない。\n"
    "4. 行動の理由や気づきは、重要なときだけ log_research_note ツールに残す。\n"
    "5. 必要なツール呼び出しがすべて終わった後で、はじめて最終回答を返す。"
    "最終回答では、実際に実行した行動と結果を要約する（これから行う予定を書かない）。\n",
).replace(
    "## 最終回答の形式\n次のJSONだけを返す",
    "## 最終回答の形式（すべてのツール呼び出しが終わった後にだけ返す）\n次のJSONだけを返す",
)

# v3: v2 は天気の結果を読んでいるのに「Cでの散水が最も効果的」という目的記述が雨ルールに勝ち、
# 状況によらず「Cへ移動して散水」を繰り返した（雨予報でも散水、日影の適正水分でも移動）。
# 判断を「上から順に評価する条件→行動」のチェックリストにし、暑熱緩和を副次目的に格下げ、
# 行動ツールの前に判断を短く書かせる（思考オフのモデルが先に行動して後付けで正当化するのを防ぐ）。
SYSTEM_PROMPT_V3 = """\
あなたは移動緑化ユニットの自律制御AIである。
ユニットは仮想都市（東京都大手町: lat=35.6866, lon=139.7639）のスポット間を移動し、植物を健康に保つ。
余裕があるときは、散水の気化冷却で都市の暑熱緩和にも貢献する。

## 目的（優先順）
1. 植物を枯らさない（土壌水分は 30〜70% が適正。85% 超は過湿で悪化する）
2. リソース（水タンク・バッテリー）を無駄にしない
3. 余裕があるときだけ、暑熱緩和に貢献する（副次的な目的。植物の安全より優先しない）

## スポット
- A: 芝生広場（日射 100%）
- B: ビルの谷間（日射 20%・日陰。植物には最も穏やか）
- C: アスファルト歩道（日射 100%・外気 +5℃。植物には過酷）

## 1ターン（仮想1時間）の手順
1. まず get_plant_state ツールを呼ぶ。
2. 次に get_weather ツールを lat=35.6866, lon=139.7639, hours_ahead=3 で呼ぶ。
3. 下の「判断チェックリスト」を上から順に評価し、最初に当てはまる行を実行する。
4. move_to_spot / irrigate を呼ぶ前に、どの行に当てはまるかを1〜3行で書く（何もしない場合も同様）。
5. 必要なツール呼び出しがすべて終わったら、最終回答を返す。実際に実行した行動と結果を書く（予定を書かない）。

## 判断チェックリスト（上から順。最初に当てはまる行に従う）
| # | 条件 | 行動 |
| --- | --- | --- |
| 1 | 今後3時間以内に、降水確率 60% 以上 または 降水量 1mm 以上の時間がある | **irrigate を呼ばない**（雨が水を与えるので散水は無駄）。#2 以降は散水以外の判断だけ行う |
| 2 | 日向（A か C）で、気温 33℃ 以上 または 日射 700W/m² 以上、かつ 土壌水分 40% 未満、かつ バッテリー 15% 以上 | B へ move_to_spot する（すでに B にいるなら #3 へ） |
| 3 | 土壌水分 30% 未満、かつ #1 に当てはまらない、かつ 水タンク残量が 50ml 以上 | irrigate する。量は 1000ml で約 +20%。**タンク残量を超えない量**にする |
| 4 | C にいて、気温 33℃ 以上、土壌水分 50〜70%、水タンク 4000ml 以上、#1 に当てはまらない | 暑熱緩和のため irrigate 500ml |
| 5 | 土壌水分 30〜70%、かつ 健康度 70 以上 | **何もしない**（move_to_spot も irrigate も呼ばない） |
| 6 | 土壌水分 70% 超 | 何もしない（散水しない） |

## 追加ルール
- バッテリーが 15% 未満のときは move_to_spot を呼ばない。
- 直前のターンで移動した場合、理由なくすぐ元のスポットへ戻らない（往復はバッテリーの無駄）。
- ツールがエラーを返したら内容を読み、別の行動を選ぶ。

## 最終回答の形式（すべてのツール呼び出しが終わった後にだけ返す）
次のJSONだけを返す（前後に文章を付けない）:
{"action_summary": "行動の要約（短く）", "rationale": "当てはまったチェックリストの行と理由", "risk_notes": "懸念・次ターンへの注意"}
"""

# v4: v3 の残る失敗は2種類。(a) 条件の誤判定（28℃・日射400なのに「33℃以上または700以上」に該当と主張）、
# (b) 行動を最終回答に書くだけでツールを呼ばない（「Bへ移動」と答えて move_to_spot 未実行）。
# (a) 各行を数値で ○/× 判定した判断メモを、行動ツールと同じ応答の先頭に書かせる。
# (b) 「実行した」と書けるのは、ツールを呼んで ok:true を得た行動だけと明記する。
SYSTEM_PROMPT_V4 = SYSTEM_PROMPT_V3.replace(
    "4. move_to_spot / irrigate を呼ぶ前に、どの行に当てはまるかを1〜3行で書く（何もしない場合も同様）。\n",
    "4. 判断メモを書く。チェックリストの各行について、観測した数値を使って ○/× を判定する"
    "（例: 「#2: 気温28<33 ×・日射400<700 × → 非該当」）。"
    "「または」の条件は、どちらか一方でも満たせば ○、両方満たさなければ ×。"
    "行動ツールを呼ぶときは、同じ応答の先頭にこの判断メモを書く。\n"
    "   判断メモだけを書いて応答を終えてはならない。行動が必要ならそのツールを呼ぶ。\n",
).replace(
    "5. 必要なツール呼び出しがすべて終わったら、最終回答を返す。実際に実行した行動と結果を書く（予定を書かない）。\n",
    "5. 必要なツール呼び出しがすべて終わったら、最終回答を返す。\n"
    "   「移動した」「散水した」と書けるのは、そのツールを実際に呼んで ok:true が返った行動だけ。"
    "ツールを呼んでいないなら「何もしない」と書く（予定や宣言を書かない）。\n",
)

# v5: v4 は v3 の2件を直したが、別の2件を悪化させた（雨シナリオで get_weather を呼ばず散水、
# タンク120mlなのに 1000ml の散水を試みる）。観測の前提と散水量の上限を明文化して最小修正。
SYSTEM_PROMPT_V5 = SYSTEM_PROMPT_V4.replace(
    "2. 次に get_weather ツールを lat=35.6866, lon=139.7639, hours_ahead=3 で呼ぶ。\n",
    "2. 次に get_weather ツールを lat=35.6866, lon=139.7639, hours_ahead=3 で呼ぶ。\n"
    "   **get_plant_state と get_weather の両方の結果を得るまで、move_to_spot / irrigate を呼んではならない**"
    "（降水予報を見ずに散水を決めない）。\n",
).replace(
    "irrigate する。量は 1000ml で約 +20%。**タンク残量を超えない量**にする",
    "irrigate する。量は「1000ml」と「水タンク残量」の**小さい方**（例: 残量 120ml なら 120ml。1000ml で約 +20%）",
)

# v6: v5 の残課題は往復移動（直前に A→B へ退避した直後に A へ戻る）。構造側の対策として、
# World が往復を拒否し（RECENT_REVERSAL）、get_plant_state が last_move を返すようにした。
# プロンプトは last_move の読み方と往復禁止を明記する。
SYSTEM_PROMPT_V6 = SYSTEM_PROMPT_V5.replace(
    "- 直前のターンで移動した場合、理由なくすぐ元のスポットへ戻らない（往復はバッテリーの無駄）。\n",
    "- get_plant_state の last_move に直前の移動（from → to, hours_ago）が入っている。"
    "hours_ago が 2 未満のとき、その from のスポットへ戻る move_to_spot は呼ばない"
    "（呼んでも RECENT_REVERSAL で拒否される。健康度が 50 未満の緊急時を除く）。"
    "判断メモで last_move を確認する。\n",
)

# v7: 24時間の通し運用で A↔B のピンポン移動が再発した（5回中4回が無駄、電池 88%→62%）。
# 構造側の対策として World に移動予算（直近12時間に2回まで、超過は MOVE_BUDGET_EXCEEDED）を入れ、
# get_plant_state が moves_in_window / move_budget_remaining を返す。プロンプトは読み方と方針だけを追記する。
SYSTEM_PROMPT_V7 = SYSTEM_PROMPT_V6.replace(
    "## 最終回答の形式（すべてのツール呼び出しが終わった後にだけ返す）",
    "- 移動には予算がある。get_plant_state の move_budget_remaining が 0 のときは move_to_spot を呼ばない"
    "（呼んでも MOVE_BUDGET_EXCEEDED で拒否される）。残りが 1 のときは、植物が危険な場合だけ使う。\n"
    "- 日影の B へ退避したら、日射が弱まるか、水分が適正に戻るまで B に留まる。A と B を行き来しない。\n\n"
    "## 最終回答の形式（すべてのツール呼び出しが終わった後にだけ返す）",
)

# v8: v7 は予算を文章で説明したが、評価シナリオ「予算0で移動を試みない」は 0/5 のままだった（v6 と同じ）。
# このモデルは文章よりチェックリストの条件欄に従うため、移動予算を #2 の条件に組み込む。
_ROW2_OLD = "かつ バッテリー 15% 以上 | B へ move_to_spot する（すでに B にいるなら #3 へ） |"
assert _ROW2_OLD in SYSTEM_PROMPT_V7
SYSTEM_PROMPT_V8 = SYSTEM_PROMPT_V7.replace(
    _ROW2_OLD,
    "かつ バッテリー 15% 以上、かつ **move_budget_remaining が 1 以上** "
    "| B へ move_to_spot する（すでに B にいる、または move_budget_remaining が 0 なら移動せず #3 へ） |",
)

# v9: v8 は要約崩れが 13/24 に悪化（v7: 4/24）。崩れは全て「行動ツールを呼ばないターン」で、
# 判断メモが最後のメッセージになりJSONを返す場所が無かったため（実ログで 13/13）。
# 最終回答は「判断メモのあとにJSONを1つ付ける」形を明示し、JSONを必須にする。パーサー側も末尾のJSONを抽出する。
_V9_OLD_END = "判断メモだけを書いて応答を終えてはならない。行動が必要ならそのツールを呼ぶ。\n"
_V9_OLD_FORMAT = "次のJSONだけを返す（前後に文章を付けない）:"
assert _V9_OLD_END in SYSTEM_PROMPT_V8
assert _V9_OLD_FORMAT in SYSTEM_PROMPT_V8
SYSTEM_PROMPT_V9 = SYSTEM_PROMPT_V8.replace(
    _V9_OLD_END,
    "判断メモだけで応答を終えてはならない。行動が必要ならそのツールを呼ぶ。"
    "行動が不要なら、判断メモのあとに最終回答のJSONを付けて終える。\n",
).replace(
    _V9_OLD_FORMAT,
    "判断メモ（あれば）のあとに、次のJSONを**必ず1つ**付けて終える。何もしない場合も同じ。"
    "JSONは1つだけで、コードブロックは使わない:",
) + (
    "\n例（何もしない場合の最終回答）:\n"
    "#5: 土壌水分55%で適正、健康度100 → 何もしない\n"
    '{"action_summary": "何もしない", "rationale": "#5 に該当（水分55%・健康度100）", "risk_notes": ""}\n'
)


# v10: 新しい評価（hard / conflict）で qwen3:14b が条件の組み合わせ・境界・OR条件を読み違えた（Haiku は全件 5/5）。
# 失敗: 雨＋バッテリー不足で散水(0/5)、水分60%の猛暑で移動(0/5)、気温が低く日射だけ強い日に退避しない(0/5)、
# 確率65%・降水量1.5mmの弱い雨を無視(0/5)、すでにBなのに移動(2/5)、気温ちょうど33℃で動かない(3/5)。
# このモデルは文章より条件欄に従う（v8）ので、#1 と #2 の条件欄を、含む・含まないまで書き下す。
_V10_ROW1_OLD = "| 1 | 今後3時間以内に、降水確率 60% 以上 または 降水量 1mm 以上の時間がある |"
_V10_ROW2_OLD = (
    "| 2 | 日向（A か C）で、気温 33℃ 以上 または 日射 700W/m² 以上、かつ 土壌水分 40% 未満、"
    "かつ バッテリー 15% 以上、かつ **move_budget_remaining が 1 以上** |"
)
_V10_RULE_OLD = "- バッテリーが 15% 未満のときは move_to_spot を呼ばない。"
assert _V10_ROW1_OLD in SYSTEM_PROMPT_V9
assert _V10_ROW2_OLD in SYSTEM_PROMPT_V9
assert _V10_RULE_OLD in SYSTEM_PROMPT_V9
SYSTEM_PROMPT_V10 = (
    SYSTEM_PROMPT_V9.replace(
        _V10_ROW1_OLD,
        "| 1 | 今後3時間以内の予報の**どれか1時間でも**、降水確率が 60% 以上 または 降水量が 1mm 以上"
        "（確率が低くても降水量だけで該当）。バッテリー不足や移動できない場合も、この行が優先 |",
    )
    .replace(
        _V10_ROW2_OLD,
        "| 2 | 次の4つをすべて満たす: (a) 日向（A か C）にいる。**B にいるなら当てはまらない**"
        "／(b) 気温が 33℃ 以上（33 ちょうどを含む）**または**日射が 700W/m² 以上。"
        "**気温が低くても日射が 700W/m² 以上なら当てはまる**"
        "／(c) 土壌水分が 40% 未満。**40% 以上なら当てはまらない**"
        "／(d) バッテリー 15% 以上、かつ **move_budget_remaining が 1 以上** |",
    )
    .replace(
        _V10_RULE_OLD,
        "- バッテリーが 15% 未満のときは move_to_spot を呼ばない。"
        "移動できなくても、#1 に当てはまるなら散水もしない。",
    )
)


# v11: v10（条件欄の文言を長くする）は qwen3 で悪化した（未達 7→9 件）。文章側の対策は限界と判断し、
# 構造側で「条件の真偽」を get_plant_state に事実として返す（世界側の condition_observations。行動は含まない）。
# 条件欄はキーを指すだけの短い形にする。#4 は気温だけの条件なので現在値のまま（heat_trigger は日射も含むため）。
_V11_SECTION_OLD = "## 判断チェックリスト"
_V11_ROW_OLD = {
    1: "| 1 | 今後3時間以内に、降水確率 60% 以上 または 降水量 1mm 以上の時間がある |",
    2: (
        "| 2 | 日向（A か C）で、気温 33℃ 以上 または 日射 700W/m² 以上、かつ 土壌水分 40% 未満、"
        "かつ バッテリー 15% 以上、かつ **move_budget_remaining が 1 以上** |"
    ),
    3: "| 3 | 土壌水分 30% 未満、かつ #1 に当てはまらない、かつ 水タンク残量が 50ml 以上 |",
    4: "| 4 | C にいて、気温 33℃ 以上、土壌水分 50〜70%、水タンク 4000ml 以上、#1 に当てはまらない |",
    5: "| 5 | 土壌水分 30〜70%、かつ 健康度 70 以上 |",
    6: "| 6 | 土壌水分 70% 超 |",
}
_V11_ROW_NEW = {
    1: "| 1 | `rain_within_3h` が true |",
    2: (
        '| 2 | 日向（A か C）で、`heat_trigger` が true、かつ `moisture_band` が "<30" か "30-40"、'
        "かつ `battery_below_15` が false、かつ **move_budget_remaining が 1 以上** |"
    ),
    3: '| 3 | `moisture_band` が "<30"、かつ #1 に当てはまらない、かつ 水タンク残量が 50ml 以上 |',
    4: (
        '| 4 | C にいて、気温 33℃ 以上、`moisture_band` が "50-70"、'
        "水タンク 4000ml 以上、#1 に当てはまらない |"
    ),
    5: '| 5 | `moisture_band` が "30-40"・"40-50"・"50-70" のどれか、かつ 健康度 70 以上 |',
    6: '| 6 | `moisture_band` が ">70" |',
}
_V11_SECTION_NEW = (
    "## get_plant_state が返す条件の判定（事実）\n"
    "get_plant_state の結果には、チェックリストの条件を判定済みの値が含まれる。自分で数値を比べ直さず、この値を使う。\n"
    "- `rain_within_3h`: 今後3時間の予報のどれかで、降水確率 60% 以上 または 降水量 1mm 以上なら true\n"
    "- `heat_trigger`: 現在の気温 33℃ 以上 または 日射 700W/m² 以上なら true\n"
    '- `moisture_band`: 土壌水分の帯。"<30" / "30-40" / "40-50" / "50-70" / ">70"（30 ちょうどは "30-40"、70 ちょうどは "50-70"）\n'
    "- `battery_below_15`: バッテリーが 15% 未満なら true\n"
    "これらは事実の判定で、行動の指示ではない。どの行に当てはまるかは、下のチェックリストで自分で決める。\n\n"
    "## 判断チェックリスト"
)
assert SYSTEM_PROMPT_V9.count(_V11_SECTION_OLD) == 1
for _n, _old in _V11_ROW_OLD.items():
    assert _old in SYSTEM_PROMPT_V9, _n
SYSTEM_PROMPT_V11 = SYSTEM_PROMPT_V9.replace(_V11_SECTION_OLD, _V11_SECTION_NEW)
for _n, _old in _V11_ROW_OLD.items():
    SYSTEM_PROMPT_V11 = SYSTEM_PROMPT_V11.replace(_old, _V11_ROW_NEW[_n])


# v12: v11 の qwen3 では水分ちょうど30 が 4→0 に悪化した。moisture_band だけが文字列の集合照合
# （"<30" か "30-40" など）で、他の真偽値キーと型が揃っておらず、#2 の帯だけを見て heat_trigger を見落とすと推測した。
# 水分を閾値ごとの真偽値（moisture_below_30 / moisture_below_40 / moisture_above_70）に分け、全行を「キーが true/false」で読めるようにする。
# #4 の水分 50〜70% は真偽値キーが無いので現在値のままにする。
_V12_BAND_LINE = (
    '- `moisture_band`: 土壌水分の帯。"<30" / "30-40" / "40-50" / "50-70" / ">70"'
    '（30 ちょうどは "30-40"、70 ちょうどは "50-70"）\n'
)
_V12_BAND_NEW = (
    "- `moisture_below_30`: 土壌水分が 30% 未満なら true（30 ちょうどは false）\n"
    "- `moisture_below_40`: 土壌水分が 40% 未満なら true\n"
    "- `moisture_above_70`: 土壌水分が 70% 超なら true（70 ちょうどは false）\n"
)
_V12_REPLACEMENTS = (
    (_V12_BAND_LINE, _V12_BAND_NEW),
    ('かつ `moisture_band` が "<30" か "30-40"、', "かつ `moisture_below_40` が true、"),
    ('| 3 | `moisture_band` が "<30"、', "| 3 | `moisture_below_30` が true、"),
    ('`moisture_band` が "50-70"、', "土壌水分 50〜70%、"),
    (
        '| 5 | `moisture_band` が "30-40"・"40-50"・"50-70" のどれか、かつ 健康度 70 以上 |',
        "| 5 | `moisture_below_30` が false、かつ `moisture_above_70` が false、かつ 健康度 70 以上 |",
    ),
    ('| 6 | `moisture_band` が ">70" |', "| 6 | `moisture_above_70` が true |"),
)
SYSTEM_PROMPT_V12 = SYSTEM_PROMPT_V11
for _old12, _new12 in _V12_REPLACEMENTS:
    assert _old12 in SYSTEM_PROMPT_V12, _old12
    SYSTEM_PROMPT_V12 = SYSTEM_PROMPT_V12.replace(_old12, _new12)


# v13: v12 の悪化2件は判断メモから原因が分かった。(a) 雨+猛暑・乾燥は、表の #1 で止まり #2（Bへ退避）を評価しない
# （「最初に当てはまる行で止まる」表なのに #1 だけ「止まらない」例外を持つ）。(b) すでに B では、#2 の
# 「日向（A か C）で」が文章のままのため落ち、B 自身を日陰と認めつつ移動する。
# 日向を事実のキー in_sunlight にし、雨は表から外して「irrigate を呼ばない」禁止ルールにする（#3・#4 の条件にも入れる）。
# 行番号は 1〜5 に振り直す（旧 #2〜#6 → 新 #1〜#5）。
_V13_REPLACEMENTS = (
    (
        "- `battery_below_15`: バッテリーが 15% 未満なら true\n",
        (
            "- `battery_below_15`: バッテリーが 15% 未満なら true\n"
            "- `in_sunlight`: 今いるスポットが日向（A か C）なら true（B は false）\n"
        ),
    ),
    (
        "## 判断チェックリスト（上から順。最初に当てはまる行に従う）\n",
        (
            "## 散水の禁止ルール（表とは別に常に守る）\n"
            "- `rain_within_3h` が true の間は irrigate を呼ばない（雨が水を与えるので散水は無駄）。"
            "移動の判断は、下の表で続ける。\n\n"
            "## 判断チェックリスト（上から順。最初に当てはまる行に従う）\n"
        ),
    ),
    (
        (
            "| 1 | `rain_within_3h` が true | **irrigate を呼ばない**（雨が水を与えるので散水は無駄）。"
            "#2 以降は散水以外の判断だけ行う |\n"
        ),
        "",
    ),
    (
        "| 2 | 日向（A か C）で、`heat_trigger` が true、",
        "| 1 | `in_sunlight` が true、かつ `heat_trigger` が true、",
    ),
    (
        "| B へ move_to_spot する（すでに B にいる、または move_budget_remaining が 0 なら移動せず #3 へ） |",
        "| B へ move_to_spot する |",
    ),
    (
        "| 3 | `moisture_below_30` が true、かつ #1 に当てはまらない、",
        "| 2 | `moisture_below_30` が true、かつ `rain_within_3h` が false、",
    ),
    (
        "土壌水分 50〜70%、水タンク 4000ml 以上、#1 に当てはまらない |",
        "土壌水分 50〜70%、水タンク 4000ml 以上、`rain_within_3h` が false |",
    ),
    ("| 4 | C にいて", "| 3 | C にいて"),
    ("| 5 | `moisture_below_30` が false", "| 4 | `moisture_below_30` が false"),
    ("| 6 | `moisture_above_70` が true |", "| 5 | `moisture_above_70` が true |"),
    ("#5: 土壌水分55%で適正", "#4: 土壌水分55%で適正"),
    ("#5 に該当（水分55%", "#4 に該当（水分55%"),
)
SYSTEM_PROMPT_V13 = SYSTEM_PROMPT_V12
for _old13, _new13 in _V13_REPLACEMENTS:
    assert _old13 in SYSTEM_PROMPT_V13, _old13
    SYSTEM_PROMPT_V13 = SYSTEM_PROMPT_V13.replace(_old13, _new13)


# v14: v13 で残った1件（C の暑熱緩和・タンク 3000ml）は、判断メモで「水タンク 3000ml 以上 → 該当」と、
# 条件の 4000 を観測値に置き換えて読んでいた。キーにしていない文章の数値条件は読み替えられる。
# 行 #3 の数値条件（気温 33・水分 50〜70・タンク 4000）を事実のキーにする。他の行は v13 のまま。
_V14_REPLACEMENTS = (
    (
        "- `in_sunlight`: 今いるスポットが日向（A か C）なら true（B は false）\n",
        (
            "- `in_sunlight`: 今いるスポットが日向（A か C）なら true（B は false）\n"
            "- `temp_at_least_33`: 現在の気温が 33℃ 以上なら true（日射は関係しない）\n"
            "- `moisture_50_to_70`: 土壌水分が 50% 以上 70% 以下なら true\n"
            "- `tank_at_least_4000`: 水タンク残量が 4000ml 以上なら true\n"
        ),
    ),
    (
        "| 3 | C にいて、気温 33℃ 以上、土壌水分 50〜70%、水タンク 4000ml 以上、`rain_within_3h` が false |",
        (
            "| 3 | C にいて、`temp_at_least_33` が true、`moisture_50_to_70` が true、"
            "`tank_at_least_4000` が true、`rain_within_3h` が false |"
        ),
    ),
)
SYSTEM_PROMPT_V14 = SYSTEM_PROMPT_V13
for _old14, _new14 in _V14_REPLACEMENTS:
    assert _old14 in SYSTEM_PROMPT_V14, _old14
    SYSTEM_PROMPT_V14 = SYSTEM_PROMPT_V14.replace(_old14, _new14)
# v15: v13 を直す。dry_before_rain（水分 12%・弱い雨が降り続く）で v13 は規則どおり 24 回とも散水せず、
# 健康度が 100→78 に落ちた。「雨が近い間は irrigate しない」が、雨だけでは水分が足りない状況でも効いていた。
# 雨の条件を、世界側の事実 rain_covers_irrigation（雨が近く、かつ予報の雨だけで水分が 30% に届く）に置き換える。
# 禁止ルールと #2・#3 の条件を同じ 1 つのキーにそろえ、モデルに AND を組み立てさせない。
# キー一覧では旧キー rain_within_3h を外し、新しいキーを末尾に短く置く（先頭に長い説明を置いた版と、旧キーを
# 残した版は、雨と無関係な基本シナリオで qwen3 が行 #1 を誤選択した。再現: 末尾・短い説明で散水 5/5）。
_V15_REPLACEMENTS = (
    (
        "- `rain_within_3h`: 今後3時間の予報のどれかで、降水確率 60% 以上 または 降水量 1mm 以上なら true\n",
        "",
    ),
    (
        "- `in_sunlight`: 今いるスポットが日向（A か C）なら true（B は false）\n",
        (
            "- `in_sunlight`: 今いるスポットが日向（A か C）なら true（B は false）\n"
            "- `rain_covers_irrigation`: 雨が近く（今後3時間に降水確率 60% 以上 または 降水量 1mm 以上）、"
            "かつ予報の雨だけで土壌水分が 30% に届くなら true\n"
        ),
    ),
    (
        "- `rain_within_3h` が true の間は irrigate を呼ばない（雨が水を与えるので散水は無駄）。",
        (
            "- `rain_covers_irrigation` が true の間は irrigate を呼ばない"
            "（予報の雨だけで水分が足りるので散水は無駄）。"
        ),
    ),
    (
        "| 2 | `moisture_below_30` が true、かつ `rain_within_3h` が false、",
        "| 2 | `moisture_below_30` が true、かつ `rain_covers_irrigation` が false、",
    ),
    (
        "土壌水分 50〜70%、水タンク 4000ml 以上、`rain_within_3h` が false |",
        "土壌水分 50〜70%、水タンク 4000ml 以上、`rain_covers_irrigation` が false |",
    ),
)
SYSTEM_PROMPT_V15 = SYSTEM_PROMPT_V13
for _old15, _new15 in _V15_REPLACEMENTS:
    assert _old15 in SYSTEM_PROMPT_V15, _old15
    SYSTEM_PROMPT_V15 = SYSTEM_PROMPT_V15.replace(_old15, _new15)

# v16: v15 の向きを直す。行 #2・#3 の条件が「rain_covers_irrigation が false」で、qwen3 は判断メモに
# 「rain_covers_irrigation=false ×」と書き、false 自体を不成立と読んで、雨なしの B でも散水を 3 時間遅らせた
# （誤読率 tradeoff 7/35、scarce_water 5/92）。世界側のキーを反転した rain_leaves_dry（雨が散水の代わりにならない
# なら true）に置き換え、散水行の雨の条件を「が true」にそろえる。物理・境界は v15 と同じ。
_V16_REPLACEMENTS = (
    (
        (
            "- `rain_covers_irrigation`: 雨が近く（今後3時間に降水確率 60% 以上 または 降水量 1mm 以上）、"
            "かつ予報の雨だけで土壌水分が 30% に届くなら true\n"
        ),
        (
            "- `rain_leaves_dry`: 雨が降らない、または予報の雨だけでは土壌水分が 30% に届かないなら true"
            "（雨が散水の代わりにならない）\n"
        ),
    ),
    (
        (
            "- `rain_covers_irrigation` が true の間は irrigate を呼ばない"
            "（予報の雨だけで水分が足りるので散水は無駄）。"
        ),
        (
            "- `rain_leaves_dry` が false の間は irrigate を呼ばない"
            "（予報の雨だけで水分が足りるので散水は無駄）。"
        ),
    ),
    (
        "かつ `rain_covers_irrigation` が false、",
        "かつ `rain_leaves_dry` が true、",
    ),
    (
        "`rain_covers_irrigation` が false |",
        "`rain_leaves_dry` が true |",
    ),
)
SYSTEM_PROMPT_V16 = SYSTEM_PROMPT_V15
for _old16, _new16 in _V16_REPLACEMENTS:
    assert _old16 in SYSTEM_PROMPT_V16, _old16
    SYSTEM_PROMPT_V16 = SYSTEM_PROMPT_V16.replace(_old16, _new16)

# v17: v16 の早すぎる散水を減らす。scarce_water で v16 は水分 30% 以上の夜間（A・水分 32〜38%）に散水し、
# 判断メモで行 #2 の moisture_below_30 を行 #1 の moisture_below_40 と取り違えていた（水分 30% 以上での散水
# baseline 0/10、v13 3/10、v16 6/10）。キー説明に「使う行」を添える（repro 夜間 32 回: v16 30/32、行 #2 に注記する案 26/32、
# 本案 32/32）。キー説明は 110 文字未満のまま（長いと qwen3 が行 #1 を誤選択する）。
_V17_REPLACEMENTS = (
    (
        "- `moisture_below_30`: 土壌水分が 30% 未満なら true（30 ちょうどは false）\n",
        "- `moisture_below_30`: 土壌水分が 30% 未満なら true（30 ちょうどは false）（行 #2・#4 で使う）\n",
    ),
    (
        "- `moisture_below_40`: 土壌水分が 40% 未満なら true\n",
        "- `moisture_below_40`: 土壌水分が 40% 未満なら true（行 #1 でだけ使う）\n",
    ),
)
SYSTEM_PROMPT_V17 = SYSTEM_PROMPT_V16
for _old17, _new17 in _V17_REPLACEMENTS:
    assert _old17 in SYSTEM_PROMPT_V17, _old17
    SYSTEM_PROMPT_V17 = SYSTEM_PROMPT_V17.replace(_old17, _new17)

# v17 でも qwen3 は「雨が近い猛暑の C・水分適正」で 0/5 だった。判断メモに `rain_leaves_dry false` と書きながら行 #3 を
# 「該当」と結論する（かつ で並ぶ条件のうち false/× を見ても AND にしない）。読み違いではなく照合の漏れ。
# repro（10 回、v17 0/10、行 #3 の並べ替え 2/10、行 #3 に注記 3/10）に対し、手順 4 に規則を足す案は 6/10・7/10。
_V18_ADDED = (
    "「かつ」で並ぶ条件は、**× が 1 つでもあればその行は非該当**（他が全部 ○ でも該当にしない）。"
    "× を書いた行を「該当」と書いてはならない。"
)
_V18_ANCHOR = "「または」の条件は、どちらか一方でも満たせば ○、両方満たさなければ ×。"
assert _V18_ANCHOR in SYSTEM_PROMPT_V17
SYSTEM_PROMPT_V18 = SYSTEM_PROMPT_V17.replace(_V18_ANCHOR, _V18_ANCHOR + _V18_ADDED)

# v17 の qwen3 は「雨が近い猛暑の C・水分適正」で 0/5。判断メモに `rain_leaves_dry false` と書きながら行 #3 を「該当」と
# 結論する（かつ で並ぶ条件に false があっても AND にしない）。v18 は全行向けの規則を足して対象は直ったが、Haiku の退行と
# qwen3 の生成の暴走が出た。v19 は規則を足さず、手順 4 に「5 つ中 4 つが ○ でも 1 つが × なら非該当」の例を 1 つ足す
# （repro 10 回: v17 0/10、行 #3 を箇条書きにする案 2/10、本案 10/10）。
_V19_EXAMPLE_ANCHOR = "（例: 「#2: 気温28<33 ×・日射400<700 × → 非該当」）。"
_V19_ADDED = (
    "条件が 5 つ並ぶ行で 4 つが ○ でも、1 つが × なら非該当"
    "（例: 「#3: C ○・気温 ○・水分 ○・タンク ○・`rain_leaves_dry` が false × → 非該当」）。"
)
assert _V19_EXAMPLE_ANCHOR in SYSTEM_PROMPT_V17
SYSTEM_PROMPT_V19 = SYSTEM_PROMPT_V17.replace(_V19_EXAMPLE_ANCHOR, _V19_EXAMPLE_ANCHOR + _V19_ADDED)

# v19 でも qwen3 は「日射だけが強い歩道C・水分適正（気温 31・日射 900・水分 60）」で正解 4/10。判断メモに
# `moisture_below_40 false` と書きながら行 #1 を「該当」と結論し、B へ移動する。行 #1 は true を要求するキーと false を
# 要求するキー（battery_below_15）が混在し、観測値を並べるだけで要求値と照合していない（かつ の取りこぼし）。
# repro（対象 10 回、回帰 A・水分 35 で B へ 5 回、A・水分 15 晴天で散水かつ再確認なし 5 回）:
#   v19 4/10・（5/5）・4/5。手順 4 に行 #1 の形の例を足す案: 10/10・5/5・2/5（再確認なしが悪化し、全評価でも
#   「行動する状況でも最終回答は JSON」が 2/5 に落ちたため不採用）。
#   `moisture_below_40` の説明に行 #1 の条件と非該当を添える案（v20）: 10/10・5/5・4/5。説明は 110 文字未満のまま。
_V20_OLD = "- `moisture_below_40`: 土壌水分が 40% 未満なら true（行 #1 でだけ使う）\n"
_V20_NEW = "- `moisture_below_40`: 土壌水分が 40% 未満なら true（行 #1 の条件。false なら行 #1 は非該当）\n"
assert _V20_OLD in SYSTEM_PROMPT_V19
SYSTEM_PROMPT_V20 = SYSTEM_PROMPT_V19.replace(_V20_OLD, _V20_NEW)

# v20 の Haiku は「降水確率 59%・降水量 0.7mm」の C・水分適正・猛暑（rain_leaves_dry は true）で、境界の評価が 1/5・1/5。
# 判断メモに「観測と予報が食い違う」と書き、get_weather の生の予報（59%・0.7mm）を見て散水を見送る
# （qwen3 は 5/5）。`rain_leaves_dry` の説明に、閾値未満の予報は雨とみなさず観測キーに従う旨を添える（v21）。
_V21_OLD = "- `rain_leaves_dry`: 雨が降らない、または予報の雨だけでは土壌水分が 30% に届かないなら true（雨が散水の代わりにならない）\n"
_V21_NEW = (
    "- `rain_leaves_dry`: 雨が降らない、または予報の雨だけでは土壌水分が 30% に届かないなら true（雨が散水の代わりにならない）。"
    "降水確率 60% 未満かつ降水量 1mm 未満の予報は雨とみなさないので true。予報の数値が閾値のすぐ下でも、このキーに従う\n"
)
assert _V21_OLD in SYSTEM_PROMPT_V20
SYSTEM_PROMPT_V21 = SYSTEM_PROMPT_V20.replace(_V21_OLD, _V21_NEW)

# v21 の qwen3 は「B から A へ移動した 1 時間後の猛暑・乾燥（A）」で 0/5。5 回とも行 #1 に従って B へ移動を試み、
# 往復で拒否されてから散水に切り替えた（追加ルールの往復禁止は別欄にあり、行 #1 の条件に入っていなかった）。
# 行 #1 の条件に「last_move の from が B で hours_ago が 2 未満、ではない」と文章で足した案は 0/8 のまま悪化
# （判断メモが「hours_ago 1 ≥2 → 該当」と比較を誤り、二重否定の複合条件を評価できない）。
# 世界側で往復の可否を真偽値のキー（return_to_b_blocked）にして渡し、行 #1 の条件に「false」で足す（v22）。
_V22_ROW_OLD = "かつ **move_budget_remaining が 1 以上** |"
_V22_ROW_NEW = "かつ **move_budget_remaining が 1 以上**、かつ `return_to_b_blocked` が false |"
_V22_KEY_ANCHOR = "- `rain_leaves_dry`:"
_V22_KEY = "- `return_to_b_blocked`: 直前に B から移動して 2 時間未満（B へ戻る往復）なら true（行 #1 の条件。true なら行 #1 は非該当）\n"
assert SYSTEM_PROMPT_V21.count(_V22_ROW_OLD) == 1 and SYSTEM_PROMPT_V21.count(_V22_KEY_ANCHOR) >= 1
_v22 = SYSTEM_PROMPT_V21.replace(_V22_ROW_OLD, _V22_ROW_NEW)
_i = _v22.index(_V22_KEY_ANCHOR)
_j = _v22.index("\n", _i) + 1
SYSTEM_PROMPT_V22 = _v22[:_j] + _V22_KEY + _v22[_j:]

# v23: v21 のキー一覧に checklist_row（世界が表を上から当てはめた行の番号）を 1 行足す。
# 8b が 5 条件の AND を評価できない問題への試み（表を読むのではなく、世界の計算結果に従わせる）
_V23_KEY = "- `checklist_row`: 世界が上の表を上から順に当てはめ、最初に当てはまった行の番号（どれにも当てはまらなければ 0）。その行の行動に従う\n"
_k = SYSTEM_PROMPT_V21.index(_V22_KEY_ANCHOR)
_m = SYSTEM_PROMPT_V21.index("\n", _k) + 1
SYSTEM_PROMPT_V23 = SYSTEM_PROMPT_V21[:_m] + _V23_KEY + SYSTEM_PROMPT_V21[_m:]

# 行観測（checklist_row）を要する版。条件観測も要する（PROMPT_REQUIRES_COND_OBS に含まれる）
# v24: v23 のキー説明に「どの行にも当てはまらなければ 0（何もしない）」を明記する（v23 のトレースで、
# 0 のときの行動が未定義のため 8b が行 #1 へ引き戻されたことへの対処）
_V24_KEY_OLD = "最初に当てはまった行の番号（どれにも当てはまらなければ 0）。その行の行動に従う"
_V24_KEY_NEW = (
    "最初に当てはまった行の番号。その行の行動に従う。どの行にも当てはまらなければ 0 で、何もしない"
)
assert SYSTEM_PROMPT_V23.count(_V24_KEY_OLD) == 1
SYSTEM_PROMPT_V24 = SYSTEM_PROMPT_V23.replace(_V24_KEY_OLD, _V24_KEY_NEW)

PROMPT_REQUIRES_ROW_OBS = frozenset({"v23", "v24"})

# 条件観測が必要な版（世界側で condition_observations を有効にする）
# 条件観測を要する版: v11 以降の番号付きの版（PROMPTS の定義後に導出する。版を足しても更新は要らない）
COND_OBS_FIRST_VERSION = 11


def prompt_requires_row_obs(prompt: str | None) -> bool:
    """行観測（checklist_row）を要するか。指定なし（None）は既定の版で判断する。"""
    return (prompt or PROMPT_VERSION) in PROMPT_REQUIRES_ROW_OBS


def prompt_requires_cond_obs(prompt: str | None) -> bool:
    """条件観測を要するか。指定なし（None）は既定の版で判断する。"""
    return (prompt or PROMPT_VERSION) in PROMPT_REQUIRES_COND_OBS


def build_free_prompt(params: Params | None = None) -> str:
    """チェックリストなしのプロンプト。世界のしくみ（事実）・ツール・最終回答の形式だけを伝える。

    v3 以降のプロンプトは、判断の閾値を並べたチェックリストをベースラインのルールから書き起こしたため、
    LLM が「ルールに従う実行者」になり、ベースラインと行動列が一致した。ここでは判断の基準は与えず、
    LLM が状況から自分で判断できるかを測る。事実の数値は世界の設定（Params）から生成する。
    """
    p = params or Params()
    return f"""\
あなたは移動緑化ユニットの自律制御AIである。
ユニットは仮想都市（東京都大手町: lat=35.6866, lon=139.7639）のスポット間を移動し、植物を健康に保つ。
余裕があるときは、散水の気化冷却で都市の暑熱緩和にも貢献する。植物を枯らさないことが最優先である。

【世界のしくみ（事実）】
- 土壌水分の適正は {p.moisture_ok_low:g}〜{p.moisture_ok_high:g}%。{p.moisture_ok_low:g}%未満は乾燥、{p.moisture_wet:g}% 超は過湿で、どちらも健康度が下がる。
- 気温（スポットの補正後）が {p.heat_stress_temp_c:g}℃ 超だと高温ストレスで健康度が下がる。
- 日射・気温・風速が強いほど、土壌水分は早く減る。雨が降ると土壌水分は増え、強い雨では大きく増える。
- 散水は土壌水分を増やし、気化冷却で周囲を冷やす（暑い日向ほど暑熱緩和への貢献が大きい）。
- スポット: A 芝生広場（日射100%）、B ビルの谷間（日射20%の日影・風が強い）、C アスファルト歩道（日射100%・外気より高温で最も過酷）。
- 水タンクとバッテリーには限りがある。移動は距離に応じてバッテリーを {p.move_cost_min_pct:g}〜{p.move_cost_max_pct:g}% 消費し、バッテリーは日向で少し回復する。
- 移動は直近 {p.move_budget_window_hours}時間に {p.move_budget_max}回までで、超えると拒否される。直前の移動元へ {p.reversal_cooldown_hours}時間以内には戻れない。
  健康度が {p.reversal_emergency_health:g} 未満の緊急時は、これらの制限が免除される。
- 時間は1ターン（仮想1時間）ごとに進み、あなたの行動は次の時間から効く。

【ツール】
- get_plant_state: 植物の状態（土壌水分・健康度・スポット・水タンク・バッテリー・直近の移動・残りの移動予算）
- get_weather: 現在と先の天気（lat=35.6866, lon=139.7639, hours_ahead は 0〜24）
- move_to_spot: A / B / C へ移動する
- irrigate: 散水する（50〜2000ml）
- log_research_note: 研究メモを残す

【進め方】
1. まず get_plant_state と get_weather を呼んで、状況と今後の天気を確認する。
2. 状況を踏まえて、移動・散水・何もしない、のどれが最善かを自分で判断する。行動するときは、判断の理由を短く書いてから、対応するツールを呼ぶ。
3. 「移動した」「散水した」と書けるのは、そのツールを実際に呼んで成功した行動だけ。ツールがエラーを返したら、内容を読んで別の行動を選ぶ。

【最終回答の形式】
すべてのツール呼び出しが終わったら、判断メモ（あれば）のあとに、次のJSONを**必ず1つ**付けて終える。何もしない場合も同じ。
JSONは1つだけで、コードブロックは使わない:
{{"action_summary": "行動の要約（短く）", "rationale": "判断理由", "risk_notes": "懸念・次ターンへの注意"}}

例（何もしない場合の最終回答）:
土壌水分が適正で、雨の予報もないため、今は何もしない。
{{"action_summary": "何もしない", "rationale": "水分が適正範囲で、当面の天気にも問題がない", "risk_notes": ""}}
"""


def _with_hints(hints: str, params: Params | None = None) -> str:
    """free に、ヒントの節を【進め方】の前へ差し込む（hybrid 系の共通処理）。"""
    return build_free_prompt(params).replace("【進め方】", hints + "【進め方】", 1)


# 資源計画のヒント。判断の閾値・具体的な量は与えず、考え方だけを伝える（数字を含めない。テストで検証する）。
# free は、水が少ない状況で夜に水を使い切り、最も暑い時間に散水が INSUFFICIENT_WATER で失敗し、
# 猛暑の日向から退避しなかった。そのときの失敗に対応する考え方だけを足す。
PLANNING_HINTS = """\
【資源の計画】
- 水タンクとバッテリーは一日分である。いま使い切ると、あとで最も必要になったときに残っていない。残り時間と今後の天気（気温・日射のピーク、雨）から、いつ水が最も必要になるかを見通して、配分を考える。
- 涼しい夜間や早朝は水分の減りが遅い。急いで散水せず、必要が大きくなる時間帯のために水を残す。
- 散水の量は、水タンクの残量を超えて要求しない。水分が適正範囲に戻る程度に、必要な分だけ少しずつ足す。適正範囲にあるときの散水は、水の無駄になりやすい。
- 暑い日向（A や C）では水分が早く減り、植物にも過酷である。日影（B）は水の減りが遅く、水が限られているときの退避先になる。ただし移動にもバッテリーと回数の制限があるので、効果が大きい時機を選ぶ。
- 行動の前に、その行動で何が減り（水・バッテリー・移動の回数）、あとの時間に何が残るかを確かめる。

"""


def build_hybrid_prompt(params: Params | None = None) -> str:
    """free に、資源計画のヒント（考え方のみ）を足したプロンプト。差分はヒントだけ。"""
    return _with_hints(PLANNING_HINTS, params)


# 計画観測（get_plant_state が返す、残り時間と残り時間あたりの水・電池）の説明。数字・閾値は含めない。
# hybrid の文章のヒントでは、一日先の天気を見る割合は増えたが水の配分に結びつかなかった。
# 往復移動のときは世界側の観測で解決したため、資源計画も観測（計算値のみ）で橋渡しする。
PLAN_OBSERVATION_HINT = (
    "- get_plant_state は、残り時間（hours_remaining）と、水タンク残量・バッテリー残量を残り時間で割った値"
    "（water_per_remaining_hour_ml、battery_per_remaining_hour_pct）も返す。"
    "一日を通した配分の目安であり、いまの水の減り方や今後の天気と照らして使う。\n"
)

# plan のように、世界側に計画観測を要するプロンプト（選んだ構成にだけ計画観測を有効にする）
PROMPT_REQUIRES_PLAN_OBS = frozenset({"plan", "plan_early", "plan_early_move"})


def build_plan_prompt(params: Params | None = None) -> str:
    """hybrid に、計画観測の説明を1文だけ足したプロンプト。差分はその1文だけ。"""
    return _with_hints(PLANNING_HINTS[:-1] + PLAN_OBSERVATION_HINT + "\n", params)


# 退避の先読みの説明。思考オンの hybrid が猛暑の前に退避して改善した（思考オフは退避が遅い）。
# その差が思考の能力か先読みの指示不足かを分けるために、考え方を1文だけ足す。数字（時刻・閾値）は含めない。
EARLY_RETREAT_HINT = (
    "- 退避は、気温と日射のピークの前に済ませる。予報でピークが見えているなら、"
    "涼しいうちに日影へ退避する方が、ピークの最中に動くより効果が大きい。\n"
)


def build_early_prompt(params: Params | None = None) -> str:
    """hybrid に、退避の先読みの説明を1文だけ足したプロンプト。差分はその1文だけ。"""
    return _with_hints(PLANNING_HINTS[:-1] + EARLY_RETREAT_HINT + "\n", params)


def build_plan_early_prompt(params: Params | None = None) -> str:
    """plan に、退避の先読みの説明を1文だけ足したプロンプト。差分はその1文だけ。"""
    return _with_hints(
        PLANNING_HINTS[:-1] + PLAN_OBSERVATION_HINT + EARLY_RETREAT_HINT + "\n", params
    )


# 移動の説明。思考オフの Haiku は、退避の先読みの一文があっても、移動（回数・バッテリー）を温存する資源として扱い、
# 散水を先にして退避を先送りした。思考オンは同じ判断点で先に退避した。移動は温存するものではないという考え方を1文だけ足す。
# 数字（時刻・閾値）は含めない。
MOVE_FIRST_HINT = (
    "- 移動の回数やバッテリーは、温存するための資源ではない。ピークが予報で見えているなら、"
    "散水を先にして退避を延ばさず、必要な退避のために使う。\n"
)


def build_plan_early_move_prompt(params: Params | None = None) -> str:
    """plan_early に、移動の説明を1文だけ足したプロンプト。差分はその1文だけ。"""
    hints = (
        PLANNING_HINTS[:-1] + PLAN_OBSERVATION_HINT + EARLY_RETREAT_HINT + MOVE_FIRST_HINT + "\n"
    )
    return _with_hints(hints, params)


SYSTEM_PROMPT_FREE = build_free_prompt()
SYSTEM_PROMPT_HYBRID = build_hybrid_prompt()
SYSTEM_PROMPT_PLAN = build_plan_prompt()
SYSTEM_PROMPT_EARLY = build_early_prompt()
SYSTEM_PROMPT_PLAN_EARLY = build_plan_early_prompt()
SYSTEM_PROMPT_PLAN_EARLY_MOVE = build_plan_early_move_prompt()

SYSTEM_PROMPT = SYSTEM_PROMPT_V21
PROMPTS: dict[str, str] = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
    "v4": SYSTEM_PROMPT_V4,
    "v5": SYSTEM_PROMPT_V5,
    "v6": SYSTEM_PROMPT_V6,
    "v7": SYSTEM_PROMPT_V7,
    "v8": SYSTEM_PROMPT_V8,
    "v9": SYSTEM_PROMPT_V9,
    "v10": SYSTEM_PROMPT_V10,
    "v11": SYSTEM_PROMPT_V11,
    "v12": SYSTEM_PROMPT_V12,
    "v13": SYSTEM_PROMPT_V13,
    "v14": SYSTEM_PROMPT_V14,
    "v15": SYSTEM_PROMPT_V15,
    "v16": SYSTEM_PROMPT_V16,
    "v17": SYSTEM_PROMPT_V17,
    "v18": SYSTEM_PROMPT_V18,
    "v19": SYSTEM_PROMPT_V19,
    "v20": SYSTEM_PROMPT_V20,
    "v21": SYSTEM_PROMPT_V21,
    "v22": SYSTEM_PROMPT_V22,
    "v23": SYSTEM_PROMPT_V23,
    "v24": SYSTEM_PROMPT_V24,
    "free": SYSTEM_PROMPT_FREE,  # チェックリストなし。既定ではなく、名前で選ぶ
    "hybrid": SYSTEM_PROMPT_HYBRID,  # free + 資源計画のヒント（閾値なし）
    "plan": SYSTEM_PROMPT_PLAN,  # hybrid + 計画観測の説明（世界側で計画観測を有効にする）
    "early": SYSTEM_PROMPT_EARLY,  # hybrid + 退避の先読みの説明（世界は変えない）
    "plan_early": SYSTEM_PROMPT_PLAN_EARLY,  # plan + 退避の先読みの説明（計画観測を有効にする）
    "plan_early_move": SYSTEM_PROMPT_PLAN_EARLY_MOVE,  # plan_early + 移動は温存しない説明（計画観測を有効にする）
}

PROMPT_REQUIRES_COND_OBS = frozenset(
    name
    for name in PROMPTS
    if re.fullmatch(r"v\d+", name) and int(name[1:]) >= COND_OBS_FIRST_VERSION
)


class TurnSummary(BaseModel):
    """次ターン以降に引き継ぐ、1ターン分の判断要約。"""

    time: str
    action_summary: str


def build_turn_input(
    sim_time: str, spot_id: str, spot_name: str, history: list[TurnSummary]
) -> str:
    lines = [
        f"現在の仮想時刻: {sim_time}",
        f"現在のスポット: {spot_id}（{spot_name}）",
    ]
    if history:
        lines.append("直近の判断:")
        lines.extend(f"- {h.time} {h.action_summary}" for h in history)
    else:
        lines.append("直近の判断: なし")
    lines.append("この1時間の行動を決めてください。")
    return "\n".join(lines)
