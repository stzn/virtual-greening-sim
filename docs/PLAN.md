# 仮想移動緑化シミュレータ 開発計画書

> Virtual Mobile Greening Simulator (VMGS)
> 作成日: 2026-10-19 / 対象: Python 3.11+

---

## 0. このドキュメントの位置づけ

- 仕様書（要求）を **実装可能な単位** に分解した計画書。
- 各フェーズに「成果物」「完了条件（Definition of Done）」を明記し、手戻りを防ぐ。
- 未確定事項は「§9 前提と未決事項」にまとめ、着手前にここだけ確認すれば進められるようにする。

### ゴール

1. 実在都市（東京都大手町）の実気象データ上で、仮想の移動緑化ユニットを LLM エージェントが24時間自律運用できる。
2. 「植物の健康維持」と「都市の暑熱緩和」のトレードオフを、エージェントが**ツール呼び出し→観測→判断**のループで解く。
3. エージェント開発の4本柱（ツール呼び出し / 状態管理 / 自己反省 / 可観測性）を、それぞれ**動くコードと計測結果**で習得する。

### 非ゴール（やらないこと）

- 高精度な植物生理・熱流体シミュレーション（数理モデルは「それらしく単純」で良い）
- GUI / Web ダッシュボード（ログ＋グラフ画像で十分。必要になったらフェーズ5以降）
- 実機連携（ただし差し替え可能な設計にはしておく）

---

## 1. 設計方針（最重要の4原則）

| 原則 | 内容 | 理由 |
| --- | --- | --- |
| **振る舞い駆動（BDD）** | 各機能は Gherkin の `.feature`（日本語）でシナリオを先に書き、失敗を確認してから実装する。LLM 部分は「合否」ではなく「評価シナリオ」として扱う（§7） | 「エージェントがどう振る舞うべきか」が仕様＝テスト＝プロンプト改善の物差しとして一本化される |
| **World と Agent の分離** | 物理世界（`world/`）は LLM を一切知らない純粋Python。エージェントはツール経由でのみ世界に触れる | 単体テスト可能。将来ツールの接続先を実機APIへ差し替えるだけで移行できる |
| **決定論的に再現できる** | 気象データはキャッシュ（JSON）から再生可能。乱数はシード固定 | LLMの挙動比較・プロンプト改善の効果測定には「同じ天気」で比較する必要がある |
| **時間は仮想時計が支配** | 「現在時刻」は `SimClock` が持つ。`get_weather` も現実の今ではなく **仮想時刻の気象** を返す | 24時間を数分で回せる。過去日の猛暑日・雨天日をシナリオとして再生できる |

---

## 2. アーキテクチャ

```
┌──────────────────────────────────────────────────────────────┐
│ runner/ (シミュレーション実行)                                │
│   for hour in 0..23:                                          │
│     world.advance(1h)  ← 物理更新（蒸散・発電・降雨）          │
│     agent.invoke(observation)  ← 1時間ごとの意思決定           │
│     logger.record(step)                                       │
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│ agent/ (LangGraph)                                            │
│   [agent_node] ⇄ [tool_node]  … tool_calls が無くなるまで反復   │
│        │  上限: MAX_TOOL_CALLS_PER_TURN（暴走防止）            │
│        ▼                                                      │
│   [finalize_node] 判断理由(rationale)を構造化して返す          │
│   checkpointer: SqliteSaver (thread_id = run_id)              │
└───────────────┬──────────────────────────────────────────────┘
                │ Function Calling (Pydanticスキーマ)
┌───────────────▼──────────────────────────────────────────────┐
│ tools/  get_weather / get_plant_state / move_to_spot /         │
│         irrigate / log_research_note                          │
│   ※ 各ツールは World を操作する薄いアダプタ（ロジックを持たない）│
└───────────────┬──────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────┐
│ world/  SimClock / WeatherProvider / Spot / PlantUnit / World  │
│ weather/ OpenMeteoClient + JSONキャッシュ                      │
└──────────────────────────────────────────────────────────────┘
```

### ディレクトリ構成（予定）

```
virtual-greening-sim/
├── pyproject.toml            # uv 管理
├── .env.example              # APIキーのひな形（.env はコミットしない）
├── README.md
├── docs/
│   ├── PLAN.md               # 本書
│   └── model.md              # 数理モデルの式と係数の根拠（フェーズ1で作成）
├── src/vmgs/
│   ├── config.py             # pydantic-settings で .env / 定数を読む
│   ├── weather/
│   │   ├── open_meteo.py     # API取得（httpx）
│   │   └── cache.py          # data/weather/*.json に保存・再生
│   ├── world/
│   │   ├── clock.py          # SimClock
│   │   ├── spots.py          # Spot A/B/C 定義
│   │   ├── plant.py          # PlantUnit（水分・健康度）
│   │   ├── resources.py      # バッテリー・タンク
│   │   ├── heat.py           # 暑熱緩和スコア
│   │   └── world.py          # 全体を束ねる World / advance()
│   ├── tools/
│   │   ├── schemas.py        # Pydantic 引数・戻り値モデル
│   │   └── tools.py          # 5ツール（LangChain @tool）
│   ├── agent/
│   │   ├── prompts.py        # システムプロンプト（バージョン管理）
│   │   ├── graph.py          # LangGraph 定義
│   │   └── baseline.py       # ルールベース比較用エージェント（LLM不使用）
│   ├── runner/
│   │   ├── simulate.py       # 24h ループ CLI
│   │   └── logger.py         # JSONL + Markdown 研究ログ
│   └── eval/
│       ├── metrics.py        # 評価指標の算出
│       └── report.py         # 比較レポート・グラフ出力
├── data/weather/             # 気象キャッシュ（シナリオ）
├── runs/                     # 実行ログ（gitignore）
├── tests/
│   ├── features/             # Gherkin（日本語）: 決定論的な振る舞い仕様
│   │   ├── world/            #   植物・リソース・スポット
│   │   ├── tools/            #   5ツールの契約
│   │   └── agent/            #   グラフ制御（Fake LLM）
│   ├── step_defs/            # pytest-bdd のステップ実装
│   └── unit/                 # 数式・境界値などの細粒度テスト
├── evals/
│   ├── features/             # Gherkin: LLM の判断品質シナリオ（実LLM・確率的判定）
│   └── step_defs/
└── tasks/
    ├── todo.md
    └── lessons.md
```

---

## 3. ドメインモデル（数理モデルの初期案）

> 係数はすべて `config.py` に集約し、後から調整できるようにする。まず「直感に合う挙動」を優先し、`docs/model.md` に根拠を残す。

### 3.1 仮想スポット

基準座標: **東京都大手町（35.6866, 139.7639）**。スポット間距離は仮想値。

| ID | 名称 | 日射係数 `solar_factor` | 風速係数 `wind_factor` | 地表温度補正 `surface_offset` | 備考 |
| --- | --- | --- | --- | --- | --- |
| A | 芝生広場 | 1.0 | 1.0 | +0℃ | 風通し良 |
| B | ビルの谷間 | 0.2 | 1.5 | −3℃ | 日影・ビル風 |
| C | アスファルト歩道 | 1.0 | 1.0 | +5℃（日射に比例） | ヒートアイランド。冷却需要が最大 |

距離行列（例）: A–B 200m / A–C 150m / B–C 300m → 移動消費 5〜10%に線形マップ。

### 3.2 植物：土壌水分（毎時）

```
有効日射 R = shortwave_radiation[W/m²] × solar_factor
体感気温 T = temperature_2m + surface_offset(R)
有効風速 U = wind_speed_10m × wind_factor

蒸散量 ET [%/h] = k_base
                + k_rad  × (R / 1000)
                + k_temp × max(0, T − 20)
                + k_wind × U
                − k_hum  × (RH / 100)         … 湿度が高いほど減りにくい
土壌水分 M ← clamp(M − ET + 降水量[mm]×k_rain + 散水量[ml]×k_irr, 0, 100)
```

### 3.3 植物：健康度スコア（0〜100）

| 条件 | 毎時の変化 |
| --- | --- |
| 30 ≤ M ≤ 70（適正） | +1（回復、上限100） |
| M < 30（乾燥） | −(30 − M) × 0.3 |
| M > 85（過湿・根腐れ） | −(M − 85) × 0.2 |
| T > 35℃（高温ストレス） | −(T − 35) × 0.5 |

### 3.4 リソース

- **水タンク**: 初期 10,000ml。`irrigate(amount_ml)` で減少。0 未満は拒否（エラーを返す）。
- **バッテリー**: 初期 100%。
  - 移動: 距離に応じて 5〜10% 消費。残量不足なら移動を拒否。
  - 発電: `+ R / 1000 × k_solar`（日向で微増、上限100）。
  - 待機消費: 毎時 −0.5%（何もしなくても減る → 「放置」が最適解にならないように）。

### 3.5 暑熱緩和スコア（都市側の目的関数）

- 散水した水の一部が地表で蒸発し、スポットの地表温度を下げる（気化冷却）。
- `cooling[℃] = min(散水量 × k_cool, cap)` を数時間かけて減衰させる。
- **スポットCで、猛暑時に冷却したときだけ大きく加点** → 植物ケアとのトレードオフを生む。
- 1日の累積を `heat_mitigation_score` とする。

### 3.6 失敗状態（フェイルセーフの練習台）

- バッテリー 0% → 移動不能（`STRANDED`）
- タンク 0ml → 散水不能
- 健康度 0 → 植物枯死（シミュレーション終了）

これらは World 側で **状態（enum）として明示** し、ツールは状態に応じてエラーを返す。エージェントがエラーから回復できるかを観察する。

---

## 4. ツール設計（Pydantic スキーマ）

| ツール | 引数 | 戻り値（要点） | 設計メモ |
| --- | --- | --- | --- |
| `get_weather` | `lat: float`, `lon: float`, `hours_ahead: int = 6` (0〜24) | 現在時刻の気温/日射/風速/湿度/降水 + 先N時間の予報 | **仮想時刻基準**。先読みがあるので「雨の直前に散水」を避けられる |
| `get_plant_state` | なし | 土壌水分、健康度、現在スポット、タンク、バッテリー、ステータス | 観測系。副作用なし |
| `move_to_spot` | `spot_id: Literal["A","B","C"]` | 成功可否、消費バッテリー、新スポット | 同一スポットへの移動はエラー（無駄移動の検出用） |
| `irrigate` | `amount_ml: int` (50〜2000) | 散水後の水分、タンク残量、冷却効果 | 範囲外は Pydantic で弾く（スキーマ違反の挙動確認） |
| `log_research_note` | `text: str` (≤1000字), `tags: list[str]` | 保存先パス | Markdown 追記。エージェントの「研究メモ」= 自己反省の材料 |

- ツールは `World` インスタンスに依存する。LangGraph の state には入れず、**ツール生成関数でクロージャに束縛** する（`make_tools(world) -> list[Tool]`）。テスト時は World を差し替えるだけ。
- docstring は LLM へのプロンプトそのもの。「いつ使うべきか / 使うべきでないか」をフェーズ4で育てる。

---

## 5. エージェント設計（LangGraph）

### 5.1 1ターン（=1仮想時間）のグラフ

```
START → agent_node ──(tool_calls あり)──→ tool_node ─┐
            ▲                                         │
            └─────────────────────────────────────────┘
        agent_node ──(tool_calls なし or 上限到達)──→ finalize_node → END
```

- `agent_node`: LLM 呼び出し（`bind_tools`）。
- `tool_node`: `langgraph.prebuilt.ToolNode`。例外は ToolMessage のエラーとして返し、LLMに再判断させる。
- `finalize_node`: 構造化出力で `TurnDecision(action_summary, rationale, risk_notes)` を生成し、ログ化。
  - ※ モデル内部の思考そのものではなく、**モデルが述べた判断理由** を記録する（再現可能で比較しやすい）。
- 上限: `MAX_TOOL_CALLS_PER_TURN = 8`（無限ループ防止）。

### 5.2 状態（State）

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    sim_time: datetime
    tool_call_count: int
    decision: TurnDecision | None
```

### 5.3 時間をまたぐ記憶

- 毎時の会話を全部持ち越すとトークンが膨らむ → **ターンごとに messages をリセット** し、代わりに
  - 直近3時間の `TurnDecision` 要約
  - `log_research_note` の最新メモ
  を次ターンの入力に含める（「短期記憶の圧縮」を学ぶポイント）。
- チェックポイント: `SqliteSaver`（`runs/<run_id>/checkpoints.sqlite`）。途中から再開・特定時刻の状態を検査できる。

### 5.4 LLM プロバイダ

- `langchain.chat_models.init_chat_model` で **OpenAI / Anthropic を設定だけで切替**。
- 既定: 安価なモデル（例: `gpt-4o-mini` もしくは `claude-haiku-5-5`）。モデル名は `.env` で指定。

### 5.5 比較用ベースライン

- `baseline.py`: if-then ルールのみ（例: 水分<35なら散水、日射>800かつ健康度<60ならBへ退避）。
- **LLM が「ルールより賢いか」を数値で判定するための物差し**。APIコストゼロでパイプライン全体のテストにも使える。

---

## 6. フェーズ別ロードマップ

### フェーズ 0: 準備（0.5日）

- [ ] `uv init` / `pyproject.toml`（python>=3.11）
- [ ] 依存追加: `httpx`, `pydantic`, `pydantic-settings`, `langgraph`, `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langgraph-checkpoint-sqlite`, `langsmith`, `python-dotenv`, `matplotlib`
- [ ] 開発依存: `pytest`, `pytest-bdd`, `ruff`, `mypy`（Pythonなので型チェックは `tsc` の代わりに **`mypy`** を使う）
- [ ] `.env.example`, `.gitignore`, `git init`
- [ ] BDD の土台: `tests/features/` と `tests/step_defs/` を作り、日本語 Gherkin（`# language: ja`）のサンプル1本が通ることを確認
- [ ] pytest マーカー定義: `live`（実LLM/実API）、`eval`（確率的評価）。既定の `pytest` 実行からは除外

**DoD**: `uv run pytest` と `uv run mypy src` が通り、サンプル feature が緑になる。

### フェーズ 1: 仮想環境とツール（2〜3日）

| # | タスク | 成果物 |
| --- | --- | --- |
| 1-0 | **振る舞い仕様を先に書く（Red）**: 植物の乾燥・雨・過湿、タンク/バッテリー不足、同一スポット移動拒否、各ツールの入出力契約 | `tests/features/world/*.feature`, `tests/features/tools/*.feature` |
| 1-1 | Open-Meteo クライアント（hourly: `temperature_2m, relative_humidity_2m, shortwave_radiation, wind_speed_10m, precipitation, precipitation_probability`, `timezone=Asia/Tokyo`） | `weather/open_meteo.py` |
| 1-2 | キャッシュ層：取得結果を `data/weather/<date>_<lat>_<lon>.json` に保存し、オフライン再生 | `weather/cache.py` |
| 1-3 | シナリオ用データ確保：猛暑日・雨天日・平常日の3日分（過去日は Open-Meteo の Historical / `past_days` で取得） | `data/weather/*.json` |
| 1-4 | `SimClock`, `Spot`, `PlantUnit`, `Resources`, `HeatModel`, `World.advance()` | `world/` |
| 1-5 | 数理モデルの式・係数を文書化 | `docs/model.md` |
| 1-6 | Pydantic スキーマ + 5ツール（`make_tools(world)`） | `tools/` |
| 1-7 | ステップ定義を実装し 1-0 の feature を緑にする（Green）。数式の単調性・境界値は `tests/unit/` で補完 | `tests/step_defs/`, `tests/unit/` |
| 1-8 | 手動スクリプト：「何もしない」で24h回し、水分・健康度の推移をグラフ化 | `runs/noop/*.png` |

**DoD**:
- オフライン（キャッシュのみ）で World が24時間回る。
- 「何もしない」と猛暑日に健康度が明確に下がる（＝エージェントが介入する意味がある）ことをグラフで確認。
- `tests/features/world`・`tests/features/tools` の全シナリオと mypy が通る。

### フェーズ 2: 最小 Agent Loop（2〜3日）

| # | タスク |
| --- | --- |
| 2-0 | **グラフ制御の振る舞い仕様（Red）**: 「ツール呼び出しが無ければ終了」「上限到達で打ち切り」「ツールエラーは ToolMessage で返り例外で落ちない」（Fake LLM で決定論的に検証） → `tests/features/agent/*.feature` |
| 2-0b | **判断品質の評価シナリオ（初版）**: 例「雨直前なら散水しない」「猛暑の日向で乾燥なら退避か散水を行う」 → `evals/features/*.feature` |
| 2-1 | システムプロンプト v1（役割、目的関数の優先順位、ツールの使い方、1ターン終了条件） |
| 2-2 | LangGraph の 1ターングラフ（agent ⇄ tool → finalize）実装 |
| 2-3 | 単発実行 CLI: `uv run vmgs turn --scenario heatwave --hour 13 --spot C` |
| 2-4 | 「天気確認 → 状態確認 → 散水/移動 → 終了」が成立するか観察 |
| 2-5 | 異常系: スキーマ違反引数・タンク空・バッテリー不足で LLM がエラーを読み再判断できるか |
| 2-6 | 2-0 のステップ定義を Fake チャットモデル（固定の tool_calls を返す）で実装し緑にする |
| 2-7 | 2-0b の評価シナリオを実LLMで実行（`pytest -m eval`）し、ベースライン値を記録 |

**DoD**:
- 1ターン内で複数ツール呼び出しを経て `TurnDecision` が返る。
- `tests/features/agent` の全シナリオが APIキー無しで通る。
- 評価シナリオの初回スコアが記録されている（この時点では合格基準未達でも可）。

### フェーズ 3: タイムステップ・シミュレーション（3〜4日）

| # | タスク |
| --- | --- |
| 3-0 | **振る舞い仕様（Red）**: 24h 完走、チェックポイントからの再開で状態が一致、枯死・`STRANDED` で正しく終了（ベースラインエージェントで決定論的に検証） |
| 3-1 | 24h ループ（`world.advance` → `agent.invoke` → 記録）と CLI: `uv run vmgs run --scenario heatwave --agent llm` |
| 3-2 | ターン間記憶（直近の判断要約＋研究メモ）の受け渡し |
| 3-3 | SqliteSaver によるチェックポイントと**途中再開** |
| 3-4 | ログ: `runs/<run_id>/steps.jsonl`（毎時の状態・ツール呼び出し・判断理由）＋ `research_notes.md` |
| 3-5 | トレードオフシナリオの作り込み（例: 14時・猛暑・スポットC・水分25%・タンク残り少） |
| 3-6 | ベースラインとの比較実行（同一シナリオ × {baseline, llm}） |
| 3-7 | 結果グラフ（水分・健康度・バッテリー・タンク・スポット位置の時系列） |

**DoD**:
- 3シナリオ × 2エージェントで24hを完走し、ログとグラフが出る。
- 任意の時刻のチェックポイントから再開できる。

### フェーズ 4: 可観測性とプロンプト最適化（2〜3日）

| # | タスク |
| --- | --- |
| 4-1 | LangSmith 接続（`LANGSMITH_TRACING=true` 等を `.env`）。run名・タグに `scenario / prompt_version / model` を付与 |
| 4-2 | 評価指標の実装（下表）と比較レポート `runs/report.md` |
| 4-3 | 典型バグを**評価シナリオとして追加**（例: 「2時間以内に往復移動しない」「降水確率≥60%の1時間前に散水しない」）。LangSmith で見つけた失敗トレースは必ずシナリオ化してから修正する |
| 4-4 | プロンプト／docstring の改善 → 評価シナリオ合格率と指標を比較（v1 → v2 → v3） |
| 4-5 | 自己反省ループ（任意）: 6時間ごとに自分のログを読み返し、方針メモを更新する `reflect_node` を追加 |
| 4-6 | 学びを `tasks/lessons.md` と `docs/` にまとめる |

#### 評価指標

| 指標 | 意味 | 望ましい方向 |
| --- | --- | --- |
| 最終健康度 / 最低健康度 | 植物ケアの質 | 高 |
| 暑熱緩和スコア | 都市への貢献 | 高 |
| 総移動回数・無駄移動回数 | バッテリー効率 | 低 |
| 散水総量・降雨直前散水回数 | 水の効率 | 低 |
| 最終バッテリー / タンク残量 | 持続性 | 高 |
| 総トークン・総レイテンシ・推定コスト | 運用コスト | 低 |
| ツールエラー回数 | ツール定義の分かりやすさ | 低 |

**DoD**:
- LangSmith 上で1ランの全ターンの推論・ツール呼び出し・トークンが追える。
- プロンプト改善前後で、少なくとも2指標が改善したことをレポートで示せる。
- 評価シナリオの合格率が v1 より改善し、その数値がレポートに残っている。

---

## 7. テスト戦略（BDD）

### 7.1 BDD を採用する理由と、AIエージェント開発での注意点

- **採用理由**: エージェント開発の本質は「この状況ならこう振る舞うべき」の積み重ね。Gherkin の Given/When/Then はそのまま
  ①実装の受け入れ条件、②システムプロンプトに書くべき行動規範、③LangSmith で見つけた失敗の再発防止テスト、の3役を兼ねる。
- **注意点**: LLM は非決定的なので、BDD を全層に同じ形で適用すると「たまに落ちるテスト」で開発が止まる。
  そこで **決定論的な層（合否で判定）** と **LLM判断の層（確率で判定）** を明確に分ける。

### 7.2 層ごとの方針

| 層 | 書き方 | 判定 | 実行タイミング | API |
| --- | --- | --- | --- | --- |
| world / weather | `tests/features/world`（pytest-bdd） | 合否（決定論） | 毎変更 | なし（キャッシュJSON） |
| tools | `tests/features/tools`（ツールの入出力契約） | 合否（決定論） | 毎変更 | なし |
| agent graph 制御 | `tests/features/agent`（Fake LLM で tool_calls を固定） | 合否（決定論） | 毎変更 | なし |
| 数式・境界値 | `tests/unit`（素の pytest、parametrize） | 合否 | 毎変更 | なし |
| LLM 判断品質 | `evals/features`（同じ Gherkin 形式） | **N回中M回以上で合格**（例: 5回中4回） | 手動 / プロンプト変更時（`-m eval`） | あり |
| E2E 24h | CLI 実行＋評価指標レポート | 指標の比較 | フェーズ3以降・手動（`-m live`） | あり |

### 7.3 シナリオ例

決定論（`tests/features/world/plant.feature`）:

```gherkin
# language: ja
機能: 土壌水分の変化
  シナリオ: 猛暑の日向では何もしないと土壌水分が減る
    前提 植物ユニットがスポット "C" にいて土壌水分は 50% である
    かつ 気温 35℃・日射 900W/m²・降水 0mm の1時間が与えられる
    もし 1時間経過する
    ならば 土壌水分は 50% より小さい

  シナリオ: タンク残量を超える散水は拒否される
    前提 水タンク残量が 100ml である
    もし 500ml の散水を指示する
    ならば 散水は "タンク残量不足" エラーで拒否される
    かつ 土壌水分とタンク残量は変化しない
```

LLM 判断品質（`evals/features/rain.feature`）:

```gherkin
# language: ja
機能: 降雨を考慮した散水判断
  シナリオ: 1時間後に強い雨が予報されているときは散水しない
    前提 土壌水分 40%・スポット "A"・1時間後の降水確率 80% の状況である
    もし エージェントに1ターン判断させる
    ならば irrigate は呼ばれない
    かつ 5回中4回以上この振る舞いになる
```

### 7.4 開発サイクル

1. 振る舞いを `.feature` に書く → 失敗を確認（Red）
2. ステップ定義と実装 → 緑にする（Green）
3. リファクタ → `uv run ruff check && uv run mypy src && uv run pytest`
4. LLM の失敗を見つけたら：**まず評価シナリオに追加 → プロンプト/docstring を直す → 合格率を比較**

- 毎変更後: `uv run ruff check && uv run mypy src && uv run pytest`（eval / live は除外）
- プロンプト変更時: `uv run pytest -m eval`

---

## 8. リスクと対策

| リスク | 対策 |
| --- | --- |
| Open-Meteo の障害・レート制限 | キャッシュ必須。テストはネットワーク非依存 |
| LLM のコスト膨張（24ターン×数ツール×比較実験） | 安価モデル、ターンごとの messages リセット、ツール上限、ベースラインで事前検証 |
| LLM 出力が非決定的で比較しにくい | `temperature=0`、各条件を複数回（例: 3回）実行して平均 |
| 数理モデルが不自然で「正解」が自明/不可能になる | フェーズ1の noop グラフで難易度を確認し係数を調整 |
| BDD の評価シナリオが LLM の揺らぎで不安定になる | 評価シナリオは通常テストから分離し、N回中M回の確率判定にする。決定論層には実LLMを入れない |
| Gherkin とステップ定義が肥大化して保守が重くなる | ステップ文言は共通語彙（`前提 ... の状況である` 等）に揃え、細かい境界値は `tests/unit` に逃がす |
| ライブラリのAPI変更（LangGraph は更新が速い） | バージョン固定（`uv.lock`）、着手時に公式ドキュメントを確認 |

---

## 9. 前提と未決事項

着手時に以下の前提で進める。変更したい場合はフェーズ0の前に決める。

| 項目 | 前提（暫定） |
| --- | --- |
| フォルダ | `~/virtual-greening-sim` |
| パッケージ管理 | `uv`（インストール済み）、Python 3.11 |
| LLM | OpenAI / Anthropic 両対応。既定モデルは `.env` で指定 |
| 型チェック | Python のため `tsc --noEmit` ではなく `mypy` を使用 |
| 開発手法 | **BDD**（pytest-bdd・日本語 Gherkin）。決定論層は合否判定、LLM 判断は確率判定の評価シナリオ（§7） |
| 研究ログ保存先 | Markdown + JSONL（DB は必要になったら SQLite を追加） |
| 気象の基準日 | 猛暑日・雨天日・平常日を過去データから各1日選定（フェーズ1で決める） |
| ツール5種に加えた追加 | 先読み予報用の `hours_ahead` 引数のみ。新規ツールは追加しない |

---

## 10. 実機移行を見据えたメモ

- `tools/` の実装は `World` を呼ぶだけ → 実機版は `RobotAdapter`（ROS 2 / MQTT / REST）を同じインターフェースで実装すれば差し替え可能。
- `WeatherProvider` を抽象化しておけば、気象庁データや PLATEAU 由来の日影・地表温度データにも差し替えられる。
- フェイルセーフ（`STRANDED` 等）を状態として明示しておくことが、実機での通信断・ハード障害時の状態遷移設計の雛形になる。
