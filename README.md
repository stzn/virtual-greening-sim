# 仮想移動緑化シミュレータ（Virtual Mobile Greening Simulator）

実在都市（東京都大手町）の実気象データ（Open-Meteo）と、仮想の植物・移動スポットを舞台に、
LLM エージェントが「植物の健康維持」と「都市の暑熱緩和」を両立するよう自律判断するシミュレーション環境です。
あわせて、**LLM がチェックリスト形式のプロンプトをどれだけ正しく読めるか**を、確率的な評価（BDD）で測る研究基盤でもあります。

- 世界: 移動する植栽が、日向の芝生広場（A）・日影のビルの谷間（B）・アスファルト歩道（C）を行き来する。水と電池に限りがある。
- エージェント: 毎時、ツール（状態・天気の取得、`irrigate`、`move_to_spot`）を呼んで判断する（LangGraph）。
- 評価: 状況を作って 5 回試行し、4 回以上が期待どおりなら通過（Gherkin、日本語）。

## 目次
- [セットアップ](#セットアップ)
- [LLM の選び方](#llm-の選び方)
- [使い方](#使い方)
- [テストと評価](#テストと評価)
- [ディレクトリ構成](#ディレクトリ構成)
- [研究の現状](#研究の現状)
- [ドキュメント](#ドキュメント)

## セットアップ
必要なもの: Python 3.11 以上、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync                              # 依存のインストール
cp .env.example .env                 # LLM の設定（.env はコミットしない）
uv run vmgs turn --dry-run           # 動作確認（Fake LLM、APIキー不要）
```

実気象データは `data/weather/`（猛暑日 `heatwave`・平常日 `normal`・雨天日 `rainy`）にキャッシュ済みです。取り直す場合は次を実行します（Open-Meteo の過去データ、キー不要）。

```bash
uv run python scripts/fetch_scenarios.py
```

## LLM の選び方
`.env` で切り替えます（`.env.example` 参照）。CLI の `--provider` / `--model` でも上書きできます。

| プロバイダ | 設定 | 備考 |
| --- | --- | --- |
| openai / anthropic | `LLM_PROVIDER` + `*_API_KEY` | クラウド LLM。API 費用がかかる |
| ollama | `LLM_PROVIDER=ollama` `LLM_MODEL=qwen3:14b` | ローカル・無料・キー不要。ツール呼び出し対応モデルを選ぶ |

### Ollama（ローカルのオープンウェイト LLM）
```bash
brew install ollama          # または https://ollama.com/download のアプリ版（起動すればサーバーも自動起動）
ollama serve                 # 別ターミナルで起動したままにする（アプリ版は不要）
ollama pull qwen3:14b        # 約9GB
ollama list                  # qwen3:14b が表示されれば準備完了
```

`.env` の例:
```
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
```

- メモリが厳しい場合は `qwen3:8b` にする。
- 1 回の生成が長くなるときは `OLLAMA_TIMEOUT_S=600` のように秒数を指定する。
- 接続できないとき、`vmgs turn` は「判断失敗（接続エラー）」として記録して終了し、評価（`pytest evals`）は理由付きでスキップされる。

## 使い方

### 1 ターンだけ判断させる
```bash
uv run vmgs turn --dry-run                                  # Fake LLM（APIキー不要）
uv run vmgs turn --scenario heatwave --hour 13 --spot C --provider ollama --model qwen3:14b
```

### 24 時間シミュレーション
```bash
uv run vmgs run --scenario heatwave --agent baseline        # ルールベースで24時間
uv run vmgs run --scenario heatwave --agent llm --provider ollama --model qwen3:14b
uv run vmgs run --scenario normal --stop-after 8 --run-id demo   # 8時間で中断
uv run vmgs run --resume demo                                    # チェックポイントから再開
uv run vmgs compare --scenario heatwave --agents baseline,noop   # 同じ気象で比較（llm も指定可）
uv run vmgs compare --preset tradeoff --agents baseline,noop     # 14時・猛暑・歩道C・水分25%・タンク少
```
- `--prompt v21` でシステムプロンプトのバージョンを選べる（既定は最新の採用版）。`--think` で Ollama の思考モードをオンにする。
- 実行ごとに `runs/<run_id>/` へ `steps.jsonl`（毎時の状態・ツール呼び出し・判断）、`summary.json`（終了理由・指標）、
  `checkpoints/`、`research_notes.md`、`timeseries.png` を保存する（`runs/` はコミットしない）。

### 複数回・複数モデルの比較と可観測性
```bash
# 初期条件を少しずつ変えた5バリアントで、構成を比較（完了済みの実行は再利用され、同名で再開できる）
uv run vmgs experiment --name exp1 --scenario heatwave --variants 5 --configs baseline,noop,llm:ollama:qwen3:14b:v21
uv run vmgs metrics <RUN_ID> ...        # 保存済みの実行から指標を再計算して並べる
uv run vmgs telemetry <RUN_ID>          # トークン・レイテンシ・コスト概算
uv run vmgs run --agent llm --trace     # LangSmith へ送信（外部サービス。プロンプトと応答が送られる。要 APIキー）
```
実行ごとに `telemetry.jsonl`（トークン・レイテンシ）と `traces.jsonl`（LLM への入出力。次ターンへ渡した記憶も確認できる）を保存する。

#### 余裕の少ない状況（ストレススイート）
猛暑日 24 時間ではベースラインが健康度 100 の天井に張り付き、構成間の差を測れない。ベースラインが天井に届かない 4 状況で比較する。
```bash
uv run vmgs experiment --suite stress --name s1 --variants 5 --configs baseline,noop,llm:ollama:qwen3:14b:v21
uv run vmgs experiment --suite stress --conditions dry_before_rain,wet_rain ...   # 状況を絞る
```
`:free` はチェックリストなしのプロンプト（世界のしくみ・ツール・回答形式のみ）。

## テストと評価
通常のテストと、LLM を呼ぶ評価は分かれています。

```bash
uv run pytest                          # 通常のテスト（LLM を呼ばない。評価と live は除外）
uv run ruff check . && uv run ruff format --check .
uv run mypy src tests evals scripts    # 型検査（strict）
```

- `tests/features/` の Gherkin（日本語）が世界・ツール・エージェントの仕様です。
- **評価**（`evals/features/`、全 59 シナリオ）は LLM の判断品質を測ります。各シナリオを 5 回試行（初期条件を少し揺らす）し、4 回以上が期待どおりなら通過です。結果は `runs/evals/` に保存されます。

```bash
# 全評価（ローカルの qwen3:14b で約 2 時間。API 費用なし）
OLLAMA_TIMEOUT_S=600 LLM_PROVIDER=ollama LLM_MODEL=qwen3:14b uv run pytest evals -m eval -v
# 一部だけ（テスト名＝シナリオ名の一部で絞る。例: 「ちょうど」を含む 14 件）
LLM_PROVIDER=ollama LLM_MODEL=qwen3:14b uv run pytest evals -m eval -v -k "ちょうど"
uv run pytest evals -m eval --collect-only -q    # シナリオ名の一覧（実行はしない）
# プロンプトのバージョンを指定する
EVAL_PROMPT=v21 LLM_PROVIDER=anthropic LLM_MODEL=claude-haiku-5-5 uv run pytest evals -m eval -v
```

評価の `.feature` は、境界値（`boundary`）、条件の組み合わせ（`combo`・`conjunction`）、雨・猛暑・資源・形式など、観点ごとに分かれています。
Anthropic や OpenAI で回すと API 費用がかかるため、まずローカルの Ollama で試してください。

## ディレクトリ構成
```
src/vmgs/
  world/        世界のモデル（植物・スポット・時計・パラメータ・毎時の更新）
  weather/      Open-Meteo クライアントとキャッシュ
  tools/        エージェントに渡すツール（状態・天気・散水・移動）
  agent/        LangGraph のグラフ、プロンプト（v1〜）、ランナー、LLM 生成
  sim/          24時間シミュレーションの実行・記録・保存
  experiment/   複数構成×バリアントの比較、ストレススイート
  eval/         評価の試行・述語・指標・レポート
  obs/          テレメトリ・トレース
  cli.py        `vmgs` コマンド
tests/          通常のテスト（features/ が仕様、step_defs/、unit/）
evals/          LLM 判断品質の評価（features/、step_defs/）
scripts/        気象データの取得など
data/weather/   キャッシュ済みの実気象
docs/           設計・研究の記録
tasks/          進捗（todo）と学び（lessons）
```

## 研究の現状
チェックリスト形式のプロンプトの「かつ」「未満」などの読み取りを、プロンプトの版（v1〜v22）の変更で改善してきました。

- 既定は **v21**。全 51 シナリオを qwen3:14b・Claude Haiku 5.5・Claude Sonnet 5.5 のすべてが通過します（境界値 10 件を含む）。
- qwen3 だけで足した「行の組み合わせ」8 件のうち、7 件通過・1 件は既知の未達（直前に B から A へ移動した直後の猛暑・乾燥で B への移動を試みる。最終行動は世界のガードで正しい）。
- 効いたこと: 世界側で条件の真偽を計算して渡す（観測キー）、誤読する 1 点だけに添える短い説明。効かなかったこと: 文章を長く強くする、否定を含む複合条件。
- v22（往復を観測キーで渡す）は、狙った件は直ったが別の 2 件が退行したため不採用です。

## ドキュメント
| 文書 | 内容 |
| --- | --- |
| [docs/PLAN.md](docs/PLAN.md) | 開発計画 |
| [docs/model.md](docs/model.md) | 数理モデル（植物・スポット・毎時の更新式） |
| [docs/prompt-history.md](docs/prompt-history.md) | プロンプト v1〜v22 の経緯・到達点・限界（まずここ） |
| [docs/eval-results.md](docs/eval-results.md) | 各評価の結果と実験の記録 |
| [docs/research-summary.md](docs/research-summary.md) | 研究のまとめ: 資源配分のボトルネックは「計画能力」か |
| [tasks/todo.md](tasks/todo.md) | 進捗 |
| [tasks/lessons.md](tasks/lessons.md) | 学び・修正パターン |

## 技術スタック
Python 3.11+ / uv / pytest-bdd（BDD） / LangGraph / Pydantic / Open-Meteo API / LangSmith / OpenAI・Anthropic・Ollama
