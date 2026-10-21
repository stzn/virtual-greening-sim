# 仮想移動緑化シミュレータ（Virtual Mobile Greening Simulator）

実在都市（東京都大手町）の実気象データ（Open-Meteo）と、仮想の植物・移動スポットを舞台に、
LLM エージェントが「植物の健康維持」と「都市の暑熱緩和」を両立するよう自律判断するシミュレーション環境。

- 開発計画: [docs/PLAN.md](docs/PLAN.md)
- 進捗: [tasks/todo.md](tasks/todo.md)
- 学び・修正パターン: [tasks/lessons.md](tasks/lessons.md)

## 技術スタック
Python 3.11+ / uv / pytest-bdd（BDD） / LangGraph / Pydantic / Open-Meteo API / LangSmith / OpenAI or Anthropic API

## ステータス
フェーズ2（最小 Agent Loop）のグラフ制御まで完了。実LLMでの評価（2-7）は APIキー設定後

## LLM の選び方
`.env` で切り替える（`.env.example` 参照）。

| プロバイダ | 設定 | 備考 |
| --- | --- | --- |
| openai / anthropic | `LLM_PROVIDER` + `*_API_KEY` | クラウドLLM |
| ollama | `LLM_PROVIDER=ollama` `LLM_MODEL=qwen3:14b` | ローカル・無料・キー不要。ツール呼び出し対応モデルを選ぶ |

```bash
# 動作確認（APIキー不要）
uv run vmgs turn --dry-run
# 1ターンだけ実LLMに判断させる
uv run vmgs turn --scenario heatwave --hour 13 --spot C --provider ollama --model qwen3:14b
# LLM判断品質の評価（5回試行、結果は runs/evals/ に保存）
uv run pytest evals -m eval
```

### Ollama（ローカルのオープンウェイトLLM）のセットアップ

```bash
brew install ollama          # または https://ollama.com/download のアプリ版（起動すればサーバーも自動起動）
ollama serve                 # 別ターミナルで起動したままにする（アプリ版は不要）
ollama pull qwen3:14b        # 約9GB。ツール呼び出し(Function Calling)対応モデルを選ぶ
ollama list                  # qwen3:14b が表示されれば準備完了
```

`.env` に次を設定する（ファイルは `~/virtual-greening-sim/.env`、コミットしない）:

```
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:14b
OLLAMA_BASE_URL=http://localhost:11434
```

- メモリが厳しい場合は `qwen3:8b` に変更する。
- Qwen3 は思考(thinking)を出力するため、最終回答のJSON解釈に影響することがある（問題が出たら要対応）。
- 接続できない場合、`vmgs turn` は「判断失敗（接続エラー）」として記録して終了し、評価(`pytest evals`)は理由付きでスキップされる。

## 24時間シミュレーション
```bash
uv run vmgs run --scenario heatwave --agent baseline        # ルールベースで24時間
uv run vmgs run --scenario heatwave --agent llm --provider ollama --model qwen3:14b
uv run vmgs run --scenario normal --stop-after 8 --run-id demo   # 8時間で中断
uv run vmgs run --resume demo                                    # チェックポイントから再開
uv run vmgs compare --scenario heatwave --agents baseline,noop   # 同じ気象で比較（llm も指定可）
uv run vmgs compare --preset tradeoff --agents baseline,noop     # 14時・猛暑・歩道C・水分25%・タンク少
```
実行ごとに `runs/<run_id>/` へ `steps.jsonl`（毎時の状態・ツール呼び出し・判断）、`summary.json`（終了理由・指標）、
`checkpoints/`、`research_notes.md`、`timeseries.png` を保存する。

## 複数回・複数モデルの比較と可観測性
```bash
# 初期条件を少しずつ変えた5バリアントで、構成を比較（完了済みの実行は再利用され、同名で再開できる）
uv run vmgs experiment --name exp1 --scenario heatwave --variants 5 --configs baseline,noop,llm:ollama:qwen3:14b:v9
uv run vmgs telemetry <RUN_ID>          # トークン・レイテンシ・コスト概算
uv run vmgs run --agent llm --trace     # LangSmith へ送信（外部サービス。プロンプトと応答が送られる。要 APIキー）
```
実行ごとに `telemetry.jsonl`（トークン・レイテンシ）と `traces.jsonl`（LLMへの入出力。次ターンへ渡した記憶も確認できる）を保存する。

### 余裕の少ない状況（ストレススイート）
猛暑日24時間ではベースラインが健康度100の天井に張り付き、構成間の差を測れない。ベースラインが天井に届かない4状況で比較する。
```bash
uv run vmgs experiment --suite stress --name s1 --variants 5 --configs baseline,noop,llm:ollama:qwen3:14b:v9,llm:ollama:qwen3:14b:free
uv run vmgs experiment --suite stress --conditions dry_before_rain,wet_rain ...   # 状況を絞る
```
`:free` はチェックリストなしのプロンプト（世界のしくみ・ツール・回答形式のみ）。
