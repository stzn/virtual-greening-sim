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
