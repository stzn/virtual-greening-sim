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
