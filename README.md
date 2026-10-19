# 仮想移動緑化シミュレータ（Virtual Mobile Greening Simulator）

実在都市（東京都大手町）の実気象データ（Open-Meteo）と、仮想の植物・移動スポットを舞台に、
LLM エージェントが「植物の健康維持」と「都市の暑熱緩和」を両立するよう自律判断するシミュレーション環境。

- 開発計画: [docs/PLAN.md](docs/PLAN.md)
- 進捗: [tasks/todo.md](tasks/todo.md)
- 学び・修正パターン: [tasks/lessons.md](tasks/lessons.md)

## 技術スタック
Python 3.11+ / uv / pytest-bdd（BDD） / LangGraph / Pydantic / Open-Meteo API / LangSmith / OpenAI or Anthropic API

## ステータス
フェーズ0（基盤構築）完了。次はフェーズ1（仮想環境とツール）
