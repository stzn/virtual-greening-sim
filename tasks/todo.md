# TODO

詳細は docs/PLAN.md §6 を参照。

- [x] フェーズ0: 準備（uv init / 依存追加 / pytest-bdd の日本語サンプル feature が緑）
- [x] フェーズ1: 仮想環境とツール（55テスト緑・mypy strict/ruff 通過）
  - [x] 1-0 feature作成（world 3本 / tools 1本、計37シナリオ、Red確認済み）
  - [x] 1-1〜1-6 実装（Open-Meteo、キャッシュ、シナリオ3日分、World、5ツール、docs/model.md）
  - [x] 1-7 ステップ定義を実装して Green に / 1-8 noop 24h グラフ（runs/noop/）
- [ ] フェーズ2: 最小 Agent Loop
  - [x] 2-0 グラフ制御の feature（tests/features/agent/graph.feature 11件・Red確認済み）
  - [x] 2-0b 評価シナリオ（evals/features 4本8件・Red確認済み。実行: `uv run pytest evals -m eval`）
  - [x] 2-1 システムプロンプト v1 / 2-2 LangGraph 1ターングラフ（agent⇄tools→finalize）/ 2-3 turn CLI（`uv run vmgs turn --dry-run`）
  - [x] 2-6 グラフ制御のステップ定義（ScriptedChatModel）で Green（全76テスト緑）
  - [ ] 2-4/2-5/2-7 実LLMでの観察・評価シナリオのステップ定義・初回スコア記録（要: .env に APIキー）
- [ ] フェーズ3: 24h タイムステップ・シミュレーション
- [ ] フェーズ4: 可観測性とプロンプト最適化

## 開発ルール（BDD）
- 各フェーズは「.feature を書く（Red）→ 実装（Green）→ リファクタ」の順で進める
- LLM の判断品質は evals/features に書き、`pytest -m eval` で確率判定（PLAN.md §7）
