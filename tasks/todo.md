# TODO

詳細は docs/PLAN.md §6 を参照。

- [x] フェーズ0: 準備（uv init / 依存追加 / pytest-bdd の日本語サンプル feature が緑）
- [ ] フェーズ1: 仮想環境とツール
- [ ] フェーズ2: 最小 Agent Loop
- [ ] フェーズ3: 24h タイムステップ・シミュレーション
- [ ] フェーズ4: 可観測性とプロンプト最適化

## 開発ルール（BDD）
- 各フェーズは「.feature を書く（Red）→ 実装（Green）→ リファクタ」の順で進める
- LLM の判断品質は evals/features に書き、`pytest -m eval` で確率判定（PLAN.md §7）
