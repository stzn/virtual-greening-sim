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
  - [x] 2-7 評価シナリオのステップ定義・スコア記録（runs/evals/*.json）。`EVAL_FAKE=1` で配線のスモーク確認済み
  - [x] Ollama（ローカルのオープンウェイト）プロバイダ対応、`vmgs turn --provider/--model`
  - [x] 2-4/2-5 実LLM(qwen3:14b)での観察と初回スコア記録 → docs/eval-results.md（v1: ツール呼ばず / v2: ツールは呼ぶが状況に応じた判断になっていない）
  - [x] プロンプト v3〜v5: 判断ルール明示で 5/8 → 7/8（docs/eval-results.md）。往復移動は世界側ガード+last_move観測で対策済み（v6、docs/eval-results.md）。履歴が文章のみのシナリオは0/5のまま（既知）
  - [x] 評価の試行にばらつきを導入: 試行0は条件そのまま、試行1以降は水分（±5）・気温（±2）・日射（±10%）・風（±1.5）・湿度（±15）・バッテリー（±8）を (seed, 試行番号) で揺らす。判断ルールの閾値（水分30/40/50/70/85・気温33・日射700・バッテリー15）は元の値と同じ側にとどめ、タンクとスポットは変えない。過去の評価スコア（runs/evals）は同一試行だったため、新しい揺らぎとの直接比較は参考。qwen3:14b（v9、思考オフ、13シナリオ×5試行）で再計測: 弱い揺らぎ（風・湿度・バッテリーのみ）はツール呼び出しの系列が分かれたのが1シナリオだけ。強い揺らぎでは系列が分かれたのも1シナリオだが、行動要約の文が試行間で分かれたシナリオが1→6に増えた。全て5/5で合格のまま。qwen3 の判断は、この範囲の入力の揺らぎには安定している（標本数は実質1に近いが、要約の文では差が出る）
- [x] フェーズ3: 24h タイムステップ・シミュレーション
  - [x] 3-0 feature（sim/simulation 10件・checkpoint 5件・metrics 6件、agent/baseline 7件＝計28件、Red確認済み）
  - [x] 3-1 24hループ+CLI(`vmgs run`) / 3-2 ターン間記憶 / 3-3 チェックポイントと再開 / 3-4 ログ(steps.jsonl・summary.json・research_notes.md)
  - [x] 3-5 トレードオフ preset(`--preset tradeoff`) / 3-6 比較(`vmgs compare`) / 3-7 結果グラフ(timeseries.png)。全129テスト緑
  - [x] 3-6(LLM) 実LLMで24h実行しベースラインと比較（docs/eval-results.md）。健康度は同等、A↔Bピンポン4回・要約崩れ・宣言のみ1件が判明
- [x] フェーズ3 完了
- [ ] フェーズ4: 可観測性とプロンプト最適化

## 開発ルール（BDD）
- 各フェーズは「.feature を書く（Red）→ 実装（Green）→ リファクタ」の順で進める
- LLM の判断品質は evals/features に書き、`pytest -m eval` で確率判定（PLAN.md §7）

## 追記（2026-10-21）
- [x] 往復移動(ピンポン)対策: 世界側の移動予算（12時間に2回）+ get_plant_state の観測 + プロンプト v8。143テスト緑、評価 10/10、24h実LLMで移動 5→2 回
- [x] 指標の改善（無駄な移動に時間窓・拒否された試行数・要約崩れ・宣言のみ・`vmgs metrics` で再計算）。163テスト緑
- [ ] フェーズ4（LangSmith 可観測性、複数回・複数モデル比較）
- [x] 要約崩れ: 原因は旧パーサーの厳しさ（診断を訂正）。末尾JSON抽出・ツール由来の要約・旧ログ再解釈・v9・評価シナリオ。181テスト緑。記憶の汚染が行動を変えていたと判明（docs/eval-results.md）
- [x] 宣言のみの抑制: 再確認の促し(1回)+要約の置き換え。198テスト緑、実LLM24hで宣言のみ 1→0、回帰評価 13/13
- [ ] フェーズ4: 可観測性と複数回・複数モデル比較
  - [x] 4-0 feature（obs/tracing 10件・obs/telemetry 17件・experiment/variants 7件・experiment/experiment 17件＝計51件、Red確認済み）
  - [x] 4-1 トレース設定（既定オフ・明示+APIキーで有効・警告）と各ターンのメタデータ(run_name/tags/metadata)
  - [x] 4-2 テレメトリ(telemetry.jsonl: トークン・レイテンシ・LLM呼び出し数)とトレースログ(traces.jsonl)、再開時の整合、コスト概算
  - [x] 4-3 バリアント生成(シード付き)と実験ランナー(再利用・再開・失敗の隔離・対比較・警告)、`vmgs experiment`
  - [x] 4-4 実LLMで複数回・複数モデル比較（exp1）を docs/eval-results.md に記録。qwen3:14b v9 はベースラインと引き分け(行動列4/5一致)、gemma3:27b はツール非対応で打ち切り
  - [ ] LangSmith への実送信の確認（APIキー設定後に `--trace`）
  - [x] 天井効果の解消: ストレススイート4状況・天井警告・free プロンプト。306テスト緑。結果は docs/eval-results.md（free は雨天の対で3勝だが資源配分が苦手、v9 はルール従属）
  - [x] ハイブリッド（free + 資源計画のヒント）: 実装・比較済み。314テスト緑。結果は仮説を支持せず（scarce_water 0勝3敗、見通しは増えたが配分に結びつかない）
  - [x] 構造的な観測(計画観測): 実装・比較済み。344テスト緑、変異7件検出。結果は改善なし（観測は届いたが判断理由に1/72しか現れない）
  - [x] ツール側の切り詰め(irrigate): 実装・再比較済み(355テスト緑)。失敗は減ったが健康度は改善せず(free は悪化)。失敗が再判断のシグナルだった
  - [x] 思考モード(`+think`)の実装(372テスト緑・変異7件)と段階1比較: hybrid+think +15.9（早めの退避による。配分の記述なし）、plan+think +8.9（配分の計算が現れた）。事前基準では兆候なし、段階2は見送り。docs/eval-results.md
  - [x] hybrid+think の3バリアント再現: 平均80.3±9.7（思考オフ53.7、baseline 69.8）、baseline に3勝0敗。機構は早めの退避（00〜03時）で、水の配分ではない
  - [x] 退避の先読み分離(early=hybrid+1文): 379テスト緑・変異4件。平均59.9、退避は17/12/14時のまま、hybrid+think(80.3)との差が残る。改善は思考による示唆
  - [x] plan+think 3バリアント: 平均73.5（思考オフplan 54.0、hybrid+think 80.3）、baseline 2勝1敗。夜間散水が消え散水失敗0、計画観測が散水量に使われた。退避は遅く(07/03/07時)hybrid+thinkに届かず
  - [x] plan_early+think 3バリアント: 平均80.6（hybrid+think 80.3、plan+think 73.5）。退避前倒し(04/00/02時)と夜間散水・失敗の抑制は両立。成果はhybrid+thinkと同等。385テスト緑・変異3件
  - [x] バリアント5件で確認: hybrid+think 82.9、plan_early+think 81.3、baseline 74.0。思考オンは baseline に4勝（比2.4〜2.8）、2構成の間に差なし(−1.6±2.1)
  - [x] 別モデル（Haiku・Sonnet、上の項目）／思考オフを5件に揃える（Haiku・qwen3）
  - [x] 切り詰めのオプション化: Params.truncate_irrigation（既定オン）、SimConfig・実験・スイート、CLI `vmgs experiment --no-truncate`。BDD+変異6件検出（M4 は世界への反映を確かめるシナリオを足して検出）。417テスト緑
  - [x] 思考の長さの制限: 構成に `+think=<値>`（Anthropic は effort の low/medium/high、Ollama は出力トークン数の上限 num_predict）。指定なしは従来どおり。識別情報には制限があるときだけ `think_limit` を残す。effort=low は Haiku 5.5・Sonnet 5.5 の API が受理することを確認（各1回）。
  - [x] Haiku の effort での実測（stress13）: 健康度は変わらず（81.4/81.5/80.8）、low で出力 約18%減。詳細は eval-results.md
  - [x] Ollama の num_predict の実測（stress15）: 1000 は成績同等で時間も同じ、500 は思考が切れて行動が出ず崩壊（健康度 10.1）。詳細は eval-results.md
- [x] 別モデル（Claude Haiku 5.5）で scarce_water を5バリアント比較（stress5）。結果は docs/eval-results.md 2026-11-02。
- [x] 思考オフの比較を Haiku でも plan_early 単独で（思考の寄与の分離）／qwen3 と Haiku の差の原因（思考の中身）の確認
- [x] plan_early_move（移動を温存しない一文）を Haiku 5.5 思考オフで5バリアント比較。思考ありと同等。docs/eval-results.md 2026-11-02 追記。
- [x] 一文が不要な退避を誘発しないかを、退避すべきでない状況（normal・rainy 等）で確認／qwen3:14b での再現／散水の失敗（最小量未満）の対処
- [x] plan_early_move を qwen3:14b 思考オフで5バリアント比較。効果なし（Haiku 固有）。docs/eval-results.md 追記。
- [x] plan_early_move を退避が要らない状況（wet_rain / dry_before_rain）で確認。不要な移動は0回。dry_before_rain で plan_early より低い傾向（n=3、原因未確認）。
- [x] dry_before_rain を5バリアントに増やした。plan_early_move は平均 −4.2（SE 2.5）で、偶然の範囲と区別できない。
- [x] plan_early_move を tradeoff（Haiku、5バリアント）で確認。効果なし、退避は遅い傾向（SE 内）。思考ありは安定。docs/eval-results.md 2026-11-03。
- [x] 散水の最小量未満エラーにヒントを追加（d9e4c2d）、stress8 で確認。ヒントが使われたのは失敗6件中1件。
- [x] ヒントを message に入れる／ツールの説明文に書くかを検討（ヒントを読ませる方法）
- [x] ヒントを message に移した（1cfcc9a）。失敗の直後に 50 を指定して回復した割合が 1/6 → 6/7。最終健康度の改善は確認できず。docs/eval-results.md 追記。
- [x] irrigate の説明に残量が少ないときの指定の仕方を追記（dd79e16）。残量が少ないときの最小量未満の要求は 5 件 → 0 件。残りは「少量を狙って 30〜40ml を要求」する別の原因。
- [x] 残った最小量未満の要求の原因を全文で確認: 水の温存を優先して少量（30〜40ml）を狙う判断。失敗は1回で回復し、健康度への影響は見られないため、これ以上は対処しない。
- [x] Sonnet 5.5（思考あり、3構成×5バリアント、stress12）で比較。plan_early 系は全5件で baseline を上回り、退避は0〜2時。plan_early_move と plan_early の差は +1.3 で区別できない。費用 $10.58。思考オフは API が拒否するため不可。docs/eval-results.md 2026-11-04 追記。
- [x] Sonnet 5.5（思考あり）で3構成×5を比較（stress12）、判断理由の全文を確認、まとめと Artifact を更新。思考オフは API が拒否するため不可。
- [ ] 一文を OpenAI 系で試す（API キーの設定が要る）

- [x] 評価を難しくする: evals/features/hard.feature に9シナリオを追加（qwen3 で6件合格・4件未達。docs/eval-results.md 2026-11-05）
- [ ] 未達4件（雨3時間以内、降水量1mm、日影B、33℃ちょうど）を Haiku / Sonnet でも測る
