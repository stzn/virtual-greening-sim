# language: ja
機能: 条件 × seed を回し、同じ seed どうしの差の信頼区間で比べる
  ワーカーの組（seed）による違いが大きいので、条件の良し悪しは、同じ seed で対にした差の
  平均と 95% 信頼区間で比べる。

  シナリオ: サンプルの舞台で、ポッド 1 台と 2 台を 2 seed ずつ回す
    前提 サンプルの舞台 "contracts/fixtures/stage.sample.json" がある
    もし 気象 "normal"・方式 "greedy"・ポッド 1 台と 2 台・2 seed・ワーカー 30 人・1 周で比べる
    ならば 終了コードは 0 になる
    かつ 実行ごとの表には 2 条件 × 2 seed × 2 周 の 8 行がある
    かつ まとめには周の差と方式・台数の差の表がある

  シナリオ: 配車の方式 greedy と coverage を比べる
    前提 サンプルの舞台 "contracts/fixtures/stage.sample.json" がある
    もし 気象 "normal"・方式 "greedy" と "coverage"・ポッド 1 台・2 seed・ワーカー 30 人・1 周で比べる
    ならば 終了コードは 0 になる
    かつ 実行ごとの表には方式 2 通り × 2 seed × 2 周 の 8 行がある
    かつ まとめには周の差と方式・台数の差の表がある
