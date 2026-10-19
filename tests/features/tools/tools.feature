# language: ja
機能: エージェント向け5ツールの契約
  ツールはWorldを操作する薄いアダプタであり、不正入力は例外ではなく構造化エラーで返す

  背景:
    前提 仮想時刻が 13:00 の World が用意されている
    かつ 植物ユニットがスポット "C" にいる

  # --- get_weather ---
  シナリオ: get_weather は仮想時刻の気象と先読み予報を返す
    もし get_weather を lat=35.6866 lon=139.7639 hours_ahead=3 で呼ぶ
    ならば 結果に現在の 気温・日射・風速・湿度・降水 が含まれる
    かつ 予報は 3 件である

  シナリオ: hours_ahead が範囲外ならスキーマ違反エラーになる
    もし get_weather を lat=35.6866 lon=139.7639 hours_ahead=99 で呼ぶ
    ならば ツールはエラー "VALIDATION_ERROR" を返す

  シナリオ: get_weather は現実の現在時刻ではなく仮想時刻に追従する
    もし 仮想時刻を 1時間進める
    かつ get_weather を lat=35.6866 lon=139.7639 hours_ahead=0 で呼ぶ
    ならば 結果の時刻は 14:00 である

  # --- get_plant_state ---
  シナリオ: get_plant_state は全リソースと状態を返す
    もし get_plant_state を呼ぶ
    ならば 結果に 土壌水分・健康度・現在スポット・水タンク残量・バッテリー残量・ステータス が含まれる

  シナリオ: get_plant_state は状態を変化させない
    もし get_plant_state を2回呼ぶ
    ならば 2回の結果は同一である

  # --- move_to_spot ---
  シナリオ: move_to_spot は成功時に消費バッテリーと新スポットを返す
    もし move_to_spot を spot_id="B" で呼ぶ
    ならば 結果は成功である
    かつ 結果の新スポットは "B" である
    かつ 結果に消費バッテリーが含まれる

  シナリオ: 存在しないスポットはスキーマ違反エラーになる
    もし move_to_spot を spot_id="Z" で呼ぶ
    ならば ツールはエラー "VALIDATION_ERROR" を返す
    かつ 現在のスポットは "C" のままである

  シナリオ: 同一スポットへの移動はエラーになる
    もし move_to_spot を spot_id="C" で呼ぶ
    ならば ツールはエラー "ALREADY_AT_SPOT" を返す

  # --- irrigate ---
  シナリオ: irrigate は散水後の水分・タンク残量・冷却効果を返す
    もし irrigate を amount_ml=500 で呼ぶ
    ならば 結果は成功である
    かつ 結果に 散水後の土壌水分・水タンク残量・冷却効果 が含まれる

  シナリオテンプレート: irrigate の散水量が範囲外ならスキーマ違反エラーになる
    もし irrigate を amount_ml=<量> で呼ぶ
    ならば ツールはエラー "VALIDATION_ERROR" を返す
    かつ 水タンク残量は変化しない

    例:
      | 量   |
      | 0    |
      | 49   |
      | 2001 |
      | -100 |

  シナリオ: タンク残量不足の散水は構造化エラーになる
    前提 水タンク残量が 100ml である
    もし irrigate を amount_ml=500 で呼ぶ
    ならば ツールはエラー "INSUFFICIENT_WATER" を返す
    かつ エラーに現在のタンク残量が含まれる

  # --- log_research_note ---
  シナリオ: log_research_note はMarkdownに追記して保存先を返す
    もし log_research_note を text="猛暑のためBへ退避" tags=["heat","move"] で呼ぶ
    ならば 結果は成功である
    かつ 研究ログファイルに "猛暑のためBへ退避" が追記されている

  シナリオ: 1000字を超えるメモはスキーマ違反エラーになる
    もし log_research_note を 1001文字の text で呼ぶ
    ならば ツールはエラー "VALIDATION_ERROR" を返す

  # --- 全ツール共通 ---
  シナリオ: 枯死後は副作用を伴うツールが拒否される
    前提 ユニットが枯死している
    もし irrigate を amount_ml=100 で呼ぶ
    ならば ツールはエラー "UNIT_DEAD" を返す

  シナリオ: ツールは例外を投げずに常にエラーを構造化して返す
    もし 不正な引数で全ツールを順に呼ぶ
    ならば どのツールも例外を送出しない
