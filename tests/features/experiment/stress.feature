# language: ja
機能: 余裕の少ない状況（ストレス状況）
  猛暑日24時間の比較では、ベースラインの最終健康度が 100.0±0.0 で天井に張り付き、LLM との差を測れなかった。
  ベースラインが天井に届かず、かつ「ルールに従うだけ」と「状況を読む」を分けられる状況を用意する。
  とくに同じ雨天で、初期の水分によって正解が逆になる2状況（dry_before_rain / wet_rain）を対にする。

  # --- 定義 ---
  シナリオテンプレート: ストレス状況の定義
    もし ストレス状況 "<名前>" を取得する
    ならば 気象は "<気象>" で <時間> 時間・土壌水分 <水分>%・水タンク <タンク>ml・スポット "<スポット>"・<開始>時開始である

    例:
      | 名前            | 気象     | 時間 | 水分 | タンク | スポット | 開始 |
      | scarce_water    | heatwave | 24   | 40   | 1500   | A        | 0    |
      | dry_before_rain | rainy    | 24   | 20   | 10000  | A        | 0    |
      | wet_rain        | rainy    | 24   | 75   | 10000  | A        | 0    |
      | tradeoff        | heatwave | 10   | 25   | 1500   | C        | 14   |

  # --- 天井に張り付かない ---
  シナリオテンプレート: どのストレス状況でも、ベースラインの最終健康度は天井(100)に張り付かない
    もし ストレス状況 "<名前>" で "baseline" をバリアント 5 件で実行する
    ならば "baseline" の最終健康度の平均は 97 未満である

    例:
      | 名前            |
      | scarce_water    |
      | dry_before_rain |
      | wet_rain        |
      | tradeoff        |

  シナリオテンプレート: バリアントによって結果がばらつく（標本数に意味がある）
    もし ストレス状況 "<名前>" で "baseline" をバリアント 5 件で実行する
    ならば "baseline" の最終健康度の標準偏差は 5 以上である

    例:
      | 名前            |
      | scarce_water    |
      | dry_before_rain |
      | tradeoff        |

  # --- 雨天の対: 同じ雨でも、初期の水分で正解が逆になる ---
  シナリオ: dry_before_rain では、ベースラインは放置と全バリアントで同じ結果になる
    もし ストレス状況 "dry_before_rain" で "baseline" と "noop" をバリアント 5 件で実行する
    ならば "noop" は "baseline" に対して 勝ち 0・引き分け 5・負け 0 である

  シナリオ: dry_before_rain では、雨でも乾燥していれば散水する方針がベースラインを大きく上回る
    もし ストレス状況 "dry_before_rain" で "baseline" と "rain_ignoring" をバリアント 5 件で実行する
    ならば "rain_ignoring" の最終健康度の平均は "baseline" より 20 以上高い

  シナリオ: wet_rain では、ベースラインは放置と同じで、雨でも乾燥していれば散水する方針も同等である
    もし ストレス状況 "wet_rain" で "baseline" と "noop" と "rain_ignoring" をバリアント 5 件で実行する
    ならば "noop" は "baseline" に対して 勝ち 0・引き分け 5・負け 0 である
    かつ "rain_ignoring" は "baseline" に対して 勝ち 0・引き分け 5・負け 0 である

  シナリオ: wet_rain では、水を足し続ける方針は悪化する（わな）
    もし ストレス状況 "wet_rain" で "baseline" と "always_irrigate" をバリアント 5 件で実行する
    ならば "always_irrigate" の最終健康度の平均は "baseline" より 10 以上低い

  # --- 水が足りない猛暑 ---
  シナリオ: scarce_water では、早めに日影へ退避して節水する方針がベースラインを上回る
    もし ストレス状況 "scarce_water" で "baseline" と "early_shade" をバリアント 5 件で実行する
    ならば "early_shade" の最終健康度の平均は "baseline" より 5 以上高い
