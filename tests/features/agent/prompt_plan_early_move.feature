# language: ja
機能: 退避を移動の温存より優先する説明を足したプロンプト（plan_early_move）
  Haiku 5.5 の思考オフは、退避の先読みの一文（plan_early）が入っていても、移動を「温存する資源」として扱い、
  散水を先にして退避を先送りした（v4 は退避せず、v3 は水が尽きてから退避）。思考オンは同じ判断点で先に退避した。
  plan_early_move は plan_early に、移動は温存する資源ではないという考え方を1文だけ足したもの。
  思考オフでも、移動の温存が解けて退避が早まるかを見る。

  シナリオ: plan_early_move の plan_early との差分は、移動の説明の1文だけである
    前提 プロンプト "plan_early_move" を取得する
    ならば プロンプトから移動の説明を取り除くと "plan_early" と完全に一致する

  シナリオ: 移動の説明は考え方だけで、数字を含まない
    ならば 移動の説明に数字は含まれない
    かつ 移動の説明に次の語が含まれる
      | 語   |
      | 移動 |
      | 温存 |
      | 退避 |

  シナリオ: plan_early_move も、判断の閾値・チェックリスト・具体的な散水量・時刻を含まない
    前提 プロンプト "plan_early_move" を取得する
    ならば プロンプトに次の判断基準は含まれない
      | 基準           |
      | チェックリスト |
      | 33℃            |
      | 700W/m²        |
      | 降水確率 60%   |
      | 40% 未満       |
      | バッテリー 15% |
      | 非該当         |
      | 1000ml         |
      | 500ml          |
      | 0時            |
      | 3時            |
    かつ プロンプトに表形式の行は含まれない
    かつ プロンプトに "#" で始まる行は含まれない

  シナリオ: plan_early_move は既定ではなく、名前で選んだときだけ使われる
    ならば 既定のプロンプトは "plan_early_move" ではない
    かつ プロンプト名 "plan_early_move" は選択肢に含まれる

  シナリオ: 構成の指定でプロンプトに plan_early_move を指定でき、plan_early と取り違えない
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early_move" は プロバイダ "anthropic"・モデル "claude-haiku-5-5"・プロンプト "plan_early_move" と解釈される
    かつ 構成 "llm:anthropic:claude-haiku-5-5:plan_early_move" の名前は "llm_anthropic_claude-haiku-5-5_plan_early_move" になる
    かつ 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early_move" の思考モードは 無効 である

  シナリオ: 計画観測が有効になるのは plan_early_move も同じである（観測の説明を含むため）
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early_move" の計画観測は 有効 である
