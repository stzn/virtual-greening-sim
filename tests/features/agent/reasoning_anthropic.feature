# language: ja
機能: Claude（Anthropic）の拡張思考の構成指定と配線
  思考オンが効くのが qwen3:14b に固有かを見るために、クラウドの Claude でも思考のオン/オフだけを切り替えられるようにする。
  Haiku 5.5 などの新しいモデルは温度を受け付けず、思考は adaptive で指定する（budget_tokens は使えない）。
  思考の全文は、本文とは別のブロックで返るので、トレースに残す。

  # --- 構成の指定 ---
  シナリオ: Anthropic にも +think を付けられる
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:hybrid+think" は プロバイダ "anthropic"・モデル "claude-haiku-5-5"・プロンプト "hybrid" と解釈される
    かつ 構成の指定 "llm:anthropic:claude-haiku-5-5:hybrid+think" の思考モードは 有効 である
    かつ 構成 "llm:anthropic:claude-haiku-5-5:hybrid+think" の名前は "llm_anthropic_claude-haiku-5-5_hybrid_think" になる

  シナリオ: OpenAI に +think を付けるとエラーになり、対応するプロバイダが分かる
    ならば 構成の指定 "llm:openai:gpt-4o-mini:hybrid+think" はエラーになり、メッセージに "Anthropic" が含まれる

  # --- チャットモデルへの配線 ---
  シナリオ: 思考オンなら、adaptive の思考を要約つきで要求し、温度は送らない
    もし 思考モード有効で Anthropic の "claude-haiku-5-5" の LLM 設定を作る
    ならば 生成されるチャットモデルの thinking は adaptive で要約つきである
    かつ 生成されるチャットモデルの温度は 指定なし である

  シナリオ: 思考オフなら、思考を明示的に無効にし、温度は送らない
    もし 思考モード無効で Anthropic の "claude-haiku-5-5" の LLM 設定を作る
    ならば 生成されるチャットモデルの thinking は 無効 である
    かつ 生成されるチャットモデルの温度は 指定なし である

  シナリオ: 温度に対応する従来のモデルでは、温度 0 のままにする
    もし 思考モード無効で Anthropic の "claude-haiku-4-5" の LLM 設定を作る
    ならば 生成されるチャットモデルの温度は 0 である

  シナリオ: Ollama の思考の設定は、Anthropic の設定の影響を受けない
    もし 思考モード有効で Ollama の LLM 設定を作る
    ならば 設定の Anthropic の思考は 指定なし である

  # --- 実行の記録 ---
  シナリオ: Anthropic の思考モードの実行にも、reasoning=true が記録される
    ならば 思考モード有効の Anthropic の識別情報に reasoning=true が含まれる

  # --- 思考の全文のトレース ---
  シナリオ: 思考ブロックの全文が、本文とは別に reasoning として残る
    前提 本文 "散水する" と思考 "水が少ないので少量にする" を持つ AI メッセージがある
    ならば トレースの記録の reasoning は "水が少ないので少量にする" である
    かつ トレースの記録の本文は "散水する" である

  シナリオ: 思考ブロックが無いメッセージには reasoning のキーを足さない
    前提 本文 "何もしない" だけの AI メッセージがある
    ならば トレースの記録に reasoning のキーは含まれない

  シナリオ: 要約を省略された思考（空の思考）は記録しない
    前提 本文 "何もしない" と思考 "" を持つ AI メッセージがある
    ならば トレースの記録に reasoning のキーは含まれない

  シナリオ: 複数の思考ブロックは順に連結して残す
    前提 思考 "一つ目" と思考 "二つ目" と本文 "散水する" を持つ AI メッセージがある
    ならば トレースの記録の reasoning は "一つ目\n二つ目" である

  シナリオ: 空の思考ブロックが混ざっても、余計な改行を足さない
    前提 思考 "" と思考 "二つ目" と本文 "散水する" を持つ AI メッセージがある
    ならば トレースの記録の reasoning は "二つ目" である
