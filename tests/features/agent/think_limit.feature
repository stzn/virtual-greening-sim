# language: ja
機能: 思考の長さの制限（構成の +think=<値>）
  思考ありは qwen3 で 1 ステップ約 91 秒かかる。思考を短くしたとき、効果がどこまで残るかを見るために、構成で思考の長さを制限できるようにする。
  Anthropic は effort（low / medium / high）、Ollama は出力トークン数の上限（num_predict）で制限する。
  指定が無ければ従来どおり（過去の結果と比較できるよう、識別情報にもキーを足さない）。

  # --- 構成の指定 ---
  シナリオ: Anthropic の構成に effort を付けられる
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early+think=low" は プロバイダ "anthropic"・モデル "claude-haiku-5-5"・プロンプト "plan_early" と解釈される
    かつ 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early+think=low" の思考の制限は "low" である
    かつ 構成 "llm:anthropic:claude-haiku-5-5:plan_early+think=low" の名前は "llm_anthropic_claude-haiku-5-5_plan_early_think-low" になる

  シナリオ: Ollama の構成に出力トークン数の上限を付けられる
    ならば 構成の指定 "llm:ollama:qwen3:14b:plan_early+think=2000" は プロバイダ "ollama"・モデル "qwen3:14b"・プロンプト "plan_early" と解釈される
    かつ 構成の指定 "llm:ollama:qwen3:14b:plan_early+think=2000" の思考の制限は "2000" である

  シナリオ: 制限を付けない +think は従来どおりで、名前も変わらない
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early+think" の思考の制限は なし である
    かつ 構成 "llm:anthropic:claude-haiku-5-5:plan_early+think" の名前は "llm_anthropic_claude-haiku-5-5_plan_early_think" になる

  シナリオ: Anthropic に数値の制限を付けるとエラーになる
    ならば 構成の指定 "llm:anthropic:claude-haiku-5-5:plan_early+think=2000" はエラーになり、メッセージに "low / medium / high" が含まれる

  シナリオ: Ollama に effort を付けるとエラーになる
    ならば 構成の指定 "llm:ollama:qwen3:14b:plan_early+think=low" はエラーになり、メッセージに "正の整数" が含まれる

  シナリオ: Ollama の上限が 0 だとエラーになる
    ならば 構成の指定 "llm:ollama:qwen3:14b:plan_early+think=0" はエラーになり、メッセージに "正の整数" が含まれる

  # --- チャットモデルへの配線 ---
  シナリオ: Anthropic は effort を送り、思考は adaptive のままにする
    もし 思考の制限 "low" つきで Anthropic の "claude-haiku-5-5" の LLM 設定を作る
    ならば 生成されるチャットモデルの reasoning_effort は "low" である
    かつ 生成されるチャットモデルの thinking は adaptive で要約つきである

  シナリオ: 制限が無ければ Anthropic は effort を送らない
    もし 思考の制限なしで Anthropic の "claude-haiku-5-5" の LLM 設定を作る
    ならば 生成されるチャットモデルの reasoning_effort は 指定なし である

  シナリオ: Ollama は num_predict を送り、思考モードは有効のままにする
    もし 思考の制限 "2000" つきで Ollama の "qwen3:14b" の LLM 設定を作る
    ならば 生成される Ollama のチャットモデルの num_predict は 2000 である
    かつ 生成される Ollama のチャットモデルの reasoning は 有効 である

  シナリオ: 制限が無ければ Ollama は num_predict を送らない
    もし 思考の制限なしで Ollama の "qwen3:14b" の LLM 設定を作る
    ならば 生成される Ollama のチャットモデルの num_predict は 指定なし である

  # --- 識別情報 ---
  シナリオ: 制限があれば識別情報に残し、無ければキーを足さない
    ならば 思考の制限 "low" の識別情報に think_limit=low が含まれる
    かつ 思考の制限なしの識別情報に think_limit のキーは含まれない
