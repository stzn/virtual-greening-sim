# language: ja
機能: 思考モード（reasoning）の構成指定と配線
  hybrid・plan の水の配分は、文章のヒントでも計画観測でも改善しなかった。
  思考モードをオンにして、計画能力がボトルネックかを検証する。思考のオン/オフ以外は変えない。
  思考オフの構成は一切変わらない（過去の比較との互換）。

  # --- 構成の指定 ---
  シナリオ: 構成の指定にプロンプト + think を付けると、思考モードが有効になる
    ならば 構成の指定 "llm:ollama:qwen3:14b:hybrid+think" は プロバイダ "ollama"・モデル "qwen3:14b"・プロンプト "hybrid" と解釈される
    かつ 構成の指定 "llm:ollama:qwen3:14b:hybrid+think" の思考モードは 有効 である
    かつ 構成 "llm:ollama:qwen3:14b:hybrid+think" の名前は "llm_ollama_qwen3_14b_hybrid_think" になる

  シナリオ: プロンプトを省略した +think も指定できる
    ならば 構成の指定 "llm:ollama:qwen3:14b:+think" は プロバイダ "ollama"・モデル "qwen3:14b" で思考モードが有効と解釈される
    かつ 構成 "llm:ollama:qwen3:14b:+think" の名前は "llm_ollama_qwen3_14b_think" になる

  シナリオテンプレート: 思考モードが有効になるのは、+think を付けた構成だけである
    ならば 構成の指定 "<指定>" の思考モードは <有効無効> である

    例:
      | 指定                              | 有効無効 |
      | llm:ollama:qwen3:14b:hybrid+think | 有効     |
      | llm:ollama:qwen3:14b:plan+think   | 有効     |
      | llm:ollama:qwen3:14b:free+think   | 有効     |
      | llm:ollama:qwen3:14b:hybrid       | 無効     |
      | llm:ollama:qwen3:14b:plan         | 無効     |
      | llm:ollama:qwen3:14b:v9           | 無効     |
      | baseline                          | 無効     |

  シナリオ: plan + think は、計画観測も有効のままである
    ならば 構成の指定 "llm:ollama:qwen3:14b:plan+think" の計画観測は 有効 である

  シナリオ: Ollama 以外に +think を付けるとエラーになり、理由が分かる
    ならば 構成の指定 "llm:openai:gpt-4o-mini:hybrid+think" はエラーになり、メッセージに "Ollama" が含まれる

  # --- チャットモデルへの配線 ---
  シナリオ: 構成の思考モードが有効なら、LLM の設定は思考オンで、ハング対策のタイムアウトが付く
    もし 思考モード有効で Ollama の LLM 設定を作る
    ならば 設定の思考モードは有効である
    かつ 設定のタイムアウトは 600 秒である
    かつ 生成されるチャットモデルの reasoning は True である
    かつ 生成されるチャットモデルのタイムアウトは 600 秒である

  シナリオ: 思考モードが無効なら、従来どおり思考オフで、タイムアウトも付かない
    もし 思考モード無効で Ollama の LLM 設定を作る
    ならば 設定の思考モードは無効である
    かつ 設定のタイムアウトは なし である
    かつ 生成されるチャットモデルの reasoning は False である
    かつ 生成されるチャットモデルのタイムアウトは なし である

  # --- 実行の記録 ---
  シナリオ: 思考モードの実行の識別情報に、思考モードの有効が記録される
    ならば 思考モード有効の LLM の識別情報に reasoning=true が含まれる

  シナリオ: 思考モード無効の識別情報には、reasoning のキーを足さない（従来と同一）
    ならば 思考モード無効の LLM の識別情報に reasoning のキーは含まれない
