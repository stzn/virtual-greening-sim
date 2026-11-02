"""設定。秘密情報は .env（コミットしない）から読む。"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Qwen3 等の思考(thinking)モード。有効だと1呼び出し数十秒かかるため既定はオフ
    ollama_reasoning: bool = False
    # Ollama への1リクエストの上限秒数。思考モードは長く（または循環して）実験全体を止めうるため、思考の構成で設定する
    ollama_timeout_s: float | None = None
    # Claude の拡張思考。None は未指定（thinking を送らない）。構成の +think で adaptive、無しで disabled にする
    anthropic_thinking: Literal["adaptive", "disabled"] | None = None

    # 思考の長さの制限（None は制限なし）。Anthropic は effort、Ollama は出力トークン数の上限（思考の途中で切れうる）
    anthropic_effort: Literal["low", "medium", "high"] | None = None
    ollama_num_predict: int | None = None

    # LangSmith（外部サービスへ送信する。既定はオフ。obs/tracing.py で解決する）
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "vmgs"
