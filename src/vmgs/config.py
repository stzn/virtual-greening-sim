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
