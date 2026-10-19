"""設定からチャットモデルを作る。プロバイダはOpenAI / Anthropicを設定だけで切り替える。"""

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from vmgs.config import Settings


class MissingApiKeyError(Exception):
    pass


def make_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise MissingApiKeyError("OPENAI_API_KEY が未設定です（.env に設定してください）")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model, temperature=0, api_key=SecretStr(settings.openai_api_key)
        )
    if not settings.anthropic_api_key:
        raise MissingApiKeyError("ANTHROPIC_API_KEY が未設定です（.env に設定してください）")
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model_name=settings.llm_model,
        temperature=0,
        api_key=SecretStr(settings.anthropic_api_key),
        timeout=None,
        stop=None,
    )
