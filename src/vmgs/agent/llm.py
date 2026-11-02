"""設定からチャットモデルを作る。OpenAI / Anthropic / Ollama（ローカルのオープンウェイト）を切り替える。"""

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from vmgs.config import Settings


class MissingApiKeyError(Exception):
    pass


# 温度を受け付けないモデル（指定すると API が拒否する。モデルの profile で temperature=false）
_NO_TEMPERATURE_PREFIXES = ("claude-haiku-5", "claude-sonnet-5", "claude-opus-5", "claude-fable-5")

_ANTHROPIC_THINKING: dict[str, dict[str, str]] = {
    # 新しいモデルは budget_tokens を受け付けない。要約つきで返してもらい、トレースに残す
    "adaptive": {"type": "adaptive", "display": "summarized"},
    "disabled": {"type": "disabled"},
}


def make_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=0,
            base_url=settings.ollama_base_url,
            reasoning=settings.ollama_reasoning,
            num_predict=settings.ollama_num_predict,
            client_kwargs=(
                {"timeout": settings.ollama_timeout_s} if settings.ollama_timeout_s else {}
            ),
        )
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

    thinking = _ANTHROPIC_THINKING.get(settings.anthropic_thinking or "")
    temperature = None if settings.llm_model.startswith(_NO_TEMPERATURE_PREFIXES) else 0
    return ChatAnthropic(
        model_name=settings.llm_model,
        temperature=temperature,
        thinking=thinking,
        effort=settings.anthropic_effort,
        api_key=SecretStr(settings.anthropic_api_key),
        timeout=None,
        stop=None,
    )
