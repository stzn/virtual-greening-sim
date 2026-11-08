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


# 思考オフの Ollama の出力トークン数の上限（指定が無いとき）。判断メモ + ツール呼び出し + 最終回答は数百トークンで足りる。
# ollama の読み取りタイムアウトはチャンクごとで、出力が出続ける限り切れないため、生成が終わらない試行が全体を止めうる（v18 で発生）
OLLAMA_SAFETY_NUM_PREDICT = 4096


def make_chat_model(settings: Settings) -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=0,
            base_url=settings.ollama_base_url,
            reasoning=settings.ollama_reasoning,
            num_predict=_ollama_num_predict(settings),
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


def _ollama_num_predict(settings: Settings) -> int | None:
    """明示の指定を優先。無ければ、思考オフのときだけ安全上限を付ける（思考モードは長い出力が正常なので制限しない）。"""
    if settings.ollama_num_predict is not None:
        return settings.ollama_num_predict
    return None if settings.ollama_reasoning else OLLAMA_SAFETY_NUM_PREDICT
