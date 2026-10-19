"""固定の応答を返すFakeチャットモデル。グラフ制御のテストとdry-runに使う。"""

from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def tool_call_message(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedChatModel(BaseChatModel):
    """responses を順に返す。repeat_last=True なら最後の応答を繰り返す。fail=True なら例外。"""

    responses: list[AIMessage]
    repeat_last: bool = False
    fail: bool = False
    _index: int = PrivateAttr(default=0)
    _inputs: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    @property
    def inputs(self) -> list[list[BaseMessage]]:
        """各LLM呼び出しに渡されたメッセージ列。"""
        return self._inputs

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._inputs.append(list(messages))
        if self.fail:
            raise RuntimeError("scripted LLM failure")
        i = self._index
        if i >= len(self.responses):
            if not self.repeat_last:
                raise RuntimeError("ScriptedChatModel: 用意した応答を使い切りました")
            i = len(self.responses) - 1
        self._index += 1
        # 同じメッセージの再利用を避ける: add_messages は id が同じだと上書きしてしまうため、
        # 毎回コピーして id を外し、繰り返し時は tool_call の id も変える
        update: dict[str, Any] = {"id": None}
        base = self.responses[i]
        if base.tool_calls and self._index > len(self.responses):
            update["tool_calls"] = [
                {**tc, "id": f"{tc['id']}_{self._index}"} for tc in base.tool_calls
            ]
        msg = base.model_copy(update=update)
        return ChatResult(generations=[ChatGeneration(message=msg)])
