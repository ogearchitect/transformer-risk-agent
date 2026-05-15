from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from src.config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = messages[-1].content if messages else ""
        tool_context = kwargs.get("tool_context") or "No external tool context was provided."
        text = (
            "[Mock LLM Fallback]\n"
            f"User intent: {prompt}\n"
            "Using tool-grounded synthetic fleet results, here is a deterministic summary:\n"
            f"{tool_context}\n"
            "Recommendation: prioritize highest monetized-risk assets and validate with a field inspection plan."
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_llm() -> BaseChatModel:
    if OPENAI_API_KEY and LLM_PROVIDER != "mock":
        kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0.1}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return ChatOpenAI(**kwargs)
    return MockChatModel()
