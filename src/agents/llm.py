from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

try:
    from langchain_openai import AzureChatOpenAI
except ImportError:  # pragma: no cover - optional Azure backend import
    AzureChatOpenAI = None

try:
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider
except ImportError:  # pragma: no cover - optional Azure identity import
    DefaultAzureCredential = None
    get_bearer_token_provider = None

from src.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)


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


class _ToolContextAdapter(BaseChatModel):
    """Wraps a real LangChain chat model so callers can pass `tool_context=...`
    to `invoke()` the same way MockChatModel accepts it.

    The kwarg is spliced into the prompt as a system-style preface and is
    NEVER forwarded to the underlying provider (e.g. OpenAI Completions.create
    rejects unknown kwargs)."""

    inner: BaseChatModel

    @property
    def _llm_type(self) -> str:
        return f"tool-context-adapter[{self.inner._llm_type}]"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_context = kwargs.pop("tool_context", None)
        if tool_context:
            preface = (
                "You are the Transformer Risk Agent. Use the following tool-grounded "
                "context to answer concisely and accurately.\n\n"
                f"TOOL CONTEXT:\n{tool_context}\n\n"
            )
            adapted: list[BaseMessage] = list(messages)
            if adapted:
                first = adapted[0]
                adapted[0] = type(first)(content=preface + str(first.content))
            else:
                from langchain_core.messages import HumanMessage
                adapted = [HumanMessage(content=preface)]
        else:
            adapted = messages
        return self.inner._generate(adapted, stop=stop, run_manager=run_manager, **kwargs)


def _wrap(model: BaseChatModel) -> BaseChatModel:
    if isinstance(model, MockChatModel):
        return model
    return _ToolContextAdapter(inner=model)


def _azure_token_provider():
    if DefaultAzureCredential is None or get_bearer_token_provider is None:
        return None
    return get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )


def get_llm() -> BaseChatModel:
    if LLM_PROVIDER == "mock":
        return MockChatModel()
    if AZURE_OPENAI_ENDPOINT:
        if AzureChatOpenAI is None:
            return MockChatModel()

        kwargs = {
            "azure_endpoint": AZURE_OPENAI_ENDPOINT,
            "azure_deployment": AZURE_OPENAI_DEPLOYMENT,
            "api_version": AZURE_OPENAI_API_VERSION,
            "temperature": 0.1,
            "timeout": 25.0,
            "max_retries": 1,
        }
        if AZURE_OPENAI_API_KEY:
            return _wrap(AzureChatOpenAI(api_key=AZURE_OPENAI_API_KEY, **kwargs))

        token_provider = _azure_token_provider()
        if token_provider is not None:
            return _wrap(AzureChatOpenAI(azure_ad_token_provider=token_provider, **kwargs))
        return MockChatModel()
    if OPENAI_API_KEY:
        kwargs = {"model": OPENAI_MODEL, "api_key": OPENAI_API_KEY, "temperature": 0.1, "timeout": 25.0, "max_retries": 1}
        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL
        return _wrap(ChatOpenAI(**kwargs))
    return MockChatModel()
