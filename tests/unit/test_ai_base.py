from __future__ import annotations

from typing import Any

import pytest

from app.ai.base import ChatMessage, LLMProvider, LLMResponse


class FakeLLMProvider:
    """A minimal fake satisfying the LLMProvider protocol structurally."""

    name = "fake"

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        text = f"echo: {last_user.content}" if last_user else "no user message"
        return LLMResponse(content=text)


def test_fake_provider_satisfies_protocol() -> None:
    provider: LLMProvider = FakeLLMProvider()
    assert isinstance(provider, LLMProvider)


@pytest.mark.asyncio
async def test_complete_returns_normalized_response() -> None:
    provider = FakeLLMProvider()
    response = await provider.complete([ChatMessage(role="user", content="hello")])

    assert isinstance(response, LLMResponse)
    assert response.content == "echo: hello"
    assert response.tool_calls == []


def test_chat_message_is_immutable() -> None:
    msg = ChatMessage(role="user", content="hi")
    with pytest.raises(AttributeError):
        msg.content = "changed"  # type: ignore[misc]
