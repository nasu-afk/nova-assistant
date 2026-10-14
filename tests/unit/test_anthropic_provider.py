from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.anthropic_provider import AnthropicProvider
from app.ai.base import ChatMessage
from app.core.errors import ProviderError


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


def _fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


@pytest.fixture
def provider() -> AnthropicProvider:
    # A syntactically valid but fake key — no network call is made because
    # the underlying client method is mocked out below.
    return AnthropicProvider(api_key="sk-ant-fake-key-for-tests", model="claude-sonnet-4-6")


@pytest.mark.asyncio
async def test_complete_returns_text_from_response(provider: AnthropicProvider) -> None:
    provider._client.messages.create = AsyncMock(return_value=_fake_response("Hello, Hasan."))

    response = await provider.complete([ChatMessage(role="user", content="hi")])

    assert response.content == "Hello, Hasan."
    assert response.finish_reason == "end_turn"


@pytest.mark.asyncio
async def test_system_messages_are_split_out(provider: AnthropicProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("ok"))
    provider._client.messages.create = mock_create

    await provider.complete(
        [
            ChatMessage(role="system", content="You are NOVA."),
            ChatMessage(role="user", content="hello"),
        ]
    )

    _, kwargs = mock_create.call_args
    assert kwargs["system"] == "You are NOVA."
    assert kwargs["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_provider_error_raised_on_sdk_failure(provider: AnthropicProvider) -> None:
    provider._client.messages.create = AsyncMock(side_effect=RuntimeError("network down"))

    with pytest.raises(ProviderError):
        await provider.complete([ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_tools_are_translated_to_anthropic_format(provider: AnthropicProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("ok"))
    provider._client.messages.create = mock_create

    canonical_tools = [
        {
            "name": "open_application",
            "description": "Opens an app.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    ]
    messages = [ChatMessage(role="user", content="open notepad")]
    await provider.complete(messages, tools=canonical_tools)

    _, kwargs = mock_create.call_args
    assert kwargs["tools"] == [
        {
            "name": "open_application",
            "description": "Opens an app.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_tool_use_blocks_are_parsed_into_tool_calls(provider: AnthropicProvider) -> None:
    tool_use_block = SimpleNamespace(
        type="tool_use", id="call_1", name="open_application", input={"name": "notepad"}
    )
    response = SimpleNamespace(content=[tool_use_block], stop_reason="tool_use")
    provider._client.messages.create = AsyncMock(return_value=response)

    result = await provider.complete([ChatMessage(role="user", content="open notepad")])

    assert result.tool_calls == [
        {"id": "call_1", "name": "open_application", "input": {"name": "notepad"}}
    ]


def test_missing_dependency_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "anthropic":
            raise ImportError("no module named anthropic")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError):
        AnthropicProvider(api_key="sk-ant-fake")


@pytest.mark.asyncio
async def test_image_message_builds_anthropic_content_blocks(provider: AnthropicProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("I see a code editor."))
    provider._client.messages.create = mock_create

    message = ChatMessage(
        role="user",
        content="What's on the screen?",
        metadata={"image_base64": "ZmFrZWJhc2U2NA==", "media_type": "image/png"},
    )
    await provider.complete([message])

    _, kwargs = mock_create.call_args
    content = kwargs["messages"][0]["content"]
    assert content == [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "ZmFrZWJhc2U2NA==",
            },
        },
        {"type": "text", "text": "What's on the screen?"},
    ]


@pytest.mark.asyncio
async def test_plain_text_message_is_not_wrapped_in_content_blocks(
    provider: AnthropicProvider,
) -> None:
    mock_create = AsyncMock(return_value=_fake_response("hi"))
    provider._client.messages.create = mock_create

    await provider.complete([ChatMessage(role="user", content="hello")])

    _, kwargs = mock_create.call_args
    assert kwargs["messages"][0]["content"] == "hello"
