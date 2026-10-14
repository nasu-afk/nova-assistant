from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.ai.base import ChatMessage
from app.ai.groq_provider import GroqProvider
from app.core.errors import ProviderError


def _fake_response(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def provider() -> GroqProvider:
    return GroqProvider(api_key="gsk_fake_key_for_tests", model="llama-3.3-70b-versatile")


@pytest.mark.asyncio
async def test_complete_returns_text_from_response(provider: GroqProvider) -> None:
    provider._client.chat.completions.create = AsyncMock(
        return_value=_fake_response("Hello, Hasan.")
    )

    response = await provider.complete([ChatMessage(role="user", content="hi")])

    assert response.content == "Hello, Hasan."
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_messages_are_passed_through_openai_style(provider: GroqProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("ok"))
    provider._client.chat.completions.create = mock_create

    await provider.complete(
        [
            ChatMessage(role="system", content="You are NOVA."),
            ChatMessage(role="user", content="hello"),
        ]
    )

    _, kwargs = mock_create.call_args
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are NOVA."},
        {"role": "user", "content": "hello"},
    ]
    assert kwargs["model"] == "llama-3.3-70b-versatile"


@pytest.mark.asyncio
async def test_provider_error_raised_on_sdk_failure(provider: GroqProvider) -> None:
    provider._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("rate limited"))

    with pytest.raises(ProviderError):
        await provider.complete([ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_empty_content_normalizes_to_empty_string(provider: GroqProvider) -> None:
    provider._client.chat.completions.create = AsyncMock(return_value=_fake_response(None))  # type: ignore[arg-type]

    response = await provider.complete([ChatMessage(role="user", content="hi")])

    assert response.content == ""


@pytest.mark.asyncio
async def test_model_not_found_gives_actionable_user_message(provider: GroqProvider) -> None:
    error = RuntimeError(
        "Error code: 404 - {'error': {'message': 'The model `x` does not exist', "
        "'code': 'model_not_found'}}"
    )
    provider._client.chat.completions.create = AsyncMock(side_effect=error)

    with pytest.raises(ProviderError) as exc_info:
        await provider.complete([ChatMessage(role="user", content="hi")])

    assert "models.list" in exc_info.value.user_message
    assert "NOVA_LLM_MODEL" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_tools_are_translated_to_openai_format(provider: GroqProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("ok"))
    provider._client.chat.completions.create = mock_create

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
            "type": "function",
            "function": {
                "name": "open_application",
                "description": "Opens an app.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_tool_calls_parse_json_string_arguments(provider: GroqProvider) -> None:
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="open_application", arguments='{"name": "notepad"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice])
    provider._client.chat.completions.create = AsyncMock(return_value=response)

    result = await provider.complete([ChatMessage(role="user", content="open notepad")])

    assert result.tool_calls == [
        {"id": "call_1", "name": "open_application", "input": {"name": "notepad"}}
    ]


@pytest.mark.asyncio
async def test_tool_calls_with_malformed_json_default_to_empty_dict(
    provider: GroqProvider,
) -> None:
    tool_call = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name="open_application", arguments="not json")
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice])
    provider._client.chat.completions.create = AsyncMock(return_value=response)

    result = await provider.complete([ChatMessage(role="user", content="hi")])

    assert result.tool_calls[0]["input"] == {}


def test_missing_dependency_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "groq":
            raise ImportError("no module named groq")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ProviderError):
        GroqProvider(api_key="gsk_fake")


@pytest.mark.asyncio
async def test_image_message_builds_openai_content_blocks(provider: GroqProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("I see a code editor."))
    provider._client.chat.completions.create = mock_create

    message = ChatMessage(
        role="user",
        content="What's on the screen?",
        metadata={"image_base64": "ZmFrZWJhc2U2NA==", "media_type": "image/png"},
    )
    await provider.complete([message])

    _, kwargs = mock_create.call_args
    content = kwargs["messages"][0]["content"]
    assert content == [
        {"type": "text", "text": "What's on the screen?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,ZmFrZWJhc2U2NA=="},
        },
    ]


@pytest.mark.asyncio
async def test_plain_text_message_is_not_wrapped_in_content_blocks(provider: GroqProvider) -> None:
    mock_create = AsyncMock(return_value=_fake_response("hi"))
    provider._client.chat.completions.create = mock_create

    await provider.complete([ChatMessage(role="user", content="hello")])

    _, kwargs = mock_create.call_args
    assert kwargs["messages"][0]["content"] == "hello"
