"""Anthropic implementation of `LLMProvider`.

The Anthropic SDK is imported lazily inside `__init__` so that importing
this module never fails in environments without the package installed or
without an API key configured (e.g. CI, Phase 0-era test runs). Swapping to
a different provider means writing a new class here — nothing in the agent
or planner imports `anthropic` directly.
"""

from __future__ import annotations

from typing import Any

from app.ai.base import BaseLLMProvider, ChatMessage, LLMResponse
from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """LLMProvider backed by the Anthropic Messages API."""

    name = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-6") -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise ProviderError(
                "The 'anthropic' package is not installed.",
                user_message="I can't reach the language model provider right now.",
            ) from exc

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model

    @staticmethod
    def _to_anthropic_content(message: ChatMessage) -> str | list[dict[str, Any]]:
        """Plain text unless `metadata` carries an image (see
        `app.tools.vision.screen.DescribeScreenTool`), in which case build
        Anthropic's multipart content block shape: image block(s) + text."""
        image_b64 = message.metadata.get("image_base64")
        if not image_b64:
            return message.content
        media_type = message.metadata.get("media_type", "image/png")
        return [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": image_b64},
            },
            {"type": "text", "text": message.content},
        ]

    @classmethod
    def _split_system(
        cls, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Anthropic takes `system` as a separate top-level field, not a message."""
        system_parts = [m.content for m in messages if m.role == "system"]
        system = "\n".join(system_parts) if system_parts else None
        turns = [
            {"role": m.role, "content": cls._to_anthropic_content(m)}
            for m in messages
            if m.role != "system"
        ]
        return system, turns

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate the canonical {"name","description","parameters"} shape
        (see ToolRegistry.list_tool_specs) into Anthropic's tool format."""
        return [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
            }
            for t in tools
        ]

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        system, turns = self._split_system(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": turns,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._to_anthropic_tools(tools)

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error
            logger.error("anthropic_request_failed", error=str(exc))
            raise ProviderError(
                f"Anthropic API request failed: {exc}",
                user_message="I couldn't reach the language model right now.",
            ) from exc

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    {"id": block.id, "name": block.name, "input": block.input}
                )

        return LLMResponse(
            content="".join(text_parts),
            raw=response,
            finish_reason=response.stop_reason,
            tool_calls=tool_calls,
        )
