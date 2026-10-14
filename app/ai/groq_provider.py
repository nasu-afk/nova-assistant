"""Groq implementation of `LLMProvider`.

Groq exposes an OpenAI-compatible chat API and offers a genuinely free tier
(no credit card) for open-weight models like Llama 3.3 70B — useful for
running NOVA's voice/agent loop without an Anthropic API key. The official
`groq` SDK is imported lazily for the same reason as `AnthropicProvider`:
this module must import cleanly without the package installed.
"""

from __future__ import annotations

import json
from typing import Any

from app.ai.base import BaseLLMProvider, ChatMessage, LLMResponse
from app.core.errors import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqProvider(BaseLLMProvider):
    """LLMProvider backed by Groq's OpenAI-compatible chat completions API."""

    name = "groq"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise ProviderError(
                "The 'groq' package is not installed (run `uv add groq`).",
                user_message="I can't reach the language model provider right now.",
            ) from exc

        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    @staticmethod
    def _to_openai_content(message: ChatMessage) -> str | list[dict[str, Any]]:
        """Plain text unless `metadata` carries an image (see
        `app.tools.vision.screen.DescribeScreenTool`), in which case build
        OpenAI/Groq's multipart content block shape: text + an image_url
        data URI. Note: this only works if the configured Groq *model*
        itself supports vision input — most of Groq's text models (e.g.
        gpt-oss) do not, so this will surface as a normal API error from
        Groq if the active model can't accept images."""
        image_b64 = message.metadata.get("image_base64")
        if not image_b64:
            return message.content
        media_type = message.metadata.get("media_type", "image/png")
        return [
            {"type": "text", "text": message.content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
            },
        ]

    @classmethod
    def _to_openai_messages(cls, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        # Groq's chat API takes the same {role, content} shape as OpenAI,
        # system messages included inline — no splitting needed (unlike
        # Anthropic's separate top-level `system` field).
        return [{"role": m.role, "content": cls._to_openai_content(m)} for m in messages]

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Translate the canonical {"name","description","parameters"} shape
        (see ToolRegistry.list_tool_specs) into OpenAI/Groq's function-calling
        format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                },
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
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": self._to_openai_messages(messages),
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = self._to_openai_tools(tools)

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize any SDK error
            logger.error("groq_request_failed", error=str(exc))
            if "model_not_found" in str(exc) or "does not exist" in str(exc):
                raise ProviderError(
                    f"Groq model '{self._model}' is not available on this account: {exc}",
                    user_message=(
                        f"The model '{self._model}' isn't available right now. Run "
                        "`uv run python -c \"from groq import Groq; c = Groq(); "
                        "print('\\n'.join(sorted(m.id for m in c.models.list().data)))\"` "
                        "to see which models your key can use, then update NOVA_LLM_MODEL."
                    ),
                ) from exc
            raise ProviderError(
                f"Groq API request failed: {exc}",
                user_message="I couldn't reach the language model right now.",
            ) from exc

        choice = response.choices[0]
        tool_calls: list[dict[str, Any]] = []
        for call in getattr(choice.message, "tool_calls", None) or []:
            # OpenAI/Groq return function arguments as a JSON *string*, not a
            # dict (unlike Anthropic's already-parsed `block.input`) — parse
            # it here so LLMResponse.tool_calls has a consistent shape
            # regardless of provider.
            try:
                arguments = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append({"id": call.id, "name": call.function.name, "input": arguments})

        return LLMResponse(
            content=choice.message.content or "",
            raw=response,
            finish_reason=choice.finish_reason,
            tool_calls=tool_calls,
        )
