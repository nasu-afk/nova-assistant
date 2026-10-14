"""LLM provider abstraction.

The agent (app/core/agent.py, added in a later phase) must depend only on
this interface, never on a concrete SDK. This keeps the LLM provider
swappable per the project's engineering principles (Anthropic today,
OpenAI/local later, without touching agent logic).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One turn in a conversation sent to/from an LLM provider."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response from any LLM provider."""

    content: str
    raw: Any = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol every concrete LLM provider (Anthropic, OpenAI, local) implements."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send `messages` to the model and return a normalized response."""
        ...


class BaseLLMProvider(ABC):
    """Optional convenience base class for LLM providers.

    Concrete providers may implement `LLMProvider` directly (structural
    typing), or subclass this for shared helpers as they accumulate.
    """

    name: str

    @abstractmethod
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse: ...
