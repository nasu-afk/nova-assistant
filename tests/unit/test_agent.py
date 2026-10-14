from __future__ import annotations

from typing import Any

import pytest

from app.ai.base import ChatMessage, LLMResponse
from app.core.agent import ConversationAgent


class ScriptedLLM:
    """Fake LLMProvider that returns canned replies in order."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.seen_messages: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.seen_messages.append(messages)
        return LLMResponse(content=self._replies.pop(0))


@pytest.mark.asyncio
async def test_respond_returns_llm_reply() -> None:
    llm = ScriptedLLM(["Hi there!"])
    agent = ConversationAgent(llm)

    reply = await agent.respond("hello")

    assert reply == "Hi there!"


@pytest.mark.asyncio
async def test_respond_includes_system_prompt_first() -> None:
    llm = ScriptedLLM(["ok"])
    agent = ConversationAgent(llm, system_prompt="You are NOVA.")

    await agent.respond("hello")

    sent = llm.seen_messages[0]
    assert sent[0].role == "system"
    assert sent[0].content == "You are NOVA."


@pytest.mark.asyncio
async def test_history_accumulates_across_turns() -> None:
    llm = ScriptedLLM(["first reply", "second reply"])
    agent = ConversationAgent(llm)

    await agent.respond("turn one")
    await agent.respond("turn two")

    assert [m.content for m in agent.history] == [
        "turn one",
        "first reply",
        "turn two",
        "second reply",
    ]


@pytest.mark.asyncio
async def test_reset_clears_history() -> None:
    llm = ScriptedLLM(["reply"])
    agent = ConversationAgent(llm)
    await agent.respond("hello")

    agent.reset()

    assert agent.history == []


@pytest.mark.asyncio
async def test_history_is_trimmed_to_max_history() -> None:
    llm = ScriptedLLM([f"reply {i}" for i in range(5)])
    agent = ConversationAgent(llm, max_history=4)

    for i in range(5):
        await agent.respond(f"turn {i}")

    assert len(agent.history) == 4
