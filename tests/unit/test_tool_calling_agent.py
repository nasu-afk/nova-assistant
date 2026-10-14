from __future__ import annotations

from typing import Any, ClassVar

import pytest

from app.ai.base import ChatMessage, LLMResponse
from app.core.agent import ToolCallingAgent
from app.core.enums import RiskLevel
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.registry import ToolRegistry


class OpenApplicationFakeTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="open_application",
        description="Opens an app.",
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, message=f"Opened {payload['name']}.")


class FailingFakeTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="always_fails", description="fails", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="disk full")


class ScriptedToolLLM:
    """Fake LLMProvider returning a scripted sequence of LLMResponses,
    one per `complete()` call."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[ChatMessage], list[dict[str, Any]] | None]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append((messages, tools))
        return self._responses.pop(0)


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(OpenApplicationFakeTool())
    reg.register(FailingFakeTool())
    return reg


@pytest.mark.asyncio
async def test_no_tool_call_returns_direct_reply(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM([LLMResponse(content="Hi there!")])
    agent = ToolCallingAgent(llm, registry)

    reply = await agent.respond("hello")

    assert reply == "Hi there!"
    assert llm.calls[0][1] is not None  # tools were offered even for plain chat


@pytest.mark.asyncio
async def test_single_tool_call_executes_and_summarizes(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {"id": "1", "name": "open_application", "input": {"name": "notepad"}}
                ],
            ),
            LLMResponse(content="I opened Notepad for you."),
        ]
    )
    agent = ToolCallingAgent(llm, registry)

    reply = await agent.respond("open notepad")

    assert reply == "I opened Notepad for you."
    # The tool result should have been folded into history before the final call.
    assert any("Opened notepad" in m.content for m in agent.history if m.role == "assistant")


@pytest.mark.asyncio
async def test_tool_call_with_unregistered_tool_reports_failure(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM(
        [
            LLMResponse(
                content="", tool_calls=[{"id": "1", "name": "does_not_exist", "input": {}}]
            ),
            LLMResponse(content="Sorry, that didn't work."),
        ]
    )
    agent = ToolCallingAgent(llm, registry)

    reply = await agent.respond("do the thing")

    assert reply == "Sorry, that didn't work."
    assert any("failed" in m.content for m in agent.history if m.role == "assistant")


@pytest.mark.asyncio
async def test_failing_tool_reports_error_in_summary(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM(
        [
            LLMResponse(
                content="", tool_calls=[{"id": "1", "name": "always_fails", "input": {}}]
            ),
            LLMResponse(content="That failed because the disk is full."),
        ]
    )
    agent = ToolCallingAgent(llm, registry)

    reply = await agent.respond("do the risky thing")

    assert "disk is full" in reply


@pytest.mark.asyncio
async def test_max_tool_rounds_bounds_repeated_tool_calls(registry: ToolRegistry) -> None:
    # LLM keeps requesting tools forever; max_tool_rounds=1 must stop it.
    call = {"id": "1", "name": "open_application", "input": {"name": "notepad"}}
    llm = ScriptedToolLLM(
        [
            LLMResponse(content="", tool_calls=[call]),
            LLMResponse(content="", tool_calls=[call]),  # would loop forever without the cap
            LLMResponse(content="Done, eventually."),  # fallback summarize call
        ]
    )
    agent = ToolCallingAgent(llm, registry, max_tool_rounds=1)

    reply = await agent.respond("open notepad repeatedly")

    assert reply == "Done, eventually."
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_no_tools_registered_still_works_as_plain_chat() -> None:
    empty_registry = ToolRegistry()
    llm = ScriptedToolLLM([LLMResponse(content="Just chatting.")])
    agent = ToolCallingAgent(llm, empty_registry)

    reply = await agent.respond("hi")

    assert reply == "Just chatting."
    assert llm.calls[0][1] is None  # no tools to offer


def test_reset_clears_history(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM([])
    agent = ToolCallingAgent(llm, registry)
    agent._history.append(ChatMessage(role="user", content="hi"))

    agent.reset()

    assert agent.history == []


# --- memory injection ---


@pytest.mark.asyncio
async def test_memory_facts_are_injected_into_system_prompt(registry: ToolRegistry) -> None:
    from app.database.engine import Database

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    from app.memory.manager import LongTermMemory

    long_term = LongTermMemory(db)
    await long_term.remember("music_pref", "likes lo-fi music")

    llm = ScriptedToolLLM([LLMResponse(content="Sure, I'll play something chill.")])
    agent = ToolCallingAgent(llm, registry, memory=long_term)

    await agent.respond("play me something")

    system_message = llm.calls[0][0][0]
    assert system_message.role == "system"
    assert "likes lo-fi music" in system_message.content


@pytest.mark.asyncio
async def test_no_memory_manager_means_plain_system_prompt(registry: ToolRegistry) -> None:
    llm = ScriptedToolLLM([LLMResponse(content="ok")])
    agent = ToolCallingAgent(llm, registry)  # no memory passed

    await agent.respond("hi")

    system_message = llm.calls[0][0][0]
    assert "Known facts about the user" not in system_message.content


@pytest.mark.asyncio
async def test_empty_memory_means_plain_system_prompt(registry: ToolRegistry) -> None:
    from app.database.engine import Database
    from app.memory.manager import LongTermMemory

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    long_term = LongTermMemory(db)

    llm = ScriptedToolLLM([LLMResponse(content="ok")])
    agent = ToolCallingAgent(llm, registry, memory=long_term)

    await agent.respond("hi")

    system_message = llm.calls[0][0][0]
    assert "Known facts about the user" not in system_message.content


@pytest.mark.asyncio
async def test_memory_fact_saved_mid_conversation_is_visible_next_turn(
    registry: ToolRegistry,
) -> None:
    from app.database.engine import Database
    from app.memory.manager import LongTermMemory

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    long_term = LongTermMemory(db)

    llm = ScriptedToolLLM([LLMResponse(content="Got it."), LLMResponse(content="Sure thing.")])
    agent = ToolCallingAgent(llm, registry, memory=long_term)

    await agent.respond("I like jazz")
    # Simulate the remember_fact tool having run and written to memory.
    await long_term.remember("music_pref", "likes jazz")
    await agent.respond("play me something")

    second_call_system_message = llm.calls[1][0][0]
    assert "likes jazz" in second_call_system_message.content


@pytest.mark.asyncio
async def test_agent_resumes_persisted_conversation(registry: ToolRegistry) -> None:
    from app.database.engine import Database
    from app.memory.conversation import ConversationHistory

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    history = ConversationHistory(db)

    first_llm = ScriptedToolLLM([LLMResponse(content="Persistent answer")])
    first_agent = ToolCallingAgent(first_llm, registry, conversation_history=history)
    await first_agent.respond("Persistent question")

    resumed_llm = ScriptedToolLLM([LLMResponse(content="Welcome back")])
    resumed_agent = ToolCallingAgent(
        resumed_llm,
        registry,
        conversation_history=ConversationHistory(db),
    )
    await resumed_agent.respond("What did we discuss?")

    sent_messages = resumed_llm.calls[0][0]
    assert any(message.content == "Persistent question" for message in sent_messages)
    assert any(message.content == "Persistent answer" for message in sent_messages)
    await db.dispose()


# --- event publishing ---


@pytest.mark.asyncio
async def test_response_ready_is_published_with_final_text(registry: ToolRegistry) -> None:
    from app.core.enums import EventType
    from app.events.bus import Event, EventBus

    events = EventBus()
    captured: list[Event] = []

    async def record(event: Event) -> None:
        captured.append(event)

    events.subscribe(EventType.RESPONSE_READY, record)

    llm = ScriptedToolLLM([LLMResponse(content="Hi there!")])
    agent = ToolCallingAgent(llm, registry, events=events)

    await agent.respond("hello")

    assert len(captured) == 1
    assert captured[0].payload["text"] == "Hi there!"


@pytest.mark.asyncio
async def test_tool_lifecycle_events_are_published(registry: ToolRegistry) -> None:
    from app.core.enums import EventType
    from app.events.bus import Event, EventBus

    events = EventBus()
    captured: list[Event] = []

    async def record(event: Event) -> None:
        captured.append(event)

    events.subscribe(EventType.TOOL_STARTED, record)
    events.subscribe(EventType.TOOL_COMPLETED, record)

    llm = ScriptedToolLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {"id": "1", "name": "open_application", "input": {"name": "notepad"}}
                ],
            ),
            LLMResponse(content="Opened it."),
        ]
    )
    agent = ToolCallingAgent(llm, registry, events=events)

    await agent.respond("open notepad")

    assert [e.type for e in captured] == [EventType.TOOL_STARTED, EventType.TOOL_COMPLETED]
    assert all(e.payload["tool"] == "open_application" for e in captured)


@pytest.mark.asyncio
async def test_tool_failed_event_is_published(registry: ToolRegistry) -> None:
    from app.core.enums import EventType
    from app.events.bus import Event, EventBus

    events = EventBus()
    captured: list[Event] = []

    async def record(event: Event) -> None:
        captured.append(event)

    events.subscribe(EventType.TOOL_FAILED, record)

    llm = ScriptedToolLLM(
        [
            LLMResponse(content="", tool_calls=[{"id": "1", "name": "always_fails", "input": {}}]),
            LLMResponse(content="That failed."),
        ]
    )
    agent = ToolCallingAgent(llm, registry, events=events)

    await agent.respond("do the thing")

    assert len(captured) == 1
    assert captured[0].payload["tool"] == "always_fails"


@pytest.mark.asyncio
async def test_no_events_bus_means_no_publishing_attempted(registry: ToolRegistry) -> None:
    # Just confirms passing no EventBus doesn't raise anywhere.
    llm = ScriptedToolLLM([LLMResponse(content="ok")])
    agent = ToolCallingAgent(llm, registry)  # events=None (default)

    reply = await agent.respond("hi")

    assert reply == "ok"
