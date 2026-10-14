"""The NOVA agent.

Phase 1 gives the agent just enough to hold a conversation: it keeps a
rolling message history and asks the configured `LLMProvider` for a reply.
`ToolCallingAgent` extends that with tool calling: the LLM can request a
tool via the shared `ToolRegistry`, NOVA executes it, and the LLM turns the
result into a natural-language reply. Planning/re-planning for multi-step
tasks (Phase 5) is a separate, heavier-weight path (`app.core.planner`,
`app.core.executor`) for when a request genuinely needs an ordered sequence
of steps rather than "call one tool, describe the result."
"""

from __future__ import annotations

from typing import Protocol

from app.ai.base import ChatMessage, LLMProvider
from app.core.enums import EventType
from app.core.errors import NovaError
from app.core.logging import get_logger
from app.events.bus import Event, EventBus
from app.memory.conversation import ConversationHistory
from app.memory.manager import LongTermMemory
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are NOVA, a helpful, concise voice-first desktop assistant. "
    "Speak naturally, the way a helpful person would — never say things "
    "like 'Command execution has been successfully completed.' Keep "
    "responses short unless the user asks for detail."
)

# Extra instruction appended when tools are available, so the model knows
# to actually call them rather than describing what it *would* do.
TOOL_USE_SUFFIX = (
    " You have tools available for interacting with the user's computer. "
    "Call a tool whenever the request needs one (opening an app, reading a "
    "file, checking system stats, etc.) instead of just describing what you "
    "would do."
)


class RespondingAgent(Protocol):
    """What `VoicePipeline` needs from an agent — satisfied by both
    `ConversationAgent` and `ToolCallingAgent`."""

    async def respond(self, user_text: str) -> str: ...


class ConversationAgent:
    """Holds conversation state and produces replies via an `LLMProvider`."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history: int = 20,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._history: list[ChatMessage] = []

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    def reset(self) -> None:
        """Clear conversation history (e.g. on 'new conversation')."""
        self._history.clear()

    async def respond(self, user_text: str) -> str:
        """Add `user_text` to history, get a reply, and record it too."""
        self._history.append(ChatMessage(role="user", content=user_text))
        self._trim_history()

        messages = [ChatMessage(role="system", content=self._system_prompt), *self._history]
        response = await self._llm.complete(messages)

        self._history.append(ChatMessage(role="assistant", content=response.content))
        self._trim_history()
        return response.content

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]


class ToolCallingAgent:
    """A `ConversationAgent` that can call tools through a `ToolRegistry`.

    Deliberately a small, bounded number of tool-calling rounds per turn
    (`max_tool_rounds`, default 2), not a full multi-step ReAct loop: the
    LLM may request one or more tools, NOVA executes them, and if a tool
    call fails validation (wrong arguments, wrong tool for the job), the
    model gets one more round to try again with the corrected call before
    NOVA just summarizes whatever happened. That covers "open notepad" /
    "what's my CPU usage" voice-command style requests — including the
    occasional self-correction — well; a request that genuinely needs an
    ordered multi-step plan should go through `app.core.planner`/`executor`
    instead.
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT + TOOL_USE_SUFFIX,
        max_history: int = 20,
        max_tool_rounds: int = 2,
        memory: LongTermMemory | None = None,
        conversation_history: ConversationHistory | None = None,
        events: EventBus | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._max_tool_rounds = max_tool_rounds
        self._memory = memory
        self._conversation_history = conversation_history
        self._events = events
        self._history: list[ChatMessage] = []
        self._history_loaded = conversation_history is None

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
        self._history_loaded = True

    async def start_new_conversation(self, title: str = "") -> str | None:
        """Start a new durable conversation and clear the active context."""
        self._history.clear()
        self._history_loaded = True
        if self._conversation_history is None:
            return None
        return await self._conversation_history.start_new(title)

    async def _load_persisted_history(self) -> None:
        if self._history_loaded or self._conversation_history is None:
            return
        self._history = await self._conversation_history.load_recent(self._max_history)
        self._history_loaded = True

    async def _persist_message(self, role: str, content: str) -> None:
        if self._conversation_history is None:
            return
        try:
            await self._conversation_history.append(role, content)
        except Exception as exc:  # noqa: BLE001 - persistence must not lose the live reply
            logger.error("conversation_persistence_failed", role=role, error=str(exc))

    async def _current_system_prompt(self) -> str:
        """The base system prompt, plus a fresh snapshot of remembered facts
        every call — so a fact saved mid-conversation is visible on the very
        next turn, not just in future sessions."""
        if self._memory is None:
            return self._system_prompt
        facts = await self._memory.list_all()
        if not facts:
            return self._system_prompt
        # Cap how many facts get inlined so this can't grow the prompt
        # unboundedly as memory accumulates over many sessions.
        recent = facts[-30:]
        facts_block = "\n".join(f"- {f.value}" for f in recent)
        return f"{self._system_prompt}\n\nKnown facts about the user (from memory):\n{facts_block}"

    async def respond(self, user_text: str) -> str:
        await self._load_persisted_history()
        self._history.append(ChatMessage(role="user", content=user_text))
        self._trim_history()
        await self._persist_message("user", user_text)

        tool_specs = self._registry.list_tool_specs()
        rounds = 0

        while True:
            system_prompt = await self._current_system_prompt()
            messages = [
                ChatMessage(role="system", content=system_prompt),
                *self._history,
            ]
            response = await self._llm.complete(messages, tools=tool_specs or None)

            if not response.tool_calls or rounds >= self._max_tool_rounds:
                final_text = response.content or await self._summarize_without_tools()
                self._history.append(ChatMessage(role="assistant", content=final_text))
                self._trim_history()
                await self._persist_message("assistant", final_text)
                await self._publish(EventType.RESPONSE_READY, text=final_text)
                return final_text

            rounds += 1
            summary = await self._execute_tool_calls(response.tool_calls)
            self._history.append(ChatMessage(role="assistant", content=f"[Tool results] {summary}"))
            self._trim_history()
            # Loop back so the LLM can turn the tool results into a reply —
            # tools are still offered in case it needs a second call, but
            # `max_tool_rounds` bounds how many times that can happen.

    async def _publish(self, event_type: EventType, **payload: object) -> None:
        if self._events is not None:
            await self._events.publish(Event(type=event_type, payload=payload))

    async def _execute_tool_calls(self, tool_calls: list[dict[str, object]]) -> str:
        lines = []
        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("input") or {}
            if not isinstance(args, dict):
                args = {}
            await self._publish(EventType.TOOL_STARTED, tool=name)
            try:
                result = await self._registry.invoke(name, args)
                outcome = result.message if result.success else f"failed: {result.error}"
                if result.success:
                    await self._publish(EventType.TOOL_COMPLETED, tool=name)
                else:
                    await self._publish(EventType.TOOL_FAILED, tool=name, error=result.error)
            except NovaError as exc:
                outcome = f"failed: {exc.user_message}"
                await self._publish(EventType.TOOL_FAILED, tool=name, error=exc.user_message)
            logger.info("agent_tool_call", tool=name, args=args, outcome=outcome)
            lines.append(f"{name}({args}) -> {outcome}")
        return "; ".join(lines)

    async def _summarize_without_tools(self) -> str:
        """Force one final text-only reply if the last LLM turn produced tool
        calls but no content (common right after a tool round)."""
        system_prompt = await self._current_system_prompt()
        messages = [
            ChatMessage(role="system", content=system_prompt),
            *self._history,
            ChatMessage(
                role="user",
                content="Briefly tell me what happened in one natural sentence.",
            ),
        ]
        response = await self._llm.complete(messages)
        return response.content

    def _trim_history(self) -> None:
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
