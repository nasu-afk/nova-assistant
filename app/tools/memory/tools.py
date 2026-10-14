"""Tools exposing NOVA's long-term memory (Phase 6) to the LLM.

Kept deliberately simple: facts are free text, not structured key/value
pairs the model has to invent a schema for. `list_memories` returns
everything (optionally filtered by category) rather than doing a keyed
lookup, so the LLM can read through what's stored and decide what's
relevant itself - a keyed exact-match lookup would require the model to
guess the exact key used when the fact was saved, which it can't do
reliably across separate conversations/sessions.
"""

from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.memory.manager import LongTermMemory
from app.tools.base import Tool, ToolMetadata, ToolResult


class RememberFactTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="remember_fact",
        description=(
            "Save a fact, preference, or piece of information about the user for later "
            "(e.g. 'likes lo-fi music', 'works as a software engineer'). Persists across "
            "sessions - call this whenever the user tells you something worth remembering, "
            "even if they don't explicitly say 'remember this'."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember, in plain text"},
                "category": {
                    "type": "string",
                    "description": "Optional grouping, e.g. 'preference', 'project'",
                },
            },
            "required": ["fact"],
        },
    )

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        fact = payload.get("fact")
        if not fact or not isinstance(fact, str):
            raise ValidationError("remember_fact requires a 'fact' string")
        fact = " ".join(fact.split())  # single line: facts end up inside the system prompt
        if len(fact) > 500:
            raise ValidationError("remember_fact: that's too long to store as a single fact.")
        category = payload.get("category") or "general"
        if not isinstance(category, str):
            raise ValidationError("remember_fact 'category' must be a string")
        return {"fact": fact, "category": category}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        fact = payload["fact"].strip()
        existing = {r.value.strip().lower() for r in await self._memory.list_all()}
        if fact.lower() in existing:
            # Saying the same thing twice must not pile up duplicate rows that
            # later crowd the "last 30 facts" prompt window.
            return ToolResult(success=True, message=f"I already knew that: {fact}")
        key = uuid4().hex[:12]
        await self._memory.remember(key, fact, category=payload["category"])
        return ToolResult(success=True, message=f"Got it, I'll remember that: {payload['fact']}")


class ListMemoriesTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="list_memories",
        description=(
            "List everything currently remembered about the user, optionally filtered by category."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional category filter"}
            },
        },
    )

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"category": payload.get("category")}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        records = await self._memory.list_all(category=payload.get("category"))
        facts = [{"id": r.id, "fact": r.value, "category": r.category} for r in records]
        if facts:
            message = "; ".join(f"[{f['id']}] {f['fact']}" for f in facts)
        else:
            message = "I don't have anything remembered yet."
        return ToolResult(success=True, data={"memories": facts}, message=message)


class ForgetFactTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="forget_fact",
        description="Delete a previously remembered fact by its memory id (see list_memories).",
        risk_level=RiskLevel.MEDIUM,
        input_schema={
            "type": "object",
            "properties": {"memory_id": {"type": "string", "description": "The memory's id"}},
            "required": ["memory_id"],
        },
    )

    def __init__(self, memory: LongTermMemory) -> None:
        self._memory = memory

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        memory_id = payload.get("memory_id")
        if not memory_id or not isinstance(memory_id, str):
            raise ValidationError("forget_fact requires a 'memory_id' string")
        return {"memory_id": memory_id}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        removed = await self._memory.forget(payload["memory_id"])
        if not removed:
            memory_id = payload["memory_id"]
            return ToolResult(success=False, error=f"No memory found with id '{memory_id}'.")
        return ToolResult(success=True, message="Forgotten.")
