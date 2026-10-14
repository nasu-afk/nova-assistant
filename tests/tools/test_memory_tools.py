from __future__ import annotations

import pytest

from app.core.errors import ValidationError
from app.database.engine import Database
from app.memory.manager import LongTermMemory
from app.tools.memory.tools import ForgetFactTool, ListMemoriesTool, RememberFactTool


@pytest.fixture
async def memory() -> LongTermMemory:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    return LongTermMemory(db)


@pytest.mark.asyncio
async def test_remember_fact_stores_and_confirms(memory: LongTermMemory) -> None:
    tool = RememberFactTool(memory)
    payload = tool.validate_input({"fact": "likes lo-fi music"})

    result = await tool.execute(payload)

    assert result.success is True
    assert "lo-fi" in result.message
    stored = await memory.list_all()
    assert stored[0].value == "likes lo-fi music"


def test_remember_fact_requires_fact_string() -> None:
    with pytest.raises(ValidationError):
        RememberFactTool(memory=None).validate_input({})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_remember_fact_defaults_category_to_general(memory: LongTermMemory) -> None:
    tool = RememberFactTool(memory)
    await tool.execute(tool.validate_input({"fact": "test fact"}))

    stored = await memory.list_all()
    assert stored[0].category == "general"


@pytest.mark.asyncio
async def test_remember_fact_respects_custom_category(memory: LongTermMemory) -> None:
    tool = RememberFactTool(memory)
    await tool.execute(tool.validate_input({"fact": "prefers dark mode", "category": "preference"}))

    stored = await memory.list_all(category="preference")
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_list_memories_returns_stored_facts(memory: LongTermMemory) -> None:
    await memory.remember("k1", "likes jazz")
    await memory.remember("k2", "works as a developer")
    tool = ListMemoriesTool(memory)

    result = await tool.execute(tool.validate_input({}))

    assert result.success is True
    assert len(result.data["memories"]) == 2
    assert "likes jazz" in result.message


@pytest.mark.asyncio
async def test_list_memories_empty_gives_friendly_message(memory: LongTermMemory) -> None:
    tool = ListMemoriesTool(memory)

    result = await tool.execute(tool.validate_input({}))

    assert result.success is True
    assert result.data["memories"] == []
    assert "don't have anything" in result.message


@pytest.mark.asyncio
async def test_list_memories_filters_by_category(memory: LongTermMemory) -> None:
    await memory.remember("k1", "likes jazz", category="preference")
    await memory.remember("k2", "project deadline is Friday", category="project")
    tool = ListMemoriesTool(memory)

    result = await tool.execute(tool.validate_input({"category": "preference"}))

    assert len(result.data["memories"]) == 1
    assert result.data["memories"][0]["fact"] == "likes jazz"


@pytest.mark.asyncio
async def test_forget_fact_removes_it(memory: LongTermMemory) -> None:
    memory_id = await memory.remember("k1", "likes jazz")
    tool = ForgetFactTool(memory)

    result = await tool.execute(tool.validate_input({"memory_id": memory_id}))

    assert result.success is True
    assert await memory.list_all() == []


@pytest.mark.asyncio
async def test_forget_fact_missing_id_reports_failure(memory: LongTermMemory) -> None:
    tool = ForgetFactTool(memory)

    result = await tool.execute(tool.validate_input({"memory_id": "does-not-exist"}))

    assert result.success is False


def test_forget_fact_requires_memory_id() -> None:
    with pytest.raises(ValidationError):
        ForgetFactTool(memory=None).validate_input({})  # type: ignore[arg-type]
