from __future__ import annotations

import pytest

from app.database.engine import Database
from app.memory.conversation import ConversationHistory
from app.memory.manager import MemoryManager, ShortTermMemory


@pytest.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.init()
    yield database
    await database.dispose()


def test_short_term_memory_tracks_recent_turns() -> None:
    stm = ShortTermMemory(max_turns=3)
    stm.add("user", "hi")
    stm.add("assistant", "hello")
    stm.add("user", "how are you")
    stm.add("assistant", "good")

    assert stm.recent() == [
        ("assistant", "hello"),
        ("user", "how are you"),
        ("assistant", "good"),
    ]


def test_short_term_memory_clear() -> None:
    stm = ShortTermMemory()
    stm.add("user", "hi")
    stm.clear()
    assert stm.recent() == []


@pytest.mark.asyncio
async def test_long_term_memory_remember_and_recall(db: Database) -> None:
    manager = MemoryManager(db)

    await manager.long_term.remember("favorite_editor", "VS Code")
    results = await manager.long_term.recall("favorite_editor")

    assert len(results) == 1
    assert results[0].value == "VS Code"


@pytest.mark.asyncio
async def test_long_term_memory_list_all_filters_by_category(db: Database) -> None:
    manager = MemoryManager(db)
    await manager.long_term.remember("editor", "VS Code", category="preference")
    await manager.long_term.remember("goal", "ship PropLens", category="project")

    preferences = await manager.long_term.list_all(category="preference")

    assert len(preferences) == 1
    assert preferences[0].key == "editor"


@pytest.mark.asyncio
async def test_long_term_memory_forget_removes_one(db: Database) -> None:
    manager = MemoryManager(db)
    memory_id = await manager.long_term.remember("k", "v")

    removed = await manager.long_term.forget(memory_id)
    remaining = await manager.long_term.list_all()

    assert removed is True
    assert remaining == []


@pytest.mark.asyncio
async def test_long_term_memory_forget_all(db: Database) -> None:
    manager = MemoryManager(db)
    await manager.long_term.remember("a", "1")
    await manager.long_term.remember("b", "2")

    count = await manager.long_term.forget_all()
    remaining = await manager.long_term.list_all()

    assert count == 2
    assert remaining == []


@pytest.mark.asyncio
async def test_episodic_memory_records_and_lists_tasks(db: Database) -> None:
    manager = MemoryManager(db)
    await manager.episodic.record_task("open VS Code", "completed", summary="Opened VS Code.")

    tasks = await manager.episodic.recent_tasks()

    assert len(tasks) == 1
    assert tasks[0].goal == "open VS Code"
    assert tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_inspect_all_returns_everything(db: Database) -> None:
    manager = MemoryManager(db)
    manager.short_term.add("user", "hello")
    await manager.long_term.remember("fact", "value")
    await manager.episodic.record_task("goal", "completed")

    snapshot = await manager.inspect_all()

    assert snapshot["recent_conversation"] == [("user", "hello")]
    assert len(snapshot["long_term_facts"]) == 1  # type: ignore[arg-type]
    assert len(snapshot["recent_tasks"]) == 1  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_forget_everything_clears_all_memory_types(db: Database) -> None:
    manager = MemoryManager(db)
    manager.short_term.add("user", "hello")
    await manager.long_term.remember("fact", "value")
    await manager.episodic.record_task("goal", "completed")

    await manager.forget_everything()
    snapshot = await manager.inspect_all()

    assert snapshot["recent_conversation"] == []
    assert snapshot["long_term_facts"] == []
    assert snapshot["recent_tasks"] == []


@pytest.mark.asyncio
async def test_conversation_history_persists_and_resumes_latest(db: Database) -> None:
    first = ConversationHistory(db)
    await first.append("user", "Remember this conversation")
    await first.append("assistant", "I will remember it.")

    resumed = ConversationHistory(db)
    messages = await resumed.load_recent()

    assert [(message.role, message.content) for message in messages] == [
        ("user", "Remember this conversation"),
        ("assistant", "I will remember it."),
    ]
    assert resumed.conversation_id == first.conversation_id


@pytest.mark.asyncio
async def test_start_new_conversation_isolates_future_messages(db: Database) -> None:
    history = ConversationHistory(db)
    await history.append("user", "old conversation")

    old_id = history.conversation_id
    new_id = await history.start_new("Fresh chat")
    await history.append("user", "new conversation")

    assert new_id != old_id
    messages = await history.load_recent()
    assert [message.content for message in messages] == ["new conversation"]
