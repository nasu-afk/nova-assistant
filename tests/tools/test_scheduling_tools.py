from __future__ import annotations

import asyncio

import pytest

from app.core.errors import ValidationError
from app.database.engine import Database
from app.scheduler.reminders import PersistentReminderManager
from app.scheduler.scheduler import TaskScheduler
from app.tools.scheduling.tools import SetReminderTool


@pytest.mark.asyncio
async def test_set_reminder_schedules_and_confirms() -> None:
    scheduler = TaskScheduler()
    fired: list[str] = []

    async def on_fire(message: str) -> None:
        fired.append(message)

    tool = SetReminderTool(scheduler, on_fire)
    payload = tool.validate_input({"message": "take a break", "minutes": 0.001})

    result = await tool.execute(payload)

    assert result.success is True
    assert "take a break" in result.message
    await asyncio.sleep(0.2)
    assert fired == ["take a break"]


def test_requires_message_and_minutes() -> None:
    tool = SetReminderTool(TaskScheduler(), lambda m: asyncio.sleep(0))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        tool.validate_input({"minutes": 5})
    with pytest.raises(ValidationError):
        tool.validate_input({"message": "hi"})


def test_rejects_non_positive_minutes() -> None:
    tool = SetReminderTool(TaskScheduler(), lambda m: asyncio.sleep(0))  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        tool.validate_input({"message": "hi", "minutes": 0})
    with pytest.raises(ValidationError):
        tool.validate_input({"message": "hi", "minutes": -5})


@pytest.mark.asyncio
async def test_singular_minute_wording() -> None:
    scheduler = TaskScheduler()

    async def on_fire(message: str) -> None:
        return None

    tool = SetReminderTool(scheduler, on_fire)
    result = await tool.execute(tool.validate_input({"message": "check oven", "minutes": 1}))

    assert "1 minute:" in result.message
    assert "minutes:" not in result.message


@pytest.mark.asyncio
async def test_persistent_reminder_survives_scheduler_restart() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    first_scheduler = TaskScheduler()

    async def original_callback(message: str) -> None:
        return None

    first_manager = PersistentReminderManager(db, first_scheduler, original_callback)
    await first_manager.schedule("survives restart", delay_seconds=0.01)
    first_scheduler.cancel_all()
    await asyncio.sleep(0.02)

    fired: list[str] = []

    async def restored_callback(message: str) -> None:
        fired.append(message)

    restored_scheduler = TaskScheduler()
    restored_manager = PersistentReminderManager(db, restored_scheduler, restored_callback)
    restored_count = await restored_manager.restore_pending()
    await asyncio.sleep(0.05)

    assert restored_count == 1
    assert fired == ["survives restart"]
    assert await restored_manager.pending() == []
    restored_scheduler.cancel_all()
    await db.dispose()


@pytest.mark.asyncio
async def test_set_reminder_tool_persists_reminder_metadata() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    scheduler = TaskScheduler()

    async def on_fire(message: str) -> None:
        return None

    reminders = PersistentReminderManager(db, scheduler, on_fire)
    tool = SetReminderTool(scheduler, on_fire, reminders=reminders)
    result = await tool.execute(
        tool.validate_input({"message": "persist me", "minutes": 10})
    )

    pending = await reminders.pending()
    assert len(pending) == 1
    assert pending[0].message == "persist me"
    assert result.data["reminder_id"] == pending[0].id
    assert "due_at" in result.data
    scheduler.cancel_all()
    await db.dispose()
