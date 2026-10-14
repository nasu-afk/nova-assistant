from __future__ import annotations

import pytest

from app.core.errors import ConfirmationRequiredError
from app.tools.defaults import build_default_registry


def test_build_default_registry_registers_expected_tools() -> None:
    registry = build_default_registry(confirmation_prompt=None)
    names = {m.name for m in registry.list_metadata()}

    assert {
        "get_system_info",
        "get_cpu_usage",
        "get_memory_usage",
        "get_processes",
        "open_application",
        "close_application",
        "open_webpage",
        "play_youtube_video",
        "search_files",
        "read_file",
        "create_file",
        "write_file",
        "create_directory",
        "move_file",
        "copy_file",
        "delete_file",
    }.issubset(names)


@pytest.mark.asyncio
async def test_low_risk_tool_runs_without_confirmation() -> None:
    registry = build_default_registry(confirmation_prompt=None)

    result = await registry.invoke("get_cpu_usage", {})

    assert result.success is True


@pytest.mark.asyncio
async def test_high_risk_tool_without_prompt_requires_confirmation_error() -> None:
    registry = build_default_registry(confirmation_prompt=None)

    with pytest.raises(ConfirmationRequiredError):
        await registry.invoke("delete_file", {"path": "/tmp/does-not-matter"})


@pytest.mark.asyncio
async def test_high_risk_tool_with_approved_prompt_proceeds(tmp_path) -> None:  # type: ignore[no-untyped-def]
    async def always_yes(tool_name: str, payload: dict) -> bool:  # type: ignore[type-arg]
        return True

    target = tmp_path / "to_delete.txt"
    target.write_text("bye")
    registry = build_default_registry(confirmation_prompt=always_yes)

    result = await registry.invoke("delete_file", {"path": str(target)})

    assert result.success is True
    assert not target.exists()


@pytest.mark.asyncio
async def test_high_risk_tool_with_declined_prompt_does_not_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from app.core.errors import PermissionDeniedError

    async def always_no(tool_name: str, payload: dict) -> bool:  # type: ignore[type-arg]
        return False

    target = tmp_path / "keep_me.txt"
    target.write_text("still here")
    registry = build_default_registry(confirmation_prompt=always_no)

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("delete_file", {"path": str(target)})

    assert target.exists()


@pytest.mark.asyncio
async def test_memory_tools_not_registered_by_default() -> None:
    registry = build_default_registry(confirmation_prompt=None)
    names = {m.name for m in registry.list_metadata()}
    assert "remember_fact" not in names


@pytest.mark.asyncio
async def test_memory_tools_registered_when_memory_provided() -> None:
    from app.database.engine import Database
    from app.memory.manager import LongTermMemory

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    long_term = LongTermMemory(db)

    registry = build_default_registry(confirmation_prompt=None, memory=long_term)
    names = {m.name for m in registry.list_metadata()}

    assert {"remember_fact", "list_memories", "forget_fact"}.issubset(names)


@pytest.mark.asyncio
async def test_memory_tool_actually_persists_through_registry() -> None:
    from app.database.engine import Database
    from app.memory.manager import LongTermMemory

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    long_term = LongTermMemory(db)
    registry = build_default_registry(confirmation_prompt=None, memory=long_term)

    await registry.invoke("remember_fact", {"fact": "likes lo-fi music"})
    result = await registry.invoke("list_memories", {})

    assert "likes lo-fi music" in result.message


def test_vision_tool_not_registered_by_default() -> None:
    registry = build_default_registry(confirmation_prompt=None)
    names = {m.name for m in registry.list_metadata()}
    assert "describe_screen" not in names


def test_vision_tool_registered_when_vision_llm_provided() -> None:
    class FakeLLM:
        async def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.7):  # type: ignore[no-untyped-def]
            raise NotImplementedError

    registry = build_default_registry(confirmation_prompt=None, vision_llm=FakeLLM())  # type: ignore[arg-type]
    names = {m.name for m in registry.list_metadata()}

    assert "describe_screen" in names


def test_reminder_tool_not_registered_without_both_scheduler_and_callback() -> None:
    from app.scheduler.scheduler import TaskScheduler

    registry_no_callback = build_default_registry(
        confirmation_prompt=None, scheduler=TaskScheduler()
    )
    assert "set_reminder" not in {m.name for m in registry_no_callback.list_metadata()}

    async def on_fire(message: str) -> None:
        return None

    registry_no_scheduler = build_default_registry(
        confirmation_prompt=None, on_reminder_fire=on_fire
    )
    assert "set_reminder" not in {m.name for m in registry_no_scheduler.list_metadata()}


@pytest.mark.asyncio
async def test_reminder_tool_registered_when_both_provided() -> None:
    from app.scheduler.scheduler import TaskScheduler

    async def on_fire(message: str) -> None:
        return None

    registry = build_default_registry(
        confirmation_prompt=None, scheduler=TaskScheduler(), on_reminder_fire=on_fire
    )
    names = {m.name for m in registry.list_metadata()}

    assert "set_reminder" in names
