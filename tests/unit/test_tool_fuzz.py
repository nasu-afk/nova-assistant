"""Garbage-in sweep over every tool's input validation.

The model controls tool arguments, so a tool must reject nonsense with a
`NovaError` (which the agent reports naturally) — never crash with a raw
`TypeError` / `ValueError` / `OverflowError`. Only `validate_input`,
`check_permission` and `needs_confirmation` run; nothing executes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.config.settings import Settings
from app.core.errors import NovaError
from app.database.engine import Database
from app.memory.manager import LongTermMemory
from app.scheduler.scheduler import TaskScheduler
from app.tools.base import Tool
from app.tools.browser.tools import (
    BrowserScreenshotTool,
    ClickTool,
    ExtractTextTool,
    NavigateTool,
    OpenUrlTool,
    SearchWebTool,
    TypeTextTool,
)
from app.tools.defaults import build_default_registry
from app.tools.windows.media import SetVolumeTool, TakeScreenshotTool

GARBAGE: list[Any] = [
    None,
    True,
    0,
    -1,
    10**30,
    1.5,
    float("nan"),
    float("inf"),
    "",
    " ",
    "x" * 5000,
    "\x00",
    "a\nb",
    "../../etc/passwd",
    "C:\\Windows\\System32",
    "~",
    "**",
    [],
    ["a"],
    {},
    {"a": 1},
    b"bytes",
]
COMMON_KEYS = [
    "path", "name", "limit", "command", "cwd", "args", "text", "selector", "url", "query",
    "level", "direction", "fact", "message", "minutes", "memory_id", "category", "script",
    "question", "source", "destination", "directory", "pattern", "content", "recursive",
]  # fmt: skip


async def _all_tools(tmp_path: Path) -> tuple[list[Tool], Database]:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()

    class _Vision:
        async def complete(self, *args: Any, **kwargs: Any) -> None: ...

    registry = build_default_registry(
        confirmation_prompt=None,
        settings=Settings(enable_developer_tools=True, _env_file=None),  # type: ignore[call-arg]
        memory=LongTermMemory(db),
        vision_llm=_Vision(),  # type: ignore[arg-type]
        scheduler=TaskScheduler(),
        on_reminder_fire=lambda message: asyncio.sleep(0),
        workspace_root=tmp_path,
    )
    tools = registry.iter_tools()
    for browser_tool in (
        OpenUrlTool, NavigateTool, ClickTool, TypeTextTool, ExtractTextTool,
        BrowserScreenshotTool, SearchWebTool,
    ):  # fmt: skip
        tools.append(browser_tool(None))  # type: ignore[arg-type]
    tools += [SetVolumeTool(), TakeScreenshotTool()]
    return tools, db


@pytest.mark.asyncio
async def test_no_tool_crashes_on_garbage_arguments(tmp_path: Path) -> None:
    tools, db = await _all_tools(tmp_path)
    crashes: dict[tuple[str, str, str], str] = {}

    for tool in tools:
        declared = list((tool.metadata.input_schema or {}).get("properties", {}))
        for key in dict.fromkeys([*declared, *COMMON_KEYS]):
            for garbage in GARBAGE:
                try:
                    validated = tool.validate_input({key: garbage})
                    tool.check_permission(validated)
                    tool.needs_confirmation(validated)
                except NovaError:
                    pass  # a clean, user-explainable rejection is the right outcome
                except Exception as exc:  # noqa: BLE001
                    crashes.setdefault(
                        (tool.metadata.name, type(exc).__name__, str(exc)[:60]),
                        f"{key}={garbage!r:.30}",
                    )

    await db.dispose()
    assert not crashes, "\n".join(f"{k}: e.g. {v}" for k, v in sorted(crashes.items()))
