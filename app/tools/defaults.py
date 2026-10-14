"""Default tool set + security wiring for running NOVA end to end.

Kept out of `app/main.py` so the choice of "which tools NOVA can use out of
the box" is a single, reviewable place, and so a GUI or test harness can
build the same registry without duplicating this list.

Memory, vision, and reminder tools are optional (passed in, not built
here) because they need collaborators (`LongTermMemory`, a vision-capable
`LLMProvider`, a `TaskScheduler` + delivery callback) that only the caller
knows how to construct — `build_default_registry` stays usable with zero
arguments for tests and simple cases, and gains those tools only when the
caller actually wants them wired up.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.security.manager import ConfirmationManager, SecurityGate
from app.tools.filesystem.files import (
    CopyFileTool,
    CreateDirectoryTool,
    CreateFileTool,
    DeleteFileTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from app.tools.registry import ToolRegistry
from app.tools.web.browser_launch import OpenWebpageTool
from app.tools.web.youtube import PlayYoutubeVideoTool
from app.tools.windows.applications import CloseApplicationTool, OpenApplicationTool
from app.tools.windows.system_info import (
    GetCpuUsageTool,
    GetMemoryUsageTool,
    GetProcessesTool,
    GetSystemInfoTool,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from app.ai.base import LLMProvider
    from app.memory.manager import LongTermMemory
    from app.scheduler.reminders import PersistentReminderManager
    from app.scheduler.scheduler import TaskScheduler
    from app.security.manager import AuditLogger


async def console_confirmation_prompt(tool_name: str, payload: dict[str, Any]) -> bool:
    """Blocking y/n confirmation on stdin, run off the event loop thread.

    This is the default confirmation prompt for the CLI/voice entrypoint
    (`app/main.py`). A GUI build should pass
    `MainWindow.show_confirmation_dialog` to `ConfirmationManager` instead
    (see `app/ui/main_window.py`).
    """

    def ask() -> bool:
        answer = input(f"\nNOVA wants to run '{tool_name}' with {payload}. Allow? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}

    return await asyncio.to_thread(ask)


def build_default_registry(
    confirmation_prompt: Any = console_confirmation_prompt,
    *,
    memory: LongTermMemory | None = None,
    vision_llm: LLMProvider | None = None,
    scheduler: TaskScheduler | None = None,
    on_reminder_fire: Callable[[str], Awaitable[None]] | None = None,
    reminders: PersistentReminderManager | None = None,
    audit: AuditLogger | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry with a sensible default tool set, gated by a
    `SecurityGate` that asks `confirmation_prompt` before any HIGH/CRITICAL
    action (delete_file, close_application, ...).

    Optional additions, each requiring its own collaborator:
    - `memory`: adds remember_fact/list_memories/forget_fact.
    - `vision_llm`: adds describe_screen. Needs a vision-capable model —
      most Groq text models (gpt-oss, etc.) do NOT support image input, so
      check `client.models.list()` for one that does, or use Anthropic.
    - `scheduler` + `on_reminder_fire`: adds set_reminder. Both are required
      together (a scheduler with nowhere to deliver a fired reminder isn't
      useful) — passing only one is a no-op, not an error.
    """
    gate = SecurityGate(
        confirmations=ConfirmationManager(prompt=confirmation_prompt),
        audit=audit,
    )
    registry = ToolRegistry(permission_gate=gate)

    for tool in (
        GetSystemInfoTool(),
        GetProcessesTool(),
        GetCpuUsageTool(),
        GetMemoryUsageTool(),
        OpenApplicationTool(),
        CloseApplicationTool(),
        OpenWebpageTool(),
        PlayYoutubeVideoTool(),
        SearchFilesTool(),
        ReadFileTool(),
        CreateFileTool(),
        WriteFileTool(),
        CreateDirectoryTool(),
        MoveFileTool(),
        CopyFileTool(),
        DeleteFileTool(),
    ):
        registry.register(tool)

    if memory is not None:
        from app.tools.memory.tools import ForgetFactTool, ListMemoriesTool, RememberFactTool

        registry.register(RememberFactTool(memory))
        registry.register(ListMemoriesTool(memory))
        registry.register(ForgetFactTool(memory))

    if vision_llm is not None:
        from app.tools.vision.screen import DescribeScreenTool

        registry.register(DescribeScreenTool(vision_llm))

    if scheduler is not None and on_reminder_fire is not None:
        from app.tools.scheduling.tools import SetReminderTool

        registry.register(SetReminderTool(scheduler, on_reminder_fire, reminders=reminders))

    return registry
