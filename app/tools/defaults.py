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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.enums import RiskLevel
from app.security.manager import (
    CommandPolicy,
    ConfirmationManager,
    SecurityGate,
    summarize_payload,
)
from app.tools.filesystem.files import (
    CopyFileTool,
    CreateDirectoryTool,
    CreateFileTool,
    DeleteFileTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
    _PathConfinedTool,
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
    from app.config.settings import Settings
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
        answer = input(
            f"\nNOVA wants to run '{tool_name}' with {summarize_payload(payload)}. Allow? [y/N] "
        )
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
    settings: Settings | None = None,
    workspace_root: Path | None = None,
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
    - `settings`: wires the security settings that used to be ignored —
      `require_confirmation_for_high_risk/critical_risk`, the command
      allow/deny lists + timeout, and `enable_developer_tools`.
    - `workspace_root`: the folder file tools are confined to. Defaults to
      `settings.workspace_root`, then the user's home directory.
    """
    require_for: set[RiskLevel] = {RiskLevel.HIGH, RiskLevel.CRITICAL}
    if settings is not None:
        if not settings.require_confirmation_for_high_risk:
            require_for.discard(RiskLevel.HIGH)
        if not settings.require_confirmation_for_critical_risk:
            require_for.discard(RiskLevel.CRITICAL)
        if workspace_root is None and settings.workspace_root.strip():
            workspace_root = Path(settings.workspace_root).expanduser()
    root = (workspace_root or Path.home()).expanduser().resolve()

    gate = SecurityGate(
        confirmations=ConfirmationManager(prompt=confirmation_prompt, require_for=require_for),
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
        if isinstance(tool, _PathConfinedTool):
            tool.workspace_root = root
        registry.register(tool)

    if settings is not None and settings.enable_developer_tools:
        from app.tools.developer.tools import (
            DockerStatusTool,
            GitLogTool,
            GitStatusTool,
            RunCommandTool,
            RunGradleTool,
            RunMavenTool,
            RunNpmTool,
            RunPythonTool,
        )

        policy = CommandPolicy(
            allowlist=settings.command_allowlist,
            denylist=settings.command_denylist,
            timeout_seconds=settings.command_timeout_seconds,
        )
        for dev_tool in (
            RunCommandTool(policy),
            GitStatusTool(),
            GitLogTool(),
            RunPythonTool(),
            RunMavenTool(),
            RunGradleTool(),
            RunNpmTool(),
            DockerStatusTool(),
        ):
            registry.register(dev_tool)

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
