"""Open/close application tools.

`open_application` shells out to the OS's native "run this program" facility
rather than accepting an arbitrary command line from the model — the model
supplies a program name/path, not a full shell command (that's `run_command`
in app/tools/developer, which has its own, stricter policy).

`close_application` is HIGH risk: it terminates a running process by name,
which is destructive and could kill unsaved work, so it requires
confirmation via the security layer.
"""

from __future__ import annotations

import asyncio
import platform
from typing import Any, ClassVar

import psutil

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult


class OpenApplicationTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="open_application",
        description=(
            "Launch an installed application by its bare program name or path only "
            "(e.g. 'notepad', 'code'). Never pass a URL, arguments, or a full command "
            "line here — for opening a website use open_webpage instead."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Program name or path, e.g. 'notepad'"}
            },
            "required": ["name"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise ValidationError("open_application requires a 'name' string")
        name = name.strip()
        if "://" in name or name.lower().startswith(("http:", "https:")):
            raise ValidationError(
                f"open_application got a URL ({name!r}); use open_webpage for URLs instead."
            )
        if " " in name:
            raise ValidationError(
                f"open_application takes a bare program name, not a command line "
                f"({name!r}); pass just the program (e.g. 'chrome')."
            )
        return {"name": name}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        name = payload["name"]
        try:
            if platform.system() == "Windows":
                import os

                os.startfile(name)
            else:
                # Development/CI fallback so this tool is exercisable off Windows.
                await asyncio.create_subprocess_exec(
                    name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
        except (FileNotFoundError, OSError) as exc:
            return ToolResult(success=False, error=f"Couldn't find or launch '{name}': {exc}")

        return ToolResult(success=True, message=f"Opened {name}.", data={"name": name})


class CloseApplicationTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="close_application",
        description="Terminate all running processes matching a given name.",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Process name to match, e.g. 'notepad'"}
            },
            "required": ["name"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = payload.get("name")
        if not name or not isinstance(name, str):
            raise ValidationError("close_application requires a 'name' string")
        return {"name": name.strip().lower()}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        name = payload["name"]
        killed = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if name in (proc.info["name"] or "").lower():
                    proc.terminate()
                    killed.append(proc.info["pid"])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not killed:
            return ToolResult(success=False, error=f"No running process matched '{name}'.")

        return ToolResult(
            success=True,
            data={"terminated_pids": killed},
            message=f"Closed {len(killed)} process(es) matching '{name}'.",
        )
