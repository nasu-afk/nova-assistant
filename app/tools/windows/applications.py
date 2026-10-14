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
import os
import platform
import re
from typing import Any, ClassVar

import psutil

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult

# A launchable program is a bare name like "notepad" / "code" / "chrome". Paths,
# drive letters, spaces and shell metacharacters are rejected: os.startfile()
# will happily *execute* any file it is handed (installers, .bat, .js, .hta ...),
# and this tool is LOW risk with no confirmation.
_BARE_PROGRAM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_DANGEROUS_EXTENSIONS = frozenset(
    {
        ".bat",
        ".cmd",
        ".com",
        ".js",
        ".jse",
        ".vbs",
        ".vbe",
        ".wsf",
        ".wsh",
        ".hta",
        ".ps1",
        ".msi",
        ".msc",
        ".scr",
        ".lnk",
        ".reg",
        ".jar",
        ".dll",
        ".cpl",
        ".url",
    }
)

# Never terminated by close_application, whatever the model asks for.
_PROTECTED_PROCESSES = frozenset(
    {
        "system",
        "registry",
        "smss",
        "csrss",
        "wininit",
        "winlogon",
        "services",
        "lsass",
        "svchost",
        "explorer",
        "dwm",
        "fontdrvhost",
        "init",
        "systemd",
        "launchd",
    }
)


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
        if not _BARE_PROGRAM.fullmatch(name):
            raise ValidationError(
                f"open_application takes a bare program name, not a path or command line "
                f"({name!r}); pass just the program (e.g. 'chrome')."
            )
        suffix = os.path.splitext(name)[1].lower()
        if suffix in _DANGEROUS_EXTENSIONS:
            raise ValidationError(f"open_application won't launch {suffix} files ({name!r}).")
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
        name = name.strip().lower()
        if name.endswith(".exe"):
            name = name[:-4]
        if len(name) < 3:
            # A 1-2 letter "name" used to substring-match (and kill) half the machine.
            raise ValidationError(f"close_application needs the full program name (got {name!r}).")
        if name in _PROTECTED_PROCESSES:
            raise ValidationError(f"I won't close '{name}': it's a core system process.")
        return {"name": name}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        name = payload["name"]
        killed = []
        # Never terminate NOVA itself or the shell/terminal that launched it.
        protected_pids = {os.getpid()}
        try:
            protected_pids.update(p.pid for p in psutil.Process().parents())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                proc_name = (proc.info["name"] or "").lower()
                if proc_name.endswith(".exe"):
                    proc_name = proc_name[:-4]
                # Exact match (not substring): "note" must not hit "notepad".
                if proc_name != name or proc.info["pid"] in protected_pids:
                    continue
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
