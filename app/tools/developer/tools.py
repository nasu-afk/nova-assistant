"""Developer tools.

`RunCommandTool` is the only tool that runs an arbitrary command line — and
even then, never as a raw shell string. It's split into a program + argument
list, checked against a `CommandPolicy` (allowlist/denylist), executed with
a timeout, and every invocation is captured for the audit log. The other
tools here (git/maven/gradle/npm/docker) are thin, safer wrappers that only
ever invoke one known binary with a fixed, safe subcommand shape.
"""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from typing import Any, ClassVar

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.security.manager import CommandPolicy
from app.tools.base import Tool, ToolMetadata, ToolResult


async def _run(
    program: str, args: list[str], cwd: str | None, timeout_seconds: int
) -> tuple[int, str, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            program,
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        return 127, "", f"'{program}' is not installed or not on PATH: {exc}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"Command timed out after {timeout_seconds}s."

    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


class RunCommandTool(Tool):
    """Runs an allow/deny-listed command. HIGH risk — always confirmed."""

    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="run_command",
        description="Run a shell command (program + args), subject to the command policy.",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
    )

    def __init__(self, policy: CommandPolicy | None = None) -> None:
        self._policy = policy or CommandPolicy()

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = payload.get("command")
        if not command or not isinstance(command, str):
            raise ValidationError("run_command requires a 'command' string")
        cwd = payload.get("cwd")
        if cwd is not None and not isinstance(cwd, str):
            raise ValidationError("run_command 'cwd' must be a string if provided")
        return {"command": command, "cwd": cwd}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        command = payload["command"]
        allowed, reason = self._policy.is_allowed(command)
        if not allowed:
            return ToolResult(success=False, error=f"Blocked by command policy: {reason}")

        parts = shlex.split(command)
        if not parts:
            return ToolResult(success=False, error="Empty command.")

        cwd = payload.get("cwd")
        if cwd and not await asyncio.to_thread(lambda: Path(cwd).is_dir()):
            return ToolResult(success=False, error=f"'{cwd}' is not a valid working directory.")

        exit_code, stdout, stderr = await _run(
            parts[0], parts[1:], cwd, self._policy.timeout_seconds
        )
        return ToolResult(
            success=exit_code == 0,
            data={"exit_code": exit_code, "stdout": stdout, "stderr": stderr},
            message=stdout.strip()[:500] if exit_code == 0 else "",
            error=None if exit_code == 0 else (stderr.strip()[:500] or f"Exit code {exit_code}"),
        )


class _RepoTool(Tool):
    """Shared validation for tools that operate on a repo/project directory."""

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        cwd = payload.get("cwd", ".")
        if not isinstance(cwd, str):
            raise ValidationError("'cwd' must be a string")
        return {"cwd": cwd}


class GitStatusTool(_RepoTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="git_status",
        description="Show the working tree status of a git repository.",
        risk_level=RiskLevel.LOW,
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        code, out, err = await _run("git", ["status", "--short", "--branch"], payload["cwd"], 15)
        if code != 0:
            return ToolResult(success=False, error=err.strip() or "git status failed.")
        message = out.strip() or "Working tree clean."
        return ToolResult(success=True, data={"status": out}, message=message)


class GitLogTool(_RepoTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="git_log",
        description="Show recent commit history of a git repository.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = super().validate_input(payload)
        base["limit"] = int(payload.get("limit", 10))
        return base

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        args = ["log", f"-{payload['limit']}", "--oneline"]
        code, out, err = await _run("git", args, payload["cwd"], 15)
        if code != 0:
            return ToolResult(success=False, error=err.strip() or "git log failed.")
        lines = [line for line in out.splitlines() if line.strip()]
        message = f"{len(lines)} recent commit(s)."
        return ToolResult(success=True, data={"commits": lines}, message=message)


class RunPythonTool(_RepoTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="run_python",
        description="Run a Python script and capture its output.",
        risk_level=RiskLevel.MEDIUM,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        base = super().validate_input(payload)
        script = payload.get("script")
        if not script or not isinstance(script, str):
            raise ValidationError("run_python requires a 'script' path")
        base["script"] = script
        base["args"] = [str(a) for a in payload.get("args", [])]
        return base

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        code, out, err = await _run(
            sys.executable, [payload["script"], *payload["args"]], payload["cwd"], 60
        )
        return ToolResult(
            success=code == 0,
            data={"exit_code": code, "stdout": out, "stderr": err},
            message=out.strip()[:500],
            error=None if code == 0 else err.strip()[:500],
        )


def _build_manager_tool(
    name: str, description: str, binary: str, default_args: list[str]
) -> type[_RepoTool]:
    """Factory for the near-identical maven/gradle/npm wrappers."""

    class _ManagerTool(_RepoTool):
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name=name, description=description, risk_level=RiskLevel.MEDIUM
        )

        def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
            base = super().validate_input(payload)
            base["args"] = [str(a) for a in payload.get("args", default_args)]
            return base

        async def execute(self, payload: dict[str, Any]) -> ToolResult:
            code, out, err = await _run(binary, payload["args"], payload["cwd"], 300)
            return ToolResult(
                success=code == 0,
                data={"exit_code": code, "stdout": out[-4000:], "stderr": err[-2000:]},
                message="Build succeeded." if code == 0 else "",
                error=None if code == 0 else (err.strip()[-500:] or f"{binary} exited {code}"),
            )

    _ManagerTool.__name__ = f"{name.title().replace('_', '')}Tool"
    return _ManagerTool


RunMavenTool = _build_manager_tool(
    "run_maven", "Run a Maven build (default: mvn package).", "mvn", ["package"]
)
RunGradleTool = _build_manager_tool(
    "run_gradle", "Run a Gradle build (default: gradle build).", "gradle", ["build"]
)
RunNpmTool = _build_manager_tool(
    "run_npm", "Run an npm script (default: npm install).", "npm", ["install"]
)


class DockerStatusTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="docker_status",
        description="List running Docker containers.",
        risk_level=RiskLevel.LOW,
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        code, out, err = await _run(
            "docker", ["ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Status}}"], None, 15
        )
        if code != 0:
            return ToolResult(success=False, error=err.strip() or "Docker is not available.")
        containers = [line for line in out.splitlines() if line.strip()]
        return ToolResult(
            success=True,
            data={"containers": containers},
            message=f"{len(containers)} container(s) running.",
        )
