from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.errors import ValidationError
from app.security.manager import CommandPolicy
from app.tools.developer.tools import (
    DockerStatusTool,
    GitLogTool,
    GitStatusTool,
    RunCommandTool,
    RunGradleTool,
    RunMavenTool,
    RunPythonTool,
)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "nova@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "NOVA"], cwd=tmp_path, check=True)
    (tmp_path / "README.md").write_text("# test repo")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial commit"], cwd=tmp_path, check=True)
    return tmp_path


@pytest.mark.asyncio
async def test_git_status_clean_repo(git_repo: Path) -> None:
    tool = GitStatusTool()
    result = await tool.execute(tool.validate_input({"cwd": str(git_repo)}))

    assert result.success is True


@pytest.mark.asyncio
async def test_git_status_shows_untracked_file(git_repo: Path) -> None:
    (git_repo / "new_file.txt").write_text("hello")
    tool = GitStatusTool()

    result = await tool.execute(tool.validate_input({"cwd": str(git_repo)}))

    assert "new_file.txt" in result.data["status"]


@pytest.mark.asyncio
async def test_git_log_returns_commits(git_repo: Path) -> None:
    tool = GitLogTool()
    result = await tool.execute(tool.validate_input({"cwd": str(git_repo), "limit": 5}))

    assert result.success is True
    assert len(result.data["commits"]) == 1
    assert "initial commit" in result.data["commits"][0]


@pytest.mark.asyncio
async def test_git_status_non_repo_fails_gracefully(tmp_path: Path) -> None:
    tool = GitStatusTool()
    result = await tool.execute(tool.validate_input({"cwd": str(tmp_path)}))

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_run_python_executes_script(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello from nova')")
    tool = RunPythonTool()

    result = await tool.execute(tool.validate_input({"cwd": str(tmp_path), "script": str(script)}))

    assert result.success is True
    assert "hello from nova" in result.data["stdout"]


@pytest.mark.asyncio
async def test_run_python_reports_traceback_on_error(tmp_path: Path) -> None:
    script = tmp_path / "broken.py"
    script.write_text("raise ValueError('boom')")
    tool = RunPythonTool()

    result = await tool.execute(tool.validate_input({"cwd": str(tmp_path), "script": str(script)}))

    assert result.success is False
    assert "boom" in (result.error or "")


def test_run_python_requires_script() -> None:
    with pytest.raises(ValidationError):
        RunPythonTool().validate_input({})


@pytest.mark.asyncio
async def test_missing_binary_reports_failure_not_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def missing_binary(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("test binary is unavailable")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing_binary)

    for tool_cls in (RunMavenTool, RunGradleTool, DockerStatusTool):
        tool = tool_cls()
        if tool_cls is DockerStatusTool:
            payload: dict[str, object] = {}
        else:
            payload = tool.validate_input({"cwd": str(tmp_path)})
        result = await tool.execute(payload)
        assert result.success is False
        assert result.error is not None


# --- RunCommandTool ---


@pytest.mark.asyncio
async def test_run_command_executes_allowed_command(tmp_path: Path) -> None:
    tool = RunCommandTool(CommandPolicy())
    command = f'{shlex.quote(sys.executable)} -c "print(\'hello nova\')"'
    payload = tool.validate_input({"command": command})

    result = await tool.execute(payload)

    assert result.success is True
    assert "hello nova" in result.data["stdout"]


@pytest.mark.asyncio
async def test_run_command_blocked_by_denylist() -> None:
    tool = RunCommandTool(CommandPolicy(denylist=["rm -rf /"]))
    payload = tool.validate_input({"command": "rm -rf / --no-preserve-root"})

    result = await tool.execute(payload)

    assert result.success is False
    assert "denylisted" in (result.error or "")


@pytest.mark.asyncio
async def test_run_command_blocked_by_missing_allowlist_entry() -> None:
    tool = RunCommandTool(CommandPolicy(allowlist=["git"]))
    payload = tool.validate_input({"command": "curl http://evil.example"})

    result = await tool.execute(payload)

    assert result.success is False


@pytest.mark.asyncio
async def test_run_command_times_out() -> None:
    tool = RunCommandTool(CommandPolicy(timeout_seconds=1))
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(5)"'
    payload = tool.validate_input({"command": command})

    result = await tool.execute(payload)

    assert result.success is False
    assert "timed out" in (result.error or "")


@pytest.mark.asyncio
async def test_run_command_nonzero_exit_is_failure() -> None:
    tool = RunCommandTool(CommandPolicy())
    command = f'{shlex.quote(sys.executable)} -c "raise SystemExit(1)"'
    payload = tool.validate_input({"command": command})

    result = await tool.execute(payload)

    assert result.success is False


def test_run_command_requires_command_string() -> None:
    with pytest.raises(ValidationError):
        RunCommandTool().validate_input({})


def test_run_command_is_high_risk_and_requires_confirmation() -> None:
    assert RunCommandTool.metadata.requires_confirmation is True
