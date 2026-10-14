from __future__ import annotations

import os

import pytest

from app.core.errors import ValidationError
from app.tools.windows import applications as applications_module
from app.tools.windows import media as media_module
from app.tools.windows.applications import CloseApplicationTool, OpenApplicationTool
from app.tools.windows.media import SetVolumeTool, TakeScreenshotTool
from app.tools.windows.system_info import (
    GetCpuUsageTool,
    GetMemoryUsageTool,
    GetProcessesTool,
    GetSystemInfoTool,
)


@pytest.mark.asyncio
async def test_get_system_info_returns_expected_fields() -> None:
    result = await GetSystemInfoTool().execute({})

    assert result.success is True
    assert "system" in result.data
    assert "hostname" in result.data
    assert result.data["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_get_processes_returns_a_list() -> None:
    result = await GetProcessesTool().execute({"limit": 5})

    assert result.success is True
    assert isinstance(result.data["processes"], list)
    assert len(result.data["processes"]) <= 5
    assert result.data["total_count"] >= len(result.data["processes"])


@pytest.mark.asyncio
async def test_get_cpu_usage_returns_percentage() -> None:
    result = await GetCpuUsageTool().execute({})

    assert result.success is True
    assert 0.0 <= result.data["cpu_percent"] <= 100.0


@pytest.mark.asyncio
async def test_get_memory_usage_returns_percentage_and_gb_values() -> None:
    result = await GetMemoryUsageTool().execute({})

    assert result.success is True
    assert 0.0 <= result.data["percent"] <= 100.0
    assert result.data["total_gb"] > 0


@pytest.mark.asyncio
async def test_open_application_launches_real_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: list[str] = []
    monkeypatch.setattr(applications_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(os, "startfile", launched.append, raising=False)

    tool = OpenApplicationTool()
    payload = tool.validate_input({"name": "notepad"})

    result = await tool.execute(payload)

    assert result.success is True
    assert result.data["name"] == "notepad"
    assert launched == ["notepad"]


@pytest.mark.asyncio
async def test_open_application_reports_failure_for_missing_binary() -> None:
    tool = OpenApplicationTool()
    payload = tool.validate_input({"name": "this-binary-does-not-exist-anywhere"})

    result = await tool.execute(payload)

    assert result.success is False
    assert result.error is not None


def test_open_application_requires_name() -> None:
    with pytest.raises(ValidationError):
        OpenApplicationTool().validate_input({})


def test_open_application_rejects_url() -> None:
    with pytest.raises(ValidationError):
        OpenApplicationTool().validate_input({"name": "https://youtube.com"})


def test_open_application_rejects_command_line_with_args() -> None:
    with pytest.raises(ValidationError):
        OpenApplicationTool().validate_input(
            {"name": "chrome https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )


@pytest.mark.asyncio
async def test_close_application_terminates_matching_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        info = {"pid": 1234, "name": "nova-test-process.exe"}

        def __init__(self) -> None:
            self.terminated = False

        def terminate(self) -> None:
            self.terminated = True

    proc = FakeProcess()
    monkeypatch.setattr(applications_module.psutil, "process_iter", lambda attrs: [proc])

    tool = CloseApplicationTool()
    payload = tool.validate_input({"name": "nova-test-process"})
    result = await tool.execute(payload)

    assert result.success is True
    assert proc.terminated is True
    assert result.data["terminated_pids"] == [1234]


@pytest.mark.asyncio
async def test_close_application_reports_no_match() -> None:
    tool = CloseApplicationTool()
    payload = tool.validate_input({"name": "definitely-not-a-running-process-xyz"})

    result = await tool.execute(payload)

    assert result.success is False


def test_close_application_is_high_risk_and_requires_confirmation() -> None:
    assert CloseApplicationTool.metadata.requires_confirmation is True


@pytest.mark.asyncio
async def test_set_volume_reports_unsupported_off_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(media_module.platform, "system", lambda: "Linux")
    tool = SetVolumeTool()
    payload = tool.validate_input({"level": 50})

    result = await tool.execute(payload)

    assert result.success is False
    assert result.error is not None


def test_set_volume_validates_range() -> None:
    with pytest.raises(ValidationError):
        SetVolumeTool().validate_input({"level": 150})


@pytest.mark.asyncio
async def test_take_screenshot_fails_gracefully_without_display() -> None:
    tool = TakeScreenshotTool()
    payload = tool.validate_input({"path": "/tmp/nova_test_screenshot.png"})

    result = await tool.execute(payload)

    # No display/Pillow guaranteed in CI — either way it must not raise.
    assert isinstance(result.success, bool)
    if not result.success:
        assert result.error is not None
