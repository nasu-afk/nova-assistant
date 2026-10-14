from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.ai.base import ChatMessage, LLMResponse
from app.core.enums import EventType, RiskLevel
from app.core.proactive import ProactiveMonitor, ProactiveRule
from app.events.bus import Event, EventBus
from app.scheduler.scheduler import TaskScheduler
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.vision.screen import DescribeScreenTool


class FakeScreenshotTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="take_screenshot", description="fake", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = payload["path"]
        await asyncio.to_thread(Path(path).write_bytes, b"\x89PNG\r\n\x1a\nfakeimagecontent")
        return ToolResult(success=True, data={"path": path}, message="ok")


class FailingScreenshotTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="take_screenshot", description="fake", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="no display")


class FakeVisionLLM:
    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        has_image = "image_base64" in messages[0].metadata
        return LLMResponse(content="I see a code editor." if has_image else "No image provided.")


# --- DescribeScreenTool ---


@pytest.mark.asyncio
async def test_describe_screen_returns_llm_description() -> None:
    tool = DescribeScreenTool(FakeVisionLLM(), screenshot_tool=FakeScreenshotTool())  # type: ignore[arg-type]

    result = await tool.execute(tool.validate_input({}))

    assert result.success is True
    assert "code editor" in result.data["description"]


@pytest.mark.asyncio
async def test_describe_screen_fails_gracefully_if_capture_fails() -> None:
    tool = DescribeScreenTool(FakeVisionLLM(), screenshot_tool=FailingScreenshotTool())  # type: ignore[arg-type]

    result = await tool.execute(tool.validate_input({}))

    assert result.success is False
    assert "no display" in (result.error or "")


def test_describe_screen_default_question() -> None:
    tool = DescribeScreenTool(FakeVisionLLM(), screenshot_tool=FakeScreenshotTool())  # type: ignore[arg-type]
    payload = tool.validate_input({})
    assert "screen" in payload["question"].lower()


# --- TaskScheduler ---


@pytest.mark.asyncio
async def test_schedule_once_runs_after_delay() -> None:
    scheduler = TaskScheduler()
    calls = []

    async def job() -> None:
        calls.append(1)

    handle = scheduler.schedule_once(job, delay_seconds=0.01)
    await scheduler.wait_for(handle.id)

    assert calls == [1]


@pytest.mark.asyncio
async def test_schedule_recurring_runs_multiple_times_then_cancelled() -> None:
    scheduler = TaskScheduler()
    calls = []

    async def job() -> None:
        calls.append(1)

    handle = scheduler.schedule_recurring(job, interval_seconds=0.01)
    await asyncio.sleep(0.05)
    scheduler.cancel(handle.id)
    count_at_cancel = len(calls)
    await asyncio.sleep(0.05)

    assert count_at_cancel >= 2
    assert len(calls) == count_at_cancel  # no more calls after cancel


@pytest.mark.asyncio
async def test_cancel_prevents_one_shot_job_from_running() -> None:
    scheduler = TaskScheduler()
    calls = []

    async def job() -> None:
        calls.append(1)

    handle = scheduler.schedule_once(job, delay_seconds=0.05)
    scheduler.cancel(handle.id)
    await asyncio.sleep(0.1)

    assert calls == []


@pytest.mark.asyncio
async def test_active_jobs_excludes_cancelled() -> None:
    scheduler = TaskScheduler()

    async def job() -> None:
        pass

    handle = scheduler.schedule_once(job, delay_seconds=1.0)
    assert len(scheduler.active_jobs()) == 1

    scheduler.cancel(handle.id)
    assert len(scheduler.active_jobs()) == 0


@pytest.mark.asyncio
async def test_a_failing_job_does_not_crash_the_scheduler() -> None:
    scheduler = TaskScheduler()

    async def broken_job() -> None:
        raise RuntimeError("boom")

    handle = scheduler.schedule_once(broken_job, delay_seconds=0.01)
    await scheduler.wait_for(handle.id)  # should not raise


def test_cancel_unknown_job_returns_false() -> None:
    scheduler = TaskScheduler()
    assert scheduler.cancel("does-not-exist") is False


# --- ProactiveMonitor ---


@pytest.mark.asyncio
async def test_proactive_rule_fires_when_predicate_matches() -> None:
    bus = EventBus()
    monitor = ProactiveMonitor(bus)
    triggered = []

    async def action(event: Event) -> None:
        triggered.append(event.payload)

    monitor.add_rule(
        ProactiveRule(
            name="uncommitted_changes",
            event_type=EventType.TOOL_COMPLETED,
            predicate=lambda e: e.payload.get("tool") == "git_status",
            action=action,
        )
    )

    await bus.publish(Event(type=EventType.TOOL_COMPLETED, payload={"tool": "git_status"}))
    await bus.publish(Event(type=EventType.TOOL_COMPLETED, payload={"tool": "read_file"}))

    assert triggered == [{"tool": "git_status"}]


def test_rules_property_lists_registered_rules() -> None:
    bus = EventBus()
    monitor = ProactiveMonitor(bus)

    async def action(event: Event) -> None:
        pass

    monitor.add_rule(
        ProactiveRule(
            name="r1", event_type=EventType.TASK_COMPLETED, predicate=lambda e: True, action=action
        )
    )

    assert [r.name for r in monitor.rules] == ["r1"]
