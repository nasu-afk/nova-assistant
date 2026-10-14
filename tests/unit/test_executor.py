from __future__ import annotations

from typing import ClassVar

import pytest

from app.core.enums import EventType, RiskLevel, TaskStatus
from app.core.executor import TaskExecutor
from app.core.planning import Plan, PlanStep
from app.events.bus import Event, EventBus
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.registry import ToolRegistry


class SucceedingTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="succeed", description="Always succeeds.", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict) -> ToolResult:  # type: ignore[type-arg]
        return ToolResult(success=True, message="done")


class FailingTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="fail", description="Always fails.", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict) -> ToolResult:  # type: ignore[type-arg]
        return ToolResult(success=False, error="boom")


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SucceedingTool())
    reg.register(FailingTool())
    return reg


@pytest.mark.asyncio
async def test_all_steps_succeed_marks_plan_completed(registry: ToolRegistry) -> None:
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(description="step 1", tool_name="succeed"),
            PlanStep(description="step 2", tool_name="succeed"),
        ],
    )
    executor = TaskExecutor(registry)

    result = await executor.run(plan)

    assert result.status == TaskStatus.COMPLETED
    assert all(s.status == TaskStatus.COMPLETED for s in result.steps)


@pytest.mark.asyncio
async def test_note_only_step_completes_without_tool_call(registry: ToolRegistry) -> None:
    plan = Plan(goal="test", steps=[PlanStep(description="just a note", tool_name=None)])
    executor = TaskExecutor(registry)

    result = await executor.run(plan)

    assert result.status == TaskStatus.COMPLETED
    assert result.steps[0].result_summary == "just a note"


@pytest.mark.asyncio
async def test_failing_step_stops_plan_by_default(registry: ToolRegistry) -> None:
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(description="ok", tool_name="succeed"),
            PlanStep(description="bad", tool_name="fail"),
            PlanStep(description="never runs", tool_name="succeed"),
        ],
    )
    executor = TaskExecutor(registry)

    result = await executor.run(plan)

    assert result.status == TaskStatus.FAILED
    assert result.steps[0].status == TaskStatus.COMPLETED
    assert result.steps[1].status == TaskStatus.FAILED
    assert result.steps[2].status == TaskStatus.PENDING  # never reached


@pytest.mark.asyncio
async def test_failure_handler_can_allow_continuation(registry: ToolRegistry) -> None:
    plan = Plan(
        goal="test",
        steps=[
            PlanStep(description="bad", tool_name="fail"),
            PlanStep(description="still runs", tool_name="succeed"),
        ],
    )

    async def keep_going(plan: Plan, step: PlanStep, error: str) -> bool:
        return True

    executor = TaskExecutor(registry, on_step_failed=keep_going)
    result = await executor.run(plan)

    assert result.steps[1].status == TaskStatus.COMPLETED
    # Overall plan is still marked failed since a step failed, even though
    # execution continued past it.
    assert result.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_events_are_published_for_full_lifecycle(registry: ToolRegistry) -> None:
    events = EventBus()
    seen: list[EventType] = []

    async def record(event: Event) -> None:
        seen.append(event.type)

    for event_type in EventType:
        events.subscribe(event_type, record)

    plan = Plan(goal="test", steps=[PlanStep(description="ok", tool_name="succeed")])
    executor = TaskExecutor(registry, events=events)

    await executor.run(plan)

    assert EventType.TASK_CREATED in seen
    assert EventType.TOOL_STARTED in seen
    assert EventType.TOOL_COMPLETED in seen
    assert EventType.TASK_COMPLETED in seen


@pytest.mark.asyncio
async def test_tool_events_carry_the_tool_name(registry: ToolRegistry) -> None:
    events = EventBus()
    captured: list[Event] = []

    async def record(event: Event) -> None:
        captured.append(event)

    events.subscribe(EventType.TOOL_STARTED, record)
    events.subscribe(EventType.TOOL_COMPLETED, record)

    plan = Plan(goal="test", steps=[PlanStep(description="ok", tool_name="succeed")])
    executor = TaskExecutor(registry, events=events)

    await executor.run(plan)

    assert all(e.payload.get("tool") == "succeed" for e in captured)


@pytest.mark.asyncio
async def test_tool_failed_event_carries_the_tool_name(registry: ToolRegistry) -> None:
    events = EventBus()
    captured: list[Event] = []

    async def record(event: Event) -> None:
        captured.append(event)

    events.subscribe(EventType.TOOL_FAILED, record)

    plan = Plan(goal="test", steps=[PlanStep(description="bad", tool_name="fail")])
    executor = TaskExecutor(registry, events=events)

    await executor.run(plan)

    assert captured[0].payload.get("tool") == "fail"


@pytest.mark.asyncio
async def test_tool_not_found_is_treated_as_step_failure(registry: ToolRegistry) -> None:
    plan = Plan(goal="test", steps=[PlanStep(description="ghost", tool_name="does_not_exist")])
    executor = TaskExecutor(registry)

    result = await executor.run(plan)

    assert result.status == TaskStatus.FAILED
    assert result.steps[0].status == TaskStatus.FAILED
