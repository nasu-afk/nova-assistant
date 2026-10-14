"""Executes a `Plan` step by step, publishing events, and decides what to do
after each tool result (section 14: observation + re-planning).

Re-planning itself (asking the LLM for a *new* plan after a failure) is left
as a hook (`on_step_failed`) rather than automated end-to-end here — Phase 5
establishes the observation loop; a full LLM-driven re-plan is one call to
`Planner.create_plan` away and is wired in by whatever owns the `Agent`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from app.core.enums import EventType, TaskStatus
from app.core.errors import NovaError
from app.core.logging import get_logger
from app.core.planning import Plan, PlanStep
from app.events.bus import Event, EventBus
from app.tools.registry import ToolRegistry

logger = get_logger(__name__)

# Called after a step fails; return True to keep executing remaining steps
# anyway, False to stop the whole plan. Defaults to "stop" (see below).
FailureHandler = Callable[[Plan, PlanStep, str], Awaitable[bool]]


async def _default_failure_handler(plan: Plan, step: PlanStep, error: str) -> bool:
    return False


class TaskExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        events: EventBus | None = None,
        on_step_failed: FailureHandler = _default_failure_handler,
    ) -> None:
        self._registry = registry
        self._events = events or EventBus()
        self._on_step_failed = on_step_failed

    async def _publish(self, event_type: EventType, plan: Plan, **payload: Any) -> None:
        await self._events.publish(Event(type=event_type, task_id=plan.id, payload=payload))

    async def run(self, plan: Plan) -> Plan:
        """Execute every step in order, mutating and returning the same `Plan`."""
        plan.status = TaskStatus.RUNNING
        await self._publish(EventType.TASK_CREATED, plan, goal=plan.goal)

        for step in plan.steps:
            step.status = TaskStatus.RUNNING

            if not step.is_tool_step:
                # A note-only step: nothing to execute, just mark it done.
                step.status = TaskStatus.COMPLETED
                step.result_summary = step.description
                continue

            await self._publish(
                EventType.TOOL_REQUESTED, plan, step_id=step.id, tool=step.tool_name
            )
            try:
                assert step.tool_name is not None
                await self._publish(
                    EventType.TOOL_STARTED, plan, step_id=step.id, tool=step.tool_name
                )
                result = await self._registry.invoke(step.tool_name, step.tool_args)
            except Exception as exc:  # noqa: BLE001 - a crashing tool fails its step, not the run
                user_message = (
                    exc.user_message
                    if isinstance(exc, NovaError)
                    else (f"{step.tool_name} failed unexpectedly: {exc}")
                )
                step.status = TaskStatus.FAILED
                step.result_summary = user_message
                await self._publish(
                    EventType.TOOL_FAILED,
                    plan,
                    step_id=step.id,
                    tool=step.tool_name,
                    error=user_message,
                )
                should_continue = await self._on_step_failed(plan, step, user_message)
                if not should_continue:
                    plan.status = TaskStatus.FAILED
                    await self._publish(EventType.TASK_FAILED, plan, step_id=step.id)
                    return plan
                continue

            if result.success:
                step.status = TaskStatus.COMPLETED
                step.result_summary = result.message
                await self._publish(
                    EventType.TOOL_COMPLETED, plan, step_id=step.id, tool=step.tool_name
                )
            else:
                step.status = TaskStatus.FAILED
                step.result_summary = result.error
                await self._publish(
                    EventType.TOOL_FAILED,
                    plan,
                    step_id=step.id,
                    tool=step.tool_name,
                    error=result.error,
                )
                should_continue = await self._on_step_failed(plan, step, result.error or "")
                if not should_continue:
                    plan.status = TaskStatus.FAILED
                    await self._publish(EventType.TASK_FAILED, plan, step_id=step.id)
                    return plan

        plan.status = TaskStatus.COMPLETED if not plan.has_failure() else TaskStatus.FAILED
        if plan.status == TaskStatus.COMPLETED:
            event_type = EventType.TASK_COMPLETED
        else:
            event_type = EventType.TASK_FAILED
        await self._publish(event_type, plan)
        return plan
