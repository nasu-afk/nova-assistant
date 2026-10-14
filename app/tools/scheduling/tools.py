"""Reminder tool, built on the Phase 11 `TaskScheduler`.

A reminder is a one-shot scheduled job: when it fires, `on_fire` is called
with the reminder text — the caller (see `app/main.py`) decides what that
means (speak it via TTS, print it, both). This tool only handles scheduling
the job and returning an immediate confirmation; it doesn't know or care how
the reminder is ultimately delivered.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.scheduler.scheduler import TaskScheduler
from app.tools.base import Tool, ToolMetadata, ToolResult

logger = get_logger(__name__)

if TYPE_CHECKING:
    from app.scheduler.reminders import PersistentReminderManager


class SetReminderTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="set_reminder",
        description=(
            "Set a one-time reminder that will be spoken back after a number of minutes "
            "(e.g. 'remind me to take a break in 20 minutes'). The reminder fires once, "
            "even if NOVA is doing something else at the time."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What to remind the user about"},
                "minutes": {
                    "type": "number",
                    "description": "How many minutes from now to fire the reminder",
                },
            },
            "required": ["message", "minutes"],
        },
    )

    def __init__(
        self,
        scheduler: TaskScheduler,
        on_fire: Callable[[str], Awaitable[None]],
        reminders: PersistentReminderManager | None = None,
    ) -> None:
        self._scheduler = scheduler
        self._on_fire = on_fire
        self._reminders = reminders

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        message = payload.get("message")
        minutes = payload.get("minutes")
        if not message or not isinstance(message, str):
            raise ValidationError("set_reminder requires a 'message' string")
        if not isinstance(minutes, int | float) or minutes <= 0:
            raise ValidationError("set_reminder requires a positive numeric 'minutes'")
        return {"message": message, "minutes": float(minutes)}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        message = payload["message"]
        minutes = payload["minutes"]

        data: dict[str, Any] = {"message": message, "minutes": minutes}
        if self._reminders is not None:
            record = await self._reminders.schedule(message, delay_seconds=minutes * 60)
            data.update({"reminder_id": record.id, "due_at": record.due_at.isoformat()})
        else:
            async def fire() -> None:
                logger.info("reminder_fired", message=message)
                await self._on_fire(message)

            self._scheduler.schedule_once(fire, delay_seconds=minutes * 60, name=message)

        unit = "minute" if minutes == 1 else "minutes"
        return ToolResult(
            success=True,
            message=f"I'll remind you in {minutes:g} {unit}: {message}",
            data=data,
        )
