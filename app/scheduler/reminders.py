"""Durable reminders layered on top of the in-process task scheduler."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sqlalchemy import select, update

from app.database.engine import Database
from app.database.models import ReminderRecord
from app.scheduler.scheduler import TaskScheduler

ReminderCallback = Callable[[str], Awaitable[None]]


def _as_utc(value: datetime) -> datetime:
    # SQLite may return a naive value even for DateTime(timezone=True).
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class PersistentReminderManager:
    """Persists one-shot reminders and re-arms pending ones after startup."""

    def __init__(
        self,
        db: Database,
        scheduler: TaskScheduler,
        on_fire: ReminderCallback,
    ) -> None:
        self._db = db
        self._scheduler = scheduler
        self._on_fire = on_fire
        self._scheduled: dict[str, str] = {}

    async def schedule(self, message: str, delay_seconds: float) -> ReminderRecord:
        record = ReminderRecord(
            id=uuid4().hex[:12],
            message=message,
            due_at=datetime.now(UTC) + timedelta(seconds=delay_seconds),
            status="pending",
        )
        async with self._db.session() as session:
            session.add(record)
            await session.commit()
        self._arm(record)
        return record

    async def restore_pending(self) -> int:
        async with self._db.session() as session:
            result = await session.execute(
                select(ReminderRecord)
                .where(ReminderRecord.status == "pending")
                .order_by(ReminderRecord.due_at)
            )
            records = list(result.scalars().all())
        for record in records:
            self._arm(record)
        return len(records)

    def _arm(self, record: ReminderRecord) -> None:
        if record.id in self._scheduled:
            return
        delay = max(0.0, (_as_utc(record.due_at) - datetime.now(UTC)).total_seconds())

        async def fire() -> None:
            try:
                await self._on_fire(record.message)
                await self._mark_fired(record.id)
            finally:
                self._scheduled.pop(record.id, None)

        job = self._scheduler.schedule_once(fire, delay_seconds=delay, name=record.message)
        self._scheduled[record.id] = job.id

    async def _mark_fired(self, reminder_id: str) -> None:
        async with self._db.session() as session:
            await session.execute(
                update(ReminderRecord)
                .where(ReminderRecord.id == reminder_id)
                .values(status="fired", fired_at=datetime.now(UTC))
            )
            await session.commit()

    async def pending(self) -> list[ReminderRecord]:
        async with self._db.session() as session:
            result = await session.execute(
                select(ReminderRecord)
                .where(ReminderRecord.status == "pending")
                .order_by(ReminderRecord.due_at)
            )
            return list(result.scalars().all())

