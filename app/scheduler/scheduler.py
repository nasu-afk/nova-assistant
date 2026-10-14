"""Scheduled + proactive task support (section 11, Phase 11).

Deliberately simple: an in-process asyncio scheduler, not a cron daemon. A
scheduled job is just "call this async function every N seconds" or "... once,
after N seconds". Generic jobs remain process-local; durable user reminders
are layered on top by ``app.scheduler.reminders.PersistentReminderManager``,
which stores them in SQLite and re-arms pending jobs at startup.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from uuid import uuid4

from app.core.logging import get_logger

logger = get_logger(__name__)

ScheduledCallback = Callable[[], Awaitable[None]]


@dataclass
class ScheduledJob:
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    name: str = ""
    interval_seconds: float | None = None  # None => one-shot
    delay_seconds: float = 0.0
    cancelled: bool = False


class TaskScheduler:
    """Runs callbacks on a delay and/or recurring interval."""

    def __init__(self) -> None:
        self._jobs: dict[str, ScheduledJob] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def schedule_once(
        self, callback: ScheduledCallback, delay_seconds: float, name: str = ""
    ) -> ScheduledJob:
        job = ScheduledJob(name=name, delay_seconds=delay_seconds, interval_seconds=None)
        self._jobs[job.id] = job
        self._tasks[job.id] = asyncio.create_task(self._run_once(job, callback))
        return job

    def schedule_recurring(
        self, callback: ScheduledCallback, interval_seconds: float, name: str = ""
    ) -> ScheduledJob:
        job = ScheduledJob(
            name=name, interval_seconds=interval_seconds, delay_seconds=interval_seconds
        )
        self._jobs[job.id] = job
        self._tasks[job.id] = asyncio.create_task(self._run_recurring(job, callback))
        return job

    async def _run_once(self, job: ScheduledJob, callback: ScheduledCallback) -> None:
        await asyncio.sleep(job.delay_seconds)
        if job.cancelled:
            return
        await self._invoke(job, callback)

    async def _run_recurring(self, job: ScheduledJob, callback: ScheduledCallback) -> None:
        while not job.cancelled:
            await asyncio.sleep(job.interval_seconds or 0)
            if job.cancelled:
                return
            await self._invoke(job, callback)

    async def _invoke(self, job: ScheduledJob, callback: ScheduledCallback) -> None:
        try:
            await callback()
        except Exception as exc:  # noqa: BLE001 - a broken job must not crash the scheduler
            logger.error("scheduled_job_failed", job=job.name or job.id, error=str(exc))

    def cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancelled = True
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        return True

    def cancel_all(self) -> None:
        for job_id in list(self._jobs):
            self.cancel(job_id)

    def active_jobs(self) -> list[ScheduledJob]:
        return [job for job in self._jobs.values() if not job.cancelled]

    async def wait_for(self, job_id: str) -> None:
        """Await a job's underlying task — mainly useful in tests."""
        task = self._tasks.get(job_id)
        if task is not None:
            await task
