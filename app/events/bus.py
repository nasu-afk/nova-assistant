"""Internal event bus.

The agent, tools, and security layer publish events here; the GUI and the
logging/observability layer subscribe. Nothing should import the GUI from
core agent code — this bus is the decoupling point (section 12).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.core.enums import EventType

EventHandler = Callable[["Event"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class Event:
    """A single event published on the bus."""

    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class EventBus:
    """A minimal pub/sub bus. Deliberately dependency-free (no Redis, etc.)
    per the project's "prefer simple architecture" principle — a single
    process, in-memory bus is sufficient for a desktop application.
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register `handler` to be called whenever `event_type` is published."""
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Remove a previously registered handler. No-op if not found."""
        handlers = self._handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish `event` to every subscriber, awaiting async handlers.

        A failing handler is logged (once logging is wired in Phase 1+) and
        does not prevent other handlers from running — a broken GUI
        subscriber must never take down the agent.
        """
        for handler in list(self._handlers.get(event.type, [])):
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
