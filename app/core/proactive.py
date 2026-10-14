"""Proactive automation.

A `ProactiveRule` watches one `EventType` and, when a supplied predicate
matches the event's payload, runs an action — e.g. "when a `git_status`
tool call reports uncommitted changes older than a day, suggest a commit."
This is intentionally built on the same `EventBus` and `TaskScheduler`
already in place rather than a new engine, keeping with "avoid unnecessary
abstractions."
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.core.enums import EventType
from app.core.logging import get_logger
from app.events.bus import Event, EventBus

logger = get_logger(__name__)

Predicate = Callable[[Event], bool]
Action = Callable[[Event], Awaitable[None]]


@dataclass
class ProactiveRule:
    name: str
    event_type: EventType
    predicate: Predicate
    action: Action


class ProactiveMonitor:
    """Registers `ProactiveRule`s against an `EventBus` and fires their
    action whenever a matching event is published."""

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._rules: list[ProactiveRule] = []

    def add_rule(self, rule: ProactiveRule) -> None:
        self._rules.append(rule)

        async def handler(event: Event) -> None:
            if rule.predicate(event):
                logger.info("proactive_rule_triggered", rule=rule.name)
                await rule.action(event)

        self._bus.subscribe(rule.event_type, handler)

    @property
    def rules(self) -> list[ProactiveRule]:
        return list(self._rules)
