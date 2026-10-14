from __future__ import annotations

import pytest

from app.core.enums import EventType
from app.events.bus import Event, EventBus


@pytest.mark.asyncio
async def test_subscriber_receives_published_event() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.TASK_CREATED, handler)
    await bus.publish(Event(type=EventType.TASK_CREATED, payload={"task_id": "t1"}))

    assert len(received) == 1
    assert received[0].payload["task_id"] == "t1"


@pytest.mark.asyncio
async def test_sync_handlers_are_supported() -> None:
    bus = EventBus()
    calls: list[str] = []

    def handler(event: Event) -> None:
        calls.append(event.type)

    bus.subscribe(EventType.TOOL_STARTED, handler)
    await bus.publish(Event(type=EventType.TOOL_STARTED))

    assert calls == [EventType.TOOL_STARTED]


@pytest.mark.asyncio
async def test_unsubscribed_handler_is_not_called() -> None:
    bus = EventBus()
    calls: list[str] = []

    def handler(event: Event) -> None:
        calls.append(event.type)

    bus.subscribe(EventType.TOOL_FAILED, handler)
    bus.unsubscribe(EventType.TOOL_FAILED, handler)
    await bus.publish(Event(type=EventType.TOOL_FAILED))

    assert calls == []


@pytest.mark.asyncio
async def test_event_with_no_subscribers_does_not_raise() -> None:
    bus = EventBus()
    await bus.publish(Event(type=EventType.TASK_COMPLETED))


@pytest.mark.asyncio
async def test_multiple_handlers_all_receive_event() -> None:
    bus = EventBus()
    counter = {"a": 0, "b": 0}

    async def handler_a(event: Event) -> None:
        counter["a"] += 1

    async def handler_b(event: Event) -> None:
        counter["b"] += 1

    bus.subscribe(EventType.PLAN_CREATED, handler_a)
    bus.subscribe(EventType.PLAN_CREATED, handler_b)
    await bus.publish(Event(type=EventType.PLAN_CREATED))

    assert counter == {"a": 1, "b": 1}
