"""Tests for the small, directly-importable pieces of app/main.py's GUI
loop. `run_gui_voice_loop` itself needs a real QApplication + qasync event
loop + microphone/hotkey hardware and isn't unit-testable end to end (see
its docstring) — but `_stay_open_after_setup_failure` is the one piece that
fixes a real bug (the window closing instantly on a setup failure instead
of staying open to show the error) and is small enough to test in
isolation.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from app.main import _Stack, _stay_open_after_setup_failure


class FakeWindow:
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.status: str | None = None

    def append_log(self, line: str) -> None:
        self.logs.append(line)

    def set_status(self, text: str) -> None:
        self.status = text


@pytest.mark.asyncio
async def test_logs_message_and_status_before_blocking() -> None:
    window = FakeWindow()

    task = asyncio.ensure_future(
        _stay_open_after_setup_failure(
            window, "Microphone isn't available: no device", "Microphone unavailable"
        )
    )
    await asyncio.sleep(0.05)  # let it run up to the blocking wait

    assert window.logs == ["Microphone isn't available: no device"]
    assert window.status == "Microphone unavailable"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_blocks_forever_instead_of_returning() -> None:
    """This is the actual regression test: before the fix, the equivalent
    code path returned immediately after logging, which — in the real
    `run_gui_voice_loop` — ends `loop.run_until_complete()` and closes the
    whole window before the error is ever visible. Confirm this helper
    genuinely does not complete on its own."""
    window = FakeWindow()

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            _stay_open_after_setup_failure(window, "some error", "Unavailable"), timeout=0.1
        )


@pytest.mark.asyncio
async def test_stack_close_cancels_jobs_and_disposes_database() -> None:
    class FakeScheduler:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel_all(self) -> None:
            self.cancelled = True

    class FakeDatabase:
        def __init__(self) -> None:
            self.disposed = False

        async def dispose(self) -> None:
            self.disposed = True

    scheduler = FakeScheduler()
    database = FakeDatabase()
    placeholder = object()
    stack = _Stack(
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        placeholder,
        scheduler,
        placeholder,
        database,
    )

    await stack.close()

    assert scheduler.cancelled is True
    assert database.disposed is True
