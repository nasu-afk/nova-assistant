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


@pytest.mark.asyncio
async def test_stay_open_returns_once_window_is_closed() -> None:
    """Closing the window must let the GUI loop finish its cleanup instead of
    blocking forever (which previously forced the Qt loop to die mid-coroutine)."""
    window = FakeWindow()
    closed = asyncio.Event()

    task = asyncio.ensure_future(
        _stay_open_after_setup_failure(window, "boom", "Startup failed", closed)
    )
    await asyncio.sleep(0.01)
    assert not task.done()

    closed.set()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_stack_close_closes_tools_that_hold_resources() -> None:
    from app.scheduler.scheduler import TaskScheduler
    from app.tools.registry import ToolRegistry

    class ClosableTool:
        class metadata:  # noqa: N801 - mimics ToolMetadata attribute access
            name = "closable"

        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class FailingCloseTool:
        class metadata:  # noqa: N801
            name = "failing"

        async def aclose(self) -> None:
            raise RuntimeError("already gone")

    class FakeDatabase:
        disposed = False

        async def dispose(self) -> None:
            self.disposed = True

    registry = ToolRegistry()
    good, bad = ClosableTool(), FailingCloseTool()
    registry._tools["failing"] = bad  # type: ignore[assignment]
    registry._tools["closable"] = good  # type: ignore[assignment]
    database = FakeDatabase()
    stack = _Stack(
        agent=object(),  # type: ignore[arg-type]
        registry=registry,
        stt=object(),  # type: ignore[arg-type]
        tts=object(),  # type: ignore[arg-type]
        memory=object(),  # type: ignore[arg-type]
        scheduler=TaskScheduler(),
        llm=object(),  # type: ignore[arg-type]
        database=database,  # type: ignore[arg-type]
    )

    await stack.close()  # a failing closer must not stop the rest of shutdown

    assert good.closed is True
    assert database.disposed is True


@pytest.mark.asyncio
async def test_build_stack_wires_everything_and_runs_a_tool_end_to_end(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Integration check of `_build_stack` (real DB, memory, scheduler, registry,
    settings wiring) with a stubbed LLM transport — no network."""
    from app.ai.base import LLMResponse
    from app.config.settings import LLMProviderName, Settings, VoiceProviderName
    from app.main import _build_stack

    notes = tmp_path / "notes.txt"
    notes.write_text("remember the milk")
    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_provider=LLMProviderName.GROQ,
        llm_api_key="gsk_fake",
        voice_provider=VoiceProviderName.NONE,
        data_dir=tmp_path / "data",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'nova.db'}",
        workspace_root=str(tmp_path),
    )
    settings.ensure_data_dir()
    stack = await _build_stack(settings)
    try:
        names = {m.name for m in stack.registry.list_metadata()}
        assert {"read_file", "remember_fact", "set_reminder", "describe_screen"} <= names
        assert "run_command" not in names  # developer tools are opt-in
        assert settings.llm_model == "llama-3.3-70b-versatile"  # provider default applied

        calls = iter(
            [
                LLMResponse(
                    content="",
                    tool_calls=[{"id": "1", "name": "read_file", "input": {"path": str(notes)}}],
                ),
                LLMResponse(content="Your note says remember the milk."),
            ]
        )
        seen: list[str] = []

        async def fake_complete(messages, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(messages[-1].content)
            return next(calls)

        stack.llm.complete = fake_complete  # type: ignore[method-assign]
        reply = await stack.agent.respond("what's in notes.txt?")

        assert reply == "Your note says remember the milk."
        assert "remember the milk" in seen[1]  # the file content reached the model
    finally:
        await stack.close()
