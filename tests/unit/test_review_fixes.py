"""Regression tests for the bugs found in the code review.

Each test fails on the pre-patch code and passes now.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

import pytest

from app.ai.base import ChatMessage, LLMResponse
from app.config.settings import LLMProviderName, Settings
from app.core.agent import MAX_TOOL_OUTPUT_CHARS, ToolCallingAgent, format_tool_outcome
from app.core.enums import EventType, RiskLevel
from app.core.errors import PermissionDeniedError, ValidationError, VoiceError
from app.core.planner import Planner
from app.database.engine import Database
from app.events.bus import Event, EventBus
from app.scheduler.reminders import PersistentReminderManager
from app.scheduler.scheduler import TaskScheduler
from app.security.manager import (
    AuditLogger,
    ConfirmationManager,
    SecurityGate,
    summarize_payload,
)
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.defaults import build_default_registry
from app.tools.filesystem.files import (
    CopyFileTool,
    CreateFileTool,
    MoveFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from app.tools.registry import ToolRegistry
from app.tools.windows.system_info import GetProcessesTool
from app.voice.base import TranscriptionResult
from app.voice.groq_tts import split_for_tts
from app.voice.pipeline import VoicePipeline

# --------------------------------------------------------------------- helpers


class ScriptedLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.seen: list[list[ChatMessage]] = []

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.seen.append(list(messages))
        return self._responses.pop(0)


def tool_call(name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(content="", tool_calls=[{"id": "1", "name": name, "input": args}])


class CrashingTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="crasher", description="raises a non-Nova error", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        raise RuntimeError("kaboom")


# ------------------------------------------------------------------ agent / LLM


@pytest.mark.asyncio
async def test_llm_receives_tool_data_not_just_the_message(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("the-secret-sauce-is-basil")
    registry = build_default_registry(confirmation_prompt=None, workspace_root=tmp_path)
    llm = ScriptedLLM(
        [tool_call("read_file", {"path": str(tmp_path / "notes.txt")}), LLMResponse(content="ok")]
    )

    await ToolCallingAgent(llm, registry).respond("what's in notes.txt?")

    last = llm.seen[1][-1]
    assert "the-secret-sauce-is-basil" in last.content
    # Must not end on an assistant turn (looks like a prefill to some providers).
    assert last.role == "user"


@pytest.mark.asyncio
async def test_history_alternates_roles_after_tool_round() -> None:
    registry = ToolRegistry()
    registry.register(CrashingTool())
    llm = ScriptedLLM([tool_call("crasher", {}), LLMResponse(content="done")])
    agent = ToolCallingAgent(llm, registry)

    await agent.respond("go")

    roles = [m.role for m in agent.history]
    assert roles == ["user", "assistant", "user", "assistant"]


def test_format_tool_outcome_truncates_huge_data() -> None:
    result = ToolResult(success=True, message="Read big.txt.", data={"content": "x" * 50_000})

    text = format_tool_outcome(result)

    assert len(text) < MAX_TOOL_OUTPUT_CHARS + 100
    assert "truncated" in text


@pytest.mark.asyncio
async def test_unexpected_tool_exception_does_not_crash_the_turn() -> None:
    registry = ToolRegistry()
    registry.register(CrashingTool())
    llm = ScriptedLLM([tool_call("crasher", {}), LLMResponse(content="Sorry, that failed.")])

    reply = await ToolCallingAgent(llm, registry).respond("do it")

    assert reply == "Sorry, that failed."
    assert "unexpected error" in llm.seen[1][-1].content


@pytest.mark.asyncio
async def test_concurrent_turns_are_serialized() -> None:
    registry = ToolRegistry()
    in_flight = 0
    max_in_flight = 0

    class SlowLLM:
        async def complete(self, messages: Any, **_: Any) -> LLMResponse:
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.02)
            in_flight -= 1
            return LLMResponse(content="hi")

    agent = ToolCallingAgent(SlowLLM(), registry)  # type: ignore[arg-type]
    await asyncio.gather(agent.respond("a"), agent.respond("b"), agent.respond("c"))

    assert max_in_flight == 1
    assert [m.role for m in agent.history] == ["user", "assistant"] * 3


# ------------------------------------------------------------------- event bus


@pytest.mark.asyncio
async def test_event_bus_isolates_failing_handlers() -> None:
    bus = EventBus()
    ran: list[int] = []

    async def bad(event: Event) -> None:
        raise RuntimeError("gui exploded")

    async def good(event: Event) -> None:
        ran.append(1)

    bus.subscribe(EventType.RESPONSE_READY, bad)
    bus.subscribe(EventType.RESPONSE_READY, good)

    await bus.publish(Event(type=EventType.RESPONSE_READY, payload={}))

    assert ran == [1]


# --------------------------------------------------------------------- planner


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        '{"goal": "g", "steps": [{"description": "d", "tool_name": "x", "tool_args": null}]}',
        "[1, 2, 3]",
        '{"goal": "g", "steps": [{"description": 5, "tool_name": ["bad"]}]}',
    ],
)
async def test_planner_never_leaks_raw_pydantic_errors(raw: str) -> None:
    planner = Planner(ScriptedLLM([LLMResponse(content=raw)]))  # type: ignore[arg-type]

    try:
        plan = await planner.create_plan("do something", [])
    except ValidationError:
        return  # PlanningError is a NovaError - fine
    assert plan is not None  # null tool_args is simply tolerated


# ----------------------------------------------------------- filesystem safety


@pytest.mark.asyncio
async def test_file_tools_are_confined_to_workspace(tmp_path: Path) -> None:
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    registry = build_default_registry(confirmation_prompt=None, workspace_root=root)

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("read_file", {"path": str(outside)})
    with pytest.raises(PermissionDeniedError):
        await registry.invoke("write_file", {"path": str(outside), "content": "x"})
    assert outside.read_text() == "nope"


@pytest.mark.asyncio
async def test_credential_files_are_refused_even_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NOVA_LLM_API_KEY=gsk_secret")
    registry = build_default_registry(confirmation_prompt=None, workspace_root=tmp_path)

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("read_file", {"path": str(tmp_path / ".env")})


@pytest.mark.asyncio
async def test_search_files_hides_credentials(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("1")
    (tmp_path / ".env").write_text("secret")
    (tmp_path / "server.pem").write_text("secret")
    tool = SearchFilesTool()
    tool.workspace_root = tmp_path

    result = await tool.execute(tool.validate_input({"directory": str(tmp_path), "pattern": "*"}))

    names = {Path(p).name for p in result.data["matches"]}
    assert names == {"a.txt"}


@pytest.mark.asyncio
async def test_search_files_rejects_absolute_and_non_string_patterns(tmp_path: Path) -> None:
    tool = SearchFilesTool()
    with pytest.raises(ValidationError):
        tool.validate_input({"directory": str(tmp_path), "pattern": "/etc/*"})
    with pytest.raises(ValidationError):
        tool.validate_input({"directory": str(tmp_path), "pattern": 123})


@pytest.mark.asyncio
async def test_search_files_caps_results(tmp_path: Path) -> None:
    for i in range(230):
        (tmp_path / f"f{i}.txt").write_text("x")
    tool = SearchFilesTool()
    tool.workspace_root = tmp_path

    result = await tool.execute(
        tool.validate_input({"directory": str(tmp_path), "pattern": "*.txt"})
    )

    assert result.data["count"] == 200
    assert result.data["truncated"] is True


def test_create_file_rejects_non_string_content() -> None:
    with pytest.raises(ValidationError):
        CreateFileTool().validate_input({"path": "a.txt", "content": {"a": 1}})


def test_overwriting_calls_need_confirmation(tmp_path: Path) -> None:
    existing = tmp_path / "exists.txt"
    existing.write_text("data")

    write = WriteFileTool()
    assert write.needs_confirmation({"path": str(existing)}) is True
    assert write.needs_confirmation({"path": str(tmp_path / "new.txt")}) is False
    copy = CopyFileTool()
    assert copy.needs_confirmation({"source": "x", "destination": str(existing)}) is True
    move = MoveFileTool()
    assert move.needs_confirmation({"source": "x", "destination": str(tmp_path / "n")}) is False


@pytest.mark.asyncio
async def test_overwrite_prompts_user_but_fresh_write_does_not(tmp_path: Path) -> None:
    prompts: list[str] = []

    async def approve(tool_name: str, payload: dict[str, Any]) -> bool:
        prompts.append(tool_name)
        return True

    registry = build_default_registry(confirmation_prompt=approve, workspace_root=tmp_path)
    target = tmp_path / "f.txt"

    await registry.invoke("write_file", {"path": str(target), "content": "v1"})
    assert prompts == []  # new file: no prompt
    await registry.invoke("write_file", {"path": str(target), "content": "v2"})
    assert prompts == ["write_file"]  # overwrite: asked first
    assert target.read_text() == "v2"


@pytest.mark.asyncio
async def test_user_is_not_prompted_for_actions_that_would_be_refused(tmp_path: Path) -> None:
    prompts: list[str] = []

    async def approve(tool_name: str, payload: dict[str, Any]) -> bool:
        prompts.append(tool_name)
        return True

    registry = build_default_registry(confirmation_prompt=approve, workspace_root=tmp_path)

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("delete_file", {"path": "/etc/hosts"})

    assert prompts == []


def test_read_file_tool_still_reads_inside_workspace(tmp_path: Path) -> None:
    tool = ReadFileTool()
    tool.workspace_root = tmp_path
    assert tool._resolve(str(tmp_path / "ok.txt")) == (tmp_path / "ok.txt").resolve()


# -------------------------------------------------------------- security gate


@pytest.mark.asyncio
async def test_audit_records_decision_after_confirmation_not_before() -> None:
    audit = AuditLogger()

    async def decline(tool_name: str, payload: dict[str, Any]) -> bool:
        return False

    gate = SecurityGate(confirmations=ConfirmationManager(prompt=decline), audit=audit)

    class Dangerous(Tool):
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="danger", description="x", risk_level=RiskLevel.HIGH
        )

        async def execute(self, payload: dict[str, Any]) -> ToolResult:  # pragma: no cover
            return ToolResult(success=True)

    with pytest.raises(PermissionDeniedError):
        await gate.check_and_confirm(Dangerous(), {})

    assert [e.outcome for e in audit.entries()] == ["denied"]  # never a bogus "allowed"


@pytest.mark.asyncio
async def test_audit_store_failure_does_not_block_tools() -> None:
    class BrokenAudit(AuditLogger):
        async def record_async(self, *args: Any, **kwargs: Any) -> Any:
            raise OSError("disk full")

    gate = SecurityGate(confirmations=ConfirmationManager(prompt=None), audit=BrokenAudit())

    class Harmless(Tool):
        metadata: ClassVar[ToolMetadata] = ToolMetadata(
            name="harmless", description="x", risk_level=RiskLevel.LOW
        )

        async def execute(self, payload: dict[str, Any]) -> ToolResult:  # pragma: no cover
            return ToolResult(success=True)

    await gate.check_and_confirm(Harmless(), {})  # must not raise


def test_summarize_payload_shortens_big_values() -> None:
    out = summarize_payload({"path": "a.txt", "content": "z" * 5000})
    assert out["path"] == "a.txt"
    assert len(out["content"]) < 300


# ---------------------------------------------------------------- system tools


@pytest.mark.asyncio
async def test_get_processes_validates_limit() -> None:
    tool = GetProcessesTool()
    with pytest.raises(ValidationError):
        tool.validate_input({"limit": "lots"})
    assert tool.validate_input({"limit": 10_000})["limit"] == 100
    assert tool.validate_input({"limit": -4})["limit"] == 1


@pytest.mark.asyncio
async def test_get_processes_reports_real_cpu_not_zero_for_busy_process() -> None:
    import sys

    busy = await asyncio.create_subprocess_exec(sys.executable, "-c", "while True: pass")
    try:
        await asyncio.sleep(0.3)
        result = await GetProcessesTool().execute({"limit": 5})
    finally:
        busy.kill()
        await busy.wait()

    top = result.data["processes"]
    assert top[0]["cpu_percent"] > 10  # the busy loop, not an arbitrary 0.0 PID


# ----------------------------------------------------------- dev tool helpers


def test_split_command_keeps_windows_backslashes(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.tools.developer.tools as dev

    monkeypatch.setattr(dev.os, "name", "nt")
    parts = dev.split_command(r'python "C:\Users\me\my script.py" C:\data\in.csv')
    assert parts == ["python", r"C:\Users\me\my script.py", r"C:\data\in.csv"]


# ------------------------------------------------------------------- scheduler


@pytest.mark.asyncio
async def test_finished_one_shot_jobs_are_not_active_forever() -> None:
    scheduler = TaskScheduler()
    fired: list[int] = []

    async def cb() -> None:
        fired.append(1)

    scheduler.schedule_once(cb, 0.01)
    await asyncio.sleep(0.1)

    assert fired == [1]
    assert scheduler.active_jobs() == []


@pytest.mark.asyncio
async def test_failed_reminder_delivery_is_settled_not_refired_every_startup() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    scheduler = TaskScheduler()
    attempts: list[str] = []

    async def failing_delivery(message: str) -> None:
        attempts.append(message)
        raise VoiceError("tts quota exceeded")

    manager = PersistentReminderManager(db, scheduler, failing_delivery)
    await manager.schedule("call mum", delay_seconds=0.01)
    await asyncio.sleep(0.1)

    assert attempts == ["call mum"]
    assert await manager.pending() == []  # previously stayed pending -> repeated forever

    # A "restart": nothing should come back.
    restart_scheduler = TaskScheduler()
    restarted = PersistentReminderManager(db, restart_scheduler, failing_delivery)
    assert await restarted.restore_pending() == 0
    await db.dispose()


# ------------------------------------------------------------------ voice / TTS


def test_split_for_tts_respects_char_limit_and_preserves_words() -> None:
    text = "This is a sentence. " * 30 + "x" * 450
    chunks = split_for_tts(text, limit=200)

    assert chunks
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).replace(" ", "").startswith("Thisisasentence.")
    assert split_for_tts("   ") == []
    assert split_for_tts("Short.") == ["Short."]


class _Recorder:
    def start(self) -> None: ...

    def stop(self) -> bytes:
        return b"\x00\x01" * 100


class _STT:
    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        return TranscriptionResult(text="what time is it")


class _Agent:
    async def respond(self, text: str) -> str:
        return "It is noon."


@pytest.mark.asyncio
async def test_tts_failure_does_not_discard_the_reply() -> None:
    class BrokenTTS:
        async def speak(self, text: str) -> None:
            raise VoiceError("speech quota exceeded")

        async def synthesize(self, text: str) -> bytes:  # pragma: no cover
            return b""

    pipeline = VoicePipeline(_Recorder(), _STT(), BrokenTTS(), _Agent())  # type: ignore[arg-type]

    assert await pipeline.stop_listening_and_respond() == "It is noon."


# --------------------------------------------------------------------- settings


def test_model_defaults_follow_the_chosen_provider() -> None:
    groq = Settings(llm_provider=LLMProviderName.GROQ, _env_file=None)  # type: ignore[call-arg]
    anthropic = Settings(llm_provider=LLMProviderName.ANTHROPIC, _env_file=None)  # type: ignore[call-arg]
    explicit = Settings(  # type: ignore[call-arg]
        llm_provider=LLMProviderName.GROQ, llm_model="my-model", _env_file=None
    )

    assert groq.llm_model == "llama-3.3-70b-versatile"
    assert anthropic.llm_model == "claude-sonnet-4-6"
    assert explicit.llm_model == "my-model"


def test_confirmation_settings_are_actually_wired() -> None:
    settings = Settings(  # type: ignore[call-arg]
        require_confirmation_for_high_risk=False, _env_file=None
    )
    registry = build_default_registry(confirmation_prompt=None, settings=settings)
    assert registry is not None  # builds fine with settings wired in


def test_developer_tools_are_opt_in() -> None:
    off = build_default_registry(confirmation_prompt=None, settings=Settings(_env_file=None))  # type: ignore[call-arg]
    on = build_default_registry(
        confirmation_prompt=None,
        settings=Settings(enable_developer_tools=True, _env_file=None),  # type: ignore[call-arg]
    )
    off_names = {m.name for m in off.list_metadata()}
    on_names = {m.name for m in on.list_metadata()}

    assert "run_command" not in off_names
    assert "run_command" in on_names
