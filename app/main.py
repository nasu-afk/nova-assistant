"""NOVA entrypoint.

Runnable with `python -m app.main` (or `uv run nova`) and usable without a
GUI, per the project's engineering principles. Later phases attach a
PySide6 window on top of the same core; this module must keep working
headlessly even after that.

Four modes:
  `python -m app.main`             - one-shot text sanity check (`run`)
  `python -m app.main --voice`     - push-to-talk voice loop (`run_voice_loop`)
  `python -m app.main --wakeword`  - always-listening voice loop, triggered by
                                      saying the wake word instead of holding a
                                      key (`run_wakeword_loop`)
  `python -m app.main --gui`       - push-to-talk voice loop with a PySide6
                                      window showing live status/transcript and
                                      GUI confirmation dialogs (`run_gui_voice_loop`)
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError as PydanticValidationError

from app.config.settings import get_settings
from app.core.errors import NovaError
from app.core.logging import configure_logging, get_logger

if TYPE_CHECKING:
    from app.ai.base import LLMProvider
    from app.config.settings import Settings
    from app.core.agent import ToolCallingAgent
    from app.database.engine import Database
    from app.events.bus import EventBus
    from app.memory.manager import MemoryManager
    from app.scheduler.scheduler import TaskScheduler
    from app.tools.registry import ToolRegistry
    from app.voice.base import STTProvider, TTSProvider

logger = get_logger(__name__)


async def run() -> None:
    """Start NOVA's core: config, logging, and — if an API key is set — a
    quick live sanity-check conversation via the configured LLM provider.
    """
    settings = get_settings()
    configure_logging(settings)
    settings.ensure_data_dir()

    logger.info(
        "nova_starting",
        app_name=settings.app_name,
        environment=settings.environment,
        llm_provider=settings.llm_provider,
    )

    if settings.llm_api_key is None:
        logger.info(
            "nova_no_llm_key",
            message="No NOVA_LLM_API_KEY set — copy .env.example to .env and add one "
            "(e.g. a free Groq key) to try a live conversation.",
        )
        return

    from app.ai.factory import build_llm_provider
    from app.core.agent import ConversationAgent
    from app.core.errors import NovaError

    llm = build_llm_provider(settings)
    agent = ConversationAgent(llm)
    try:
        reply = await agent.respond("Say hello in one short sentence.")
    except NovaError as exc:
        # Per section 23: explain failures naturally, no raw stack trace to
        # the user — the full technical detail is already in the logs above.
        logger.error("nova_test_reply_failed", error=exc.message)
        print(f"\nNOVA couldn't complete a test reply: {exc.user_message}\n")
        return
    logger.info("nova_test_reply", provider=settings.llm_provider, reply=reply)


class _Stack:
    """Everything both voice modes (`--voice` and `--wakeword`) need,
    built once so the two entrypoints don't duplicate this wiring."""

    def __init__(
        self,
        agent: ToolCallingAgent,
        registry: ToolRegistry,
        stt: STTProvider,
        tts: TTSProvider,
        memory: MemoryManager,
        scheduler: TaskScheduler,
        llm: LLMProvider,
        database: Database,
    ) -> None:
        self.agent = agent
        self.registry = registry
        self.stt = stt
        self.tts = tts
        self.memory = memory
        self.scheduler = scheduler
        self.llm = llm
        self.database = database

    async def close(self) -> None:
        """Release background jobs and persistent resources owned by the stack."""
        self.scheduler.cancel_all()
        # Tools holding external resources (e.g. the YouTube tool's Playwright
        # browser + ad-skip task) expose `aclose()`; they were never closed before.
        iter_tools = getattr(self.registry, "iter_tools", None)
        for tool in iter_tools() if iter_tools else []:
            closer = getattr(tool, "aclose", None)
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - keep shutting down
                logger.error("tool_close_failed", tool=tool.metadata.name, error=str(exc))
        await self.database.dispose()


async def _build_stack(
    settings: Settings,
    *,
    events: EventBus | None = None,
    confirmation_prompt: Any = None,
) -> _Stack:
    """Construct the LLM, memory, scheduler, tool registry, and agent shared
    by `run_voice_loop`, `run_wakeword_loop`, and `run_gui_voice_loop`.

    Memory persists in the same SQLite database `NOVA_DATABASE_URL` points
    at, so facts saved in one run are still there next time you start NOVA.

    Vision (`describe_screen`) reuses the same LLM/model configured for
    chat — this only works if that model actually supports image input.
    Most Groq text models (gpt-oss, qwen, etc.) do NOT; check
    `client.models.list()` for a vision-capable one, or switch
    `NOVA_LLM_PROVIDER` to `anthropic` (Claude models support vision) if
    `describe_screen` fails.

    `events`, if given, is shared with the returned agent so tool activity
    and final replies flow to any subscriber — in practice, a GUI's
    `MainWindow.wire_events()` (see `run_gui_voice_loop`). `confirmation_prompt`
    overrides the default console y/n prompt — a GUI passes
    `MainWindow.show_confirmation_dialog` instead.
    """
    from app.ai.factory import build_llm_provider
    from app.core.agent import ToolCallingAgent
    from app.core.enums import EventType
    from app.database.engine import Database
    from app.events.bus import Event
    from app.memory.conversation import ConversationHistory
    from app.memory.manager import MemoryManager
    from app.scheduler.reminders import PersistentReminderManager
    from app.scheduler.scheduler import TaskScheduler
    from app.security.manager import PersistentAuditLogger
    from app.tools.defaults import build_default_registry, console_confirmation_prompt
    from app.voice.factory import build_stt_provider, build_tts_provider

    llm = build_llm_provider(settings)
    stt = build_stt_provider(settings)
    tts = build_tts_provider(settings)

    db = Database(settings.database_url)
    await db.init()
    memory = MemoryManager(db)

    scheduler = TaskScheduler()

    async def on_reminder_fire(message: str) -> None:
        text = f"Reminder: {message}"
        print(f"\n[reminder] {message}")
        if events is not None:
            # Also surface it in a GUI's conversation view, not just spoken —
            # RESPONSE_READY is exactly the event MainWindow already renders
            # assistant text from (see app/ui/main_window.py). Done before
            # speaking so a TTS failure can't hide the reminder.
            await events.publish(Event(type=EventType.RESPONSE_READY, payload={"text": text}))
        try:
            await tts.speak(text)
        except Exception as exc:  # noqa: BLE001
            logger.error("reminder_speech_failed", error=str(exc))

    reminders = PersistentReminderManager(db, scheduler, on_reminder_fire)
    restored_reminders = await reminders.restore_pending()
    if restored_reminders:
        logger.info("reminders_restored", count=restored_reminders)

    audit = PersistentAuditLogger(db)
    conversation_history = ConversationHistory(db)

    registry = build_default_registry(
        confirmation_prompt or console_confirmation_prompt,
        memory=memory.long_term,
        vision_llm=llm,
        scheduler=scheduler,
        on_reminder_fire=on_reminder_fire,
        reminders=reminders,
        audit=audit,
        settings=settings,
    )
    agent = ToolCallingAgent(
        llm,
        registry,
        memory=memory.long_term,
        conversation_history=conversation_history,
        events=events,
    )

    return _Stack(agent, registry, stt, tts, memory, scheduler, llm, db)


class _NoVoice(Exception):
    """Internal control-flow marker: voice input isn't available in this run."""


def _handle_captured_audio(
    settings: Any,
    audio_bytes: bytes,
    debug_wav_path: Any,
    quiet_threshold: float,
) -> tuple[float, bool]:
    """Shared by the console and GUI voice loops: log the recording's level and
    keep a copy ONLY if the user opted in (`NOVA_DEBUG_SAVE_AUDIO`) or it was so
    quiet that the "check your microphone" hint needs something to play back.
    Returns (rms_level, is_quiet)."""
    from app.voice.debug import rms_energy, save_debug_wav

    energy = rms_energy(audio_bytes)
    logger.info("voice_audio_captured", bytes=len(audio_bytes), rms=round(energy, 1))
    is_quiet = energy < quiet_threshold
    if settings.debug_save_audio or is_quiet:
        save_debug_wav(debug_wav_path, audio_bytes)
    return energy, is_quiet


async def run_voice_loop() -> None:
    """Full push-to-talk voice loop: hold the configured hotkey, speak,
    release — NOVA transcribes, thinks (with tool access and persistent
    memory), and replies out loud. Runs until Ctrl+C.

    Needs: an LLM API key, a working microphone/speaker (`MicrophoneRecorder`,
    the configured `TTSProvider`), and `pynput` with a live OS input backend
    for the global hotkey (see `app.voice.hotkey`) — none of which this
    project's own sandbox has, so this path is exercised manually on a real
    machine rather than in the test suite.
    """
    settings = get_settings()
    configure_logging(settings)
    settings.ensure_data_dir()

    logger.info(
        "nova_starting",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        voice_provider=settings.voice_provider,
    )

    if settings.llm_api_key is None:
        print(
            "\nNo NOVA_LLM_API_KEY set — copy .env.example to .env and add one "
            "(e.g. a free Groq key) first.\n"
        )
        return

    from app.core.errors import NovaError
    from app.voice.audio_capture import MicrophoneRecorder
    from app.voice.hotkey import PushToTalkListener
    from app.voice.pipeline import VoicePipeline

    stack = await _build_stack(settings)

    try:
        recorder = MicrophoneRecorder()
    except NovaError as exc:
        logger.error("nova_mic_unavailable", error=exc.message)
        print(f"\nMicrophone isn't available: {exc.user_message}\n")
        await stack.close()
        return

    # Below this RMS, the recording is almost certainly silence — the
    # classic cause of Whisper hallucinating a stock phrase ("Thank you.")
    # instead of actually transcribing nothing. Surfacing this immediately
    # saves a very confusing debugging loop where it *looks* like NOVA
    # ignores what you said.
    quiet_threshold = 150.0
    debug_wav_path = settings.data_dir / "debug_last_recording.wav"

    def on_audio_captured(audio_bytes: bytes) -> None:
        energy, is_quiet = _handle_captured_audio(
            settings, audio_bytes, debug_wav_path, quiet_threshold
        )
        if is_quiet:
            print(
                f"[warning] That recording was very quiet (level {energy:.0f}) — "
                f"likely a microphone/input-device issue, not a NOVA bug. Play back "
                f"{debug_wav_path} to check whether your voice was actually captured; "
                "if it's silent, check Windows Sound settings > Input for the right "
                "default device and volume."
            )

    pipeline = VoicePipeline(
        recorder, stack.stt, stack.tts, stack.agent, on_audio_captured=on_audio_captured
    )
    loop = asyncio.get_running_loop()

    async def handle_turn() -> None:
        try:
            reply = await pipeline.stop_listening_and_respond()
        except NovaError as exc:
            logger.error("nova_turn_failed", error=exc.message)
            print(f"NOVA: (error) {exc.user_message}")
            return
        except Exception as exc:  # noqa: BLE001 - never leave the user in silence
            logger.error("nova_turn_crashed", error=repr(exc))
            print("NOVA: (error) Something went wrong on my end. Please try again.")
            return
        print(f"NOVA: {reply}" if reply else "(didn't catch that)")

    def on_start() -> None:
        print("\n[listening...]")
        try:
            pipeline.start_listening()
        except NovaError as exc:
            logger.error("nova_listen_failed", error=exc.message)
            print(f"(error) {exc.user_message}")

    def on_stop() -> None:
        print("[thinking...]")
        asyncio.run_coroutine_threadsafe(handle_turn(), loop)

    try:
        listener = PushToTalkListener(settings.push_to_talk_key, on_start=on_start, on_stop=on_stop)
        listener.start()
    except NovaError as exc:
        logger.error("nova_hotkey_unavailable", error=exc.message)
        print(f"\nPush-to-talk isn't available: {exc.user_message}\n")
        recorder.close()
        await stack.close()
        return

    print(f"NOVA is ready. Hold '{settings.push_to_talk_key}' to talk. Ctrl+C to quit.")
    try:
        await asyncio.Event().wait()  # block until Ctrl+C
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        recorder.close()
        await stack.close()
        print("\nNOVA stopped.")


async def run_wakeword_loop() -> None:
    """Always-listening voice loop: say the wake word instead of holding a
    key. Runs until Ctrl+C.

    Needs everything `run_voice_loop` needs, plus `openwakeword` (see the
    comment in pyproject.toml — install directly with
    `uv pip install openwakeword`, it isn't a declared project extra) and a
    trained wake-word model file. None of this is testable in this
    project's own sandbox (no mic, no wake-word model installed), so this
    path is exercised manually on a real machine.

    Note: this opens two separate audio streams concurrently — one for
    continuous wake-word monitoring (`ContinuousMicrophoneChunkSource`) and
    one for recording the actual command once triggered (`MicrophoneRecorder`,
    reused from the `VoicePipeline`). This should work fine under Windows'
    typical shared-mode audio (multiple apps/streams reading the same input
    device concurrently is the normal case), but hasn't been confirmed on
    real hardware — if you hit mic contention or odd audio behavior in this
    mode specifically, that dual-stream design is the first thing to
    suspect.
    """
    settings = get_settings()
    configure_logging(settings)
    settings.ensure_data_dir()

    logger.info(
        "nova_starting",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        mode="wakeword",
    )

    if settings.llm_api_key is None:
        print(
            "\nNo NOVA_LLM_API_KEY set — copy .env.example to .env and add one "
            "(e.g. a free Groq key) first.\n"
        )
        return

    from app.core.errors import NovaError
    from app.voice.audio_capture import ContinuousMicrophoneChunkSource, MicrophoneRecorder
    from app.voice.pipeline import VoicePipeline
    from app.voice.wakeword import BackgroundListener, EnergyVAD, WakeWordEngine

    stack = await _build_stack(settings)

    recorder: MicrophoneRecorder | None = None
    chunk_source: ContinuousMicrophoneChunkSource | None = None
    try:
        recorder = MicrophoneRecorder()
        chunk_source = ContinuousMicrophoneChunkSource(chunk_seconds=1.0)
    except NovaError as exc:
        logger.error("nova_mic_unavailable", error=exc.message)
        print(f"\nMicrophone isn't available: {exc.user_message}\n")
        if recorder is not None:
            recorder.close()
        if chunk_source is not None:
            chunk_source.stop()
        await stack.close()
        return

    try:
        wake_word_engine = WakeWordEngine(wakeword_model=settings.wake_word_model)
    except NovaError as exc:
        logger.error("nova_wakeword_unavailable", error=exc.message)
        print(
            f"\nWake-word detection isn't available: {exc.user_message}\n"
            "Install it with `uv pip install openwakeword` (Windows/macOS only — "
            "see the comment in pyproject.toml for why it isn't a project extra).\n"
        )
        chunk_source.stop()
        recorder.close()
        await stack.close()
        return

    pipeline = VoicePipeline(recorder, stack.stt, stack.tts, stack.agent)
    listener = BackgroundListener(
        recorder,
        EnergyVAD(threshold=settings.wake_vad_threshold),
        wake_word=wake_word_engine.is_wake_word,
        pipeline=pipeline,
        on_command_complete=chunk_source.drain,
    )

    chunk_source.start()
    from pathlib import Path

    trigger_phrase = Path(settings.wake_word_model).stem.replace("_", " ")
    print(f"NOVA is listening for the wake word — say '{trigger_phrase}'. Ctrl+C to quit.")
    try:
        await listener.run(chunk_source.get_chunk)
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        chunk_source.stop()
        recorder.close()
        await stack.close()
        print("\nNOVA stopped.")


async def _stay_open_after_setup_failure(
    window: Any, log_message: str, status: str, closed: asyncio.Event | None = None
) -> None:
    """Log an error to the window and then block forever (until Ctrl+C or
    the window is closed) instead of returning immediately.

    Exists because returning from `run_inside_gui()` (see `run_gui_voice_loop`)
    ends `loop.run_until_complete()` on the spot, which tears down the whole
    qasync loop and closes the window before Qt ever gets a chance to paint
    the log line just appended — the error would never actually be visible,
    just a window that flashes and vanishes. This was a real bug, not a
    hypothetical one.
    """
    window.append_log(log_message)
    window.set_status(status)
    # `closed` is set when the user closes the window; without it, wait forever.
    await (closed or asyncio.Event()).wait()


def run_gui_voice_loop() -> None:
    """Push-to-talk voice loop with a PySide6 window: live status
    (listening/thinking/speaking), a transcript, tool activity, and GUI
    confirmation dialogs for risky actions — instead of terminal prints and
    a console y/n prompt.

    Not `async def` like the other run_* functions: PySide6's event loop and
    asyncio's event loop are two different things that both want to own the
    thread, so this bridges them with `qasync` and manages the loop itself
    rather than being handed one by `asyncio.run()`. Needs the `gui` extra
    (`uv sync --extra gui`) for `qasync`, on top of everything `run_voice_loop`
    needs. Untestable end-to-end in this project's own sandbox for the same
    reasons `run_voice_loop` is (no mic, no display server) — this path is
    exercised manually on a real machine.
    """
    settings = get_settings()
    configure_logging(settings)
    settings.ensure_data_dir()

    logger.info(
        "nova_starting",
        app_name=settings.app_name,
        llm_provider=settings.llm_provider,
        voice_provider=settings.voice_provider,
        mode="gui",
    )

    if settings.llm_api_key is None:
        print(
            "\nNo NOVA_LLM_API_KEY set — copy .env.example to .env and add one "
            "(e.g. a free Groq key) first.\n"
        )
        return

    try:
        import qasync
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "\nThe GUI needs the 'gui' extra: `uv sync --extra gui` "
            "(installs PySide6 and qasync).\n"
        )
        return

    from app.core.errors import NovaError
    from app.events.bus import EventBus
    from app.ui.main_window import MainWindow
    from app.voice.audio_capture import MicrophoneRecorder
    from app.voice.hotkey import PushToTalkListener
    from app.voice.pipeline import VoicePipeline

    application = QApplication.instance() or QApplication(sys.argv)
    # Closing the window must not stop Qt's loop out from under our coroutines:
    # we signal `closed`, let run_inside_gui() clean up (hotkey listener, mic,
    # DB, browser) and return, and only then does the loop end.
    QApplication.setQuitOnLastWindowClosed(False)
    window = MainWindow()
    closed = asyncio.Event()
    window.closed.connect(closed.set)
    events = EventBus()
    window.wire_events(events)

    async def gui_confirmation_prompt(tool_name: str, payload: dict[str, Any]) -> bool:
        # window.show_confirmation_dialog is a normal blocking QMessageBox
        # call — safe here because qasync runs Qt's and asyncio's event
        # loops on the same thread, and QMessageBox.exec() natively pumps
        # Qt's event loop while it waits, exactly like in a plain Qt app.
        return window.show_confirmation_dialog(tool_name, payload)

    loop = qasync.QEventLoop(application)
    asyncio.set_event_loop(loop)

    async def run_inside_gui() -> None:
        try:
            stack = await _build_stack(
                settings, events=events, confirmation_prompt=gui_confirmation_prompt
            )
        except Exception as exc:  # noqa: BLE001 - show it in the window, don't just vanish
            logger.error("nova_startup_failed", error=repr(exc))
            await _stay_open_after_setup_failure(
                window, f"Startup failed: {exc}", "Startup failed", closed
            )
            return

        recorder: MicrophoneRecorder | None
        try:
            recorder = MicrophoneRecorder()
        except NovaError as exc:
            # Not fatal: without a microphone the window is still a fully working
            # text assistant (typed requests), just with voice switched off.
            logger.error("nova_mic_unavailable", error=exc.message)
            recorder = None
            window.set_voice_available(False, exc.user_message)
            window.append_log(
                f"Microphone isn't available: {exc.user_message} — typing still works."
            )

        quiet_threshold = 150.0
        debug_wav_path = settings.data_dir / "debug_last_recording.wav"

        def on_audio_captured(audio_bytes: bytes) -> None:
            energy, is_quiet = _handle_captured_audio(
                settings, audio_bytes, debug_wav_path, quiet_threshold
            )
            window.append_log(f"Captured {len(audio_bytes)} bytes (level {energy:.0f})")
            if is_quiet:
                window.append_log(
                    f"Very quiet recording (level {energy:.0f}) — check your microphone."
                )

        pipeline: VoicePipeline | None = (
            VoicePipeline(
                recorder,
                stack.stt,
                stack.tts,
                stack.agent,
                on_audio_captured=on_audio_captured,
                events=events,
            )
            if recorder is not None
            else None
        )

        turn_loop = asyncio.get_running_loop()

        async def handle_turn() -> None:
            if pipeline is None:
                return
            try:
                await pipeline.stop_listening_and_respond()
            except NovaError as exc:
                logger.error("nova_turn_failed", error=exc.message)
                window.append_log(f"Error: {exc.user_message}")
            except Exception as exc:  # noqa: BLE001 - otherwise the status sticks on "Thinking"
                logger.error("nova_turn_crashed", error=repr(exc))
                window.append_log("Error: something went wrong on my end. Please try again.")
            finally:
                # Every outcome ends at Idle. Previously only the error paths did,
                # so after a normal reply the window stayed on "SPEAKING" (and
                # after a silent press on "THINKING") until the next turn.
                window.set_status("Idle")

        def on_start() -> None:
            if pipeline is None:
                return
            try:
                pipeline.start_listening()
            except NovaError as exc:
                logger.error("nova_listen_failed", error=exc.message)
                window.append_log(f"Error: {exc.user_message}")
                return
            asyncio.run_coroutine_threadsafe(pipeline.publish_started(), turn_loop)

        def on_stop() -> None:
            if pipeline is None:
                return
            asyncio.run_coroutine_threadsafe(handle_turn(), turn_loop)

        async def handle_typed(text: str, speak: bool) -> None:
            """A typed request: same agent, text reply (spoken only if asked)."""
            window.append_conversation("You", text)
            window.set_status("Thinking")
            try:
                reply = await stack.agent.respond(text)
                if speak and reply:
                    try:
                        await stack.tts.speak(reply)
                    except Exception as exc:  # noqa: BLE001 - the text reply is already shown
                        logger.error("typed_reply_speech_failed", error=str(exc))
            except NovaError as exc:
                logger.error("nova_typed_turn_failed", error=exc.message)
                window.append_log(f"Error: {exc.user_message}")
            except Exception as exc:  # noqa: BLE001
                logger.error("nova_typed_turn_crashed", error=repr(exc))
                window.append_log("Error: something went wrong on my end. Please try again.")
            finally:
                window.set_status("Idle")

        # Keep a reference to in-flight typed turns: asyncio only holds weak
        # references to tasks, so a fire-and-forget task can be collected mid-run.
        typed_tasks: set[asyncio.Task[None]] = set()

        def on_text_submitted(text: str, speak: bool) -> None:
            task = asyncio.ensure_future(handle_typed(text, speak))
            typed_tasks.add(task)
            task.add_done_callback(typed_tasks.discard)

        # The on-screen controls work no matter what the global hotkey does:
        # the talk button was previously a dead control (never connected), and
        # there was no way to type a request at all.
        window.talk_pressed.connect(on_start)
        window.talk_released.connect(on_stop)
        window.text_submitted.connect(on_text_submitted)

        hotkey_listener: PushToTalkListener | None = None
        try:
            if recorder is None:
                raise _NoVoice
            hotkey_listener = PushToTalkListener(
                settings.push_to_talk_key, on_start=on_start, on_stop=on_stop
            )
            hotkey_listener.start()
        except _NoVoice:
            pass  # already explained above; nothing to bind a hotkey to
        except NovaError as exc:
            # Not fatal any more: the button and the text box still work.
            logger.error("nova_hotkey_unavailable", error=exc.message)
            hotkey_listener = None
            window.append_log(
                f"Global hotkey unavailable ({exc.user_message}) — "
                "use the on-screen button or type your request."
            )

        window.set_status("Idle")
        if recorder is None:
            window.append_log("Ready — type a request below.")
        elif hotkey_listener is not None:
            window.append_log(
                f"Ready — hold '{settings.push_to_talk_key}' or the on-screen button to talk, "
                "or type a request."
            )
        try:
            await closed.wait()
        finally:
            if hotkey_listener is not None:
                hotkey_listener.stop()
            if recorder is not None:
                recorder.close()
            await stack.close()

    window.show()
    with loop:
        loop.run_until_complete(run_inside_gui())


def main() -> None:
    """Synchronous entrypoint for console scripts / `python -m app.main`."""
    try:
        if "--gui" in sys.argv:
            run_gui_voice_loop()
        elif "--wakeword" in sys.argv:
            asyncio.run(run_wakeword_loop())
        elif "--voice" in sys.argv:
            asyncio.run(run_voice_loop())
        else:
            asyncio.run(run())
    except KeyboardInterrupt:
        # asyncio.run() re-raises Ctrl+C after cancelling the main task (cleanup
        # in the loops' `finally` blocks has already run); exit quietly.
        print("\nBye.")
    except NovaError as exc:
        # Misconfiguration (missing key, bad provider ...) deserves one clear
        # line, not a stack trace.
        print(f"\nNOVA couldn't start: {exc.user_message}\n")
        raise SystemExit(1) from None
    except PydanticValidationError as exc:
        print(f"\nNOVA couldn't start: invalid settings in .env / environment:\n{exc}\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
