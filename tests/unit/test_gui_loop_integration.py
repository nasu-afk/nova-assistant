"""Drives the real `run_gui_voice_loop()` headlessly (offscreen Qt + qasync).

Only the hardware / network edges are faked: the microphone, the global hotkey,
the LLM stack and STT/TTS. Everything in `app/main.py` and the window is real.
"""

from __future__ import annotations

import array
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import app.main as main_module
import app.ui.main_window as window_module
from app.config.settings import Settings
from app.core.errors import VoiceError
from app.voice.base import TranscriptionResult


def _loud_pcm() -> bytes:
    return array.array("h", [9000 if i % 2 == 0 else -9000 for i in range(1600)]).tobytes()


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def respond(self, text: str) -> str:
        self.calls.append(text)
        return "Done."


class FakeSTT:
    async def transcribe(self, audio: bytes, **_: Any) -> TranscriptionResult:
        return TranscriptionResult(text="hello from voice")


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeRecorder:
    instances: list[FakeRecorder] = []

    def __init__(self) -> None:
        self.closed = False
        FakeRecorder.instances.append(self)

    def start(self) -> None: ...

    def stop(self) -> bytes:
        return _loud_pcm()

    def close(self) -> None:
        self.closed = True


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.agent = FakeAgent()
        self.tts = FakeTTS()
        self.stack_closed = False
        self.window: window_module.MainWindow | None = None
        self.script: list[tuple[int, str]] = []  # (ms, step name)

    def stack(self) -> Any:
        async def close() -> None:
            self.stack_closed = True

        return SimpleNamespace(agent=self.agent, stt=FakeSTT(), tts=self.tts, close=close)


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    if QApplication.instance() is None:
        QApplication([])
    h = _Harness(tmp_path)
    FakeRecorder.instances.clear()

    settings = Settings(  # type: ignore[call-arg]
        _env_file=None,
        llm_api_key="gsk_fake",
        data_dir=tmp_path / "data",
    )
    monkeypatch.setattr(main_module, "get_settings", lambda: settings)
    monkeypatch.setattr(main_module, "configure_logging", lambda *_a, **_k: None)

    async def fake_build_stack(*_args: Any, **_kwargs: Any) -> Any:
        return h.stack()

    monkeypatch.setattr(main_module, "_build_stack", fake_build_stack)
    monkeypatch.setattr("app.voice.audio_capture.MicrophoneRecorder", FakeRecorder)

    class NoHotkey:
        def __init__(self, *_a: Any, **_k: Any) -> None: ...

        def start(self) -> None:
            raise VoiceError("pynput missing", user_message="the hotkey library isn't installed")

        def stop(self) -> None: ...

    monkeypatch.setattr("app.voice.hotkey.PushToTalkListener", NoHotkey)

    real_window = window_module.MainWindow

    class ScriptedWindow(real_window):  # type: ignore[valid-type, misc]
        def __init__(self) -> None:
            super().__init__()
            h.window = self
            for delay, step in h.script:
                QTimer.singleShot(delay, lambda step=step: self._run_step(step))
            QTimer.singleShot(6000, self.close)  # safety net: never hang the suite

        def _run_step(self, step: str) -> None:
            if step == "type":
                self.text_submitted.emit("open notepad", False)
            elif step == "type_spoken":
                self.text_submitted.emit("what time is it", True)
            elif step == "talk_down":
                self._push_to_talk_button.pressed.emit()
            elif step == "talk_up":
                self._push_to_talk_button.released.emit()
            elif step == "close":
                self.close()

    monkeypatch.setattr(window_module, "MainWindow", ScriptedWindow)
    return h


def _log_lines(window: Any) -> list[str]:
    return [window._log_list.item(i).text() for i in range(window._log_list.count())]


def test_typing_and_on_screen_talk_work_without_a_global_hotkey(harness: _Harness) -> None:
    harness.script = [
        (150, "type"),
        (350, "type_spoken"),
        (600, "talk_down"),
        (680, "talk_up"),
        (1400, "close"),
    ]

    main_module.run_gui_voice_loop()

    window = harness.window
    assert window is not None
    # Typed request, typed request with read-aloud, and the on-screen talk button
    # (STT result) all reached the agent, in order.
    assert harness.agent.calls == ["open notepad", "what time is it", "hello from voice"]
    conversation = window._conversation_view.toPlainText()
    assert "You: open notepad" in conversation
    assert "You: hello from voice" in conversation  # voice transcript via the event bus
    # Only the second typed request asked to be read aloud (+ the voice turn's own TTS).
    assert "Done." in harness.tts.spoken
    # A missing hotkey is explained, not fatal.
    assert any("Global hotkey unavailable" in line for line in _log_lines(window))
    # Closing the window ran the full cleanup.
    assert harness.stack_closed is True
    assert FakeRecorder.instances[0].closed is True
    assert window._status_label.text() == "IDLE"


def test_gui_voice_turn_does_not_leave_a_recording_on_disk(harness: _Harness) -> None:
    """The GUI loop used to save every recording unconditionally."""
    harness.script = [(150, "talk_down"), (230, "talk_up"), (1200, "close")]

    main_module.run_gui_voice_loop()

    assert harness.agent.calls == ["hello from voice"]
    assert not (harness.tmp_path / "data" / "debug_last_recording.wav").exists()


def test_without_a_microphone_the_window_is_still_a_working_text_assistant(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_mic() -> None:
        raise VoiceError("no device", user_message="no microphone was found")

    monkeypatch.setattr("app.voice.audio_capture.MicrophoneRecorder", no_mic)
    harness.script = [(150, "type"), (1000, "close")]

    main_module.run_gui_voice_loop()

    window = harness.window
    assert window is not None
    assert harness.agent.calls == ["open notepad"]  # typing works with no mic at all
    assert window._push_to_talk_button.isEnabled() is False
    assert "unavailable" in window._mic_label.text()
    assert "still type" in window._push_to_talk_button.accessibleDescription()
    assert any("typing still works" in line for line in _log_lines(window))
    assert harness.stack_closed is True


def test_a_crashing_agent_does_not_freeze_the_window(harness: _Harness) -> None:
    async def boom(text: str) -> str:
        raise RuntimeError("provider exploded")

    harness.agent.respond = boom  # type: ignore[method-assign]
    harness.script = [(150, "type"), (900, "close")]

    main_module.run_gui_voice_loop()

    window = harness.window
    assert window is not None
    assert window._status_label.text() == "IDLE"  # not stuck on THINKING
    assert any("went wrong" in line for line in _log_lines(window))


def test_event_loop_is_left_usable_for_later_tests() -> None:
    # run_gui_voice_loop installs a qasync loop globally; later async tests must
    # still be able to run.
    async def noop() -> int:
        return 1

    assert asyncio.run(noop()) == 1
