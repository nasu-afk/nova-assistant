from __future__ import annotations

import array

import pytest

from app.core.agent import ConversationAgent
from app.voice.audio_capture import InMemoryRecorder
from app.voice.null_providers import NullTTSProvider
from app.voice.pipeline import VoicePipeline
from app.voice.wakeword import BackgroundListener, EnergyVAD, WakeWordEngine


def _loud_chunk(amplitude: int = 20000, length: int = 800) -> bytes:
    samples = array.array("h", [amplitude if i % 2 == 0 else -amplitude for i in range(length)])
    return samples.tobytes()


def _silent_chunk(length: int = 800) -> bytes:
    return array.array("h", [0] * length).tobytes()


# --- EnergyVAD ---


def test_vad_detects_loud_audio_as_speech() -> None:
    vad = EnergyVAD(threshold=500)
    assert vad.is_speech(_loud_chunk()) is True


def test_vad_treats_silence_as_not_speech() -> None:
    vad = EnergyVAD(threshold=500)
    assert vad.is_speech(_silent_chunk()) is False


def test_vad_treats_empty_bytes_as_not_speech() -> None:
    vad = EnergyVAD()
    assert vad.is_speech(b"") is False


def test_vad_threshold_is_configurable() -> None:
    quiet_vad = EnergyVAD(threshold=100)
    strict_vad = EnergyVAD(threshold=30000)

    chunk = _loud_chunk(amplitude=5000)

    assert quiet_vad.is_speech(chunk) is True
    assert strict_vad.is_speech(chunk) is False


# --- WakeWordEngine (missing dependency path) ---


def test_wake_word_engine_raises_when_dependency_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "openwakeword.model":
            raise ImportError("no module named openwakeword")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from app.core.errors import VoiceError

    with pytest.raises(VoiceError):
        WakeWordEngine()


def _install_fake_openwakeword(  # type: ignore[no-untyped-def]
    monkeypatch: pytest.MonkeyPatch, download_error: Exception | None = None
):
    """Injects fake `openwakeword.model`/`openwakeword.utils` modules into
    sys.modules so WakeWordEngine's validation/download logic can be
    exercised without the real (heavy, unavailable-in-this-sandbox)
    dependency installed."""
    import sys
    import types

    model_module = types.ModuleType("openwakeword.model")

    class FakeModel:
        def __init__(self, wakeword_models: list[str]) -> None:
            self.wakeword_models = wakeword_models

        def predict(self, audio: object) -> dict[str, float]:
            return {}

    model_module.Model = FakeModel  # type: ignore[attr-defined]

    utils_module = types.ModuleType("openwakeword.utils")

    def fake_download_models(model_names: list[str]) -> None:
        if download_error is not None:
            raise download_error

    utils_module.download_models = fake_download_models  # type: ignore[attr-defined]

    package_module = types.ModuleType("openwakeword")
    monkeypatch.setitem(sys.modules, "openwakeword", package_module)
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_module)


def test_wake_word_engine_rejects_unknown_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch)
    from app.core.errors import VoiceError

    with pytest.raises(VoiceError) as exc_info:
        WakeWordEngine(wakeword_model="nova")

    assert "hey_jarvis" in exc_info.value.user_message
    assert "Hey NOVA" in exc_info.value.user_message


def test_wake_word_engine_accepts_pretrained_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch)

    engine = WakeWordEngine(wakeword_model="hey_jarvis")

    assert engine._wakeword_model == "hey_jarvis"


def test_wake_word_engine_wraps_download_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, download_error=RuntimeError("connection reset"))
    from app.core.errors import VoiceError

    with pytest.raises(VoiceError) as exc_info:
        WakeWordEngine(wakeword_model="hey_jarvis")

    assert "internet connection" in exc_info.value.user_message


# --- BackgroundListener ---


class FakeLLM:
    async def complete(  # type: ignore[no-untyped-def]
        self, messages, *, tools=None, max_tokens=1024, temperature=0.7
    ):
        from app.ai.base import LLMResponse

        return LLMResponse(content="Sure thing.")


class FakeSTT:
    async def transcribe(  # type: ignore[no-untyped-def]
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ):
        from app.voice.base import TranscriptionResult

        return TranscriptionResult(text="what's the weather")


@pytest.fixture
def pipeline() -> VoicePipeline:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(_loud_chunk())
    return VoicePipeline(recorder, FakeSTT(), NullTTSProvider(), ConversationAgent(FakeLLM()))


@pytest.mark.asyncio
async def test_run_once_ignores_silence(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
    )

    result = await listener.run_once(_silent_chunk())

    assert result is None


@pytest.mark.asyncio
async def test_run_once_ignores_speech_without_wake_word(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: False,
        pipeline=pipeline,
        command_seconds=0.01,
    )

    result = await listener.run_once(_loud_chunk())

    assert result is None


@pytest.mark.asyncio
async def test_run_once_triggers_full_turn_on_wake_word(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
    )

    result = await listener.run_once(_loud_chunk())

    assert result == "Sure thing."


@pytest.mark.asyncio
async def test_run_processes_chunks_until_source_exhausted(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
    )
    chunks = [_loud_chunk(), _silent_chunk(), None]

    def source() -> bytes | None:
        return chunks.pop(0)

    await listener.run(source)

    assert chunks == []


def test_stop_halts_the_loop_flag(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
    )
    listener._running = True
    listener.stop()
    assert listener._running is False


# --- on_command_complete hook (queue-backlog fix) ---


@pytest.mark.asyncio
async def test_on_command_complete_called_after_successful_turn(pipeline: VoicePipeline) -> None:
    calls = []
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
        on_command_complete=lambda: calls.append(1),
    )

    await listener.run_once(_loud_chunk())

    assert calls == [1]


@pytest.mark.asyncio
async def test_on_command_complete_not_called_without_trigger(pipeline: VoicePipeline) -> None:
    calls = []
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: False,
        pipeline=pipeline,
        command_seconds=0.01,
        on_command_complete=lambda: calls.append(1),
    )

    await listener.run_once(_loud_chunk())

    assert calls == []


@pytest.mark.asyncio
async def test_missing_on_command_complete_is_fine(pipeline: VoicePipeline) -> None:
    listener = BackgroundListener(
        InMemoryRecorder(),
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
        # on_command_complete omitted entirely
    )

    result = await listener.run_once(_loud_chunk())  # should not raise

    assert result == "Sure thing."


# --- resilience to a failed turn (must not crash the always-listening loop) ---


class FailingSTT:
    async def transcribe(self, audio_bytes: bytes, *, sample_rate: int = 16_000):  # type: ignore[no-untyped-def]
        from app.core.errors import ProviderError

        raise ProviderError("network blip", user_message="Couldn't reach the model.")


@pytest.mark.asyncio
async def test_a_failed_turn_does_not_crash_run_once() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(_loud_chunk())
    agent = ConversationAgent(FakeLLM())
    pipeline = VoicePipeline(recorder, FailingSTT(), NullTTSProvider(), agent)
    calls = []
    listener = BackgroundListener(
        recorder,
        EnergyVAD(),
        wake_word=lambda c: True,
        pipeline=pipeline,
        command_seconds=0.01,
        on_command_complete=lambda: calls.append(1),
    )

    result = await listener.run_once(_loud_chunk())  # must not raise

    assert result is None
    assert calls == [1]  # cleanup still happens even when the turn failed


@pytest.mark.asyncio
async def test_run_continues_processing_chunks_after_a_failed_turn() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(_loud_chunk())
    agent = ConversationAgent(FakeLLM())
    pipeline = VoicePipeline(recorder, FailingSTT(), NullTTSProvider(), agent)
    listener = BackgroundListener(
        recorder, EnergyVAD(), wake_word=lambda c: True, pipeline=pipeline, command_seconds=0.01
    )
    chunks = [_loud_chunk(), _loud_chunk(), None]

    def source() -> bytes | None:
        return chunks.pop(0)

    await listener.run(source)  # must not raise, must consume all chunks

    assert chunks == []
