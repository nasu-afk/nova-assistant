"""Debug-audio gating (console AND GUI loops) and the wake-word changes."""

from __future__ import annotations

import array
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app.config.settings import Settings
from app.core.errors import VoiceError
from app.main import _handle_captured_audio
from app.voice.wakeword import BackgroundListener, EnergyVAD, WakeWordEngine


def _pcm(amplitude: int, length: int = 1600) -> bytes:
    return array.array(
        "h", [amplitude if i % 2 == 0 else -amplitude for i in range(length)]
    ).tobytes()


# ------------------------------------------------------------------ debug audio


def test_normal_recording_is_not_written_to_disk(tmp_path: Path) -> None:
    wav = tmp_path / "debug.wav"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    energy, quiet = _handle_captured_audio(settings, _pcm(8000), wav, 150.0)

    assert energy > 150 and quiet is False
    assert not wav.exists()  # the GUI loop used to save this unconditionally


def test_very_quiet_recording_is_kept_so_the_mic_hint_works(tmp_path: Path) -> None:
    wav = tmp_path / "debug.wav"
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    _, quiet = _handle_captured_audio(settings, _pcm(5), wav, 150.0)

    assert quiet is True and wav.exists()


def test_recording_is_kept_when_the_user_opts_in(tmp_path: Path) -> None:
    wav = tmp_path / "debug.wav"
    settings = Settings(debug_save_audio=True, _env_file=None)  # type: ignore[call-arg]

    _handle_captured_audio(settings, _pcm(8000), wav, 150.0)

    assert wav.exists()


def test_both_voice_loops_use_the_shared_privacy_helper() -> None:
    """Guards against a loop saving audio on its own again."""
    source = (Path(__file__).parents[2] / "app" / "main.py").read_text()
    assert source.count("_handle_captured_audio(") == 3  # definition + console + GUI
    assert "save_debug_wav(" not in source.replace("        save_debug_wav(debug_wav_path", "")


# --------------------------------------------------------------------- wake word


class _Recorder:
    def start(self) -> None: ...

    def stop(self) -> bytes:
        return b""


class _Pipeline:
    def __init__(self) -> None:
        self.started = 0

    def start_listening(self) -> None:
        self.started += 1

    async def stop_listening_and_respond(self) -> str:
        return "ok"


def _listener(hits: list[bool], pipeline: _Pipeline, seen: list[bytes]) -> BackgroundListener:
    queue = list(hits)

    def wake_word(chunk: bytes) -> bool:
        seen.append(chunk)
        return queue.pop(0)

    return BackgroundListener(
        _Recorder(),  # type: ignore[arg-type]
        EnergyVAD(threshold=500),
        wake_word=wake_word,
        pipeline=pipeline,  # type: ignore[arg-type]
        command_seconds=0.0,
    )


@pytest.mark.asyncio
async def test_every_chunk_is_fed_to_the_model_even_when_quiet() -> None:
    """Skipping quiet chunks cut phrases in half (the model keeps rolling state)."""
    seen: list[bytes] = []
    listener = _listener([False, False, False], _Pipeline(), seen)

    for chunk in (_pcm(0), _pcm(10), _pcm(9000)):
        await listener.run_once(chunk)

    assert len(seen) == 3


@pytest.mark.asyncio
async def test_hit_in_near_silence_is_rejected() -> None:
    pipeline = _Pipeline()
    listener = _listener([True], pipeline, [])

    assert await listener.run_once(_pcm(0)) is None
    assert pipeline.started == 0


@pytest.mark.asyncio
async def test_hit_just_after_speech_is_accepted() -> None:
    """The model often fires on the chunk right after the phrase ended."""
    pipeline = _Pipeline()
    listener = _listener([False, True], pipeline, [])

    await listener.run_once(_pcm(9000))  # speech, no detection yet
    result = await listener.run_once(_pcm(0))  # quiet tail, model fires now

    assert result == "ok" and pipeline.started == 1


def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls: dict[str, Any] = {"downloaded": None, "model_kwargs": None}
    model_module = types.ModuleType("openwakeword.model")

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            calls["model_kwargs"] = kwargs

        def predict(self, audio: object) -> dict[str, float]:
            return {}

    model_module.Model = FakeModel  # type: ignore[attr-defined]
    utils_module = types.ModuleType("openwakeword.utils")

    def fake_download(model_names: list[str]) -> None:
        calls["downloaded"] = model_names

    utils_module.download_models = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
    monkeypatch.setitem(sys.modules, "openwakeword.model", model_module)
    monkeypatch.setitem(sys.modules, "openwakeword.utils", utils_module)
    return calls


@pytest.mark.parametrize(
    ("filename", "framework"), [("hey_nova.onnx", "onnx"), ("hey_nova.tflite", "tflite")]
)
def test_custom_trained_model_file_is_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, filename: str, framework: str
) -> None:
    calls = _install_fake_openwakeword(monkeypatch)
    model_file = tmp_path / filename
    model_file.write_bytes(b"fake model")

    engine = WakeWordEngine(wakeword_model=str(model_file))

    assert calls["model_kwargs"] == {
        "wakeword_models": [str(model_file)],
        "inference_framework": framework,
    }
    assert engine._wakeword_model == str(model_file)


def test_missing_custom_model_file_is_a_clear_error_not_a_silent_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_openwakeword(monkeypatch)

    with pytest.raises(VoiceError) as excinfo:
        WakeWordEngine(wakeword_model=str(tmp_path / "hey_nova.onnx"))

    assert "couldn't find" in excinfo.value.user_message.lower()


def test_pretrained_models_still_load_the_old_way(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_openwakeword(monkeypatch)

    WakeWordEngine(wakeword_model="hey_jarvis")

    assert calls["downloaded"] == ["hey_jarvis"]
    assert calls["model_kwargs"] == {"wakeword_models": ["hey_jarvis"]}


def test_hey_nova_error_points_at_the_custom_model_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch)

    with pytest.raises(VoiceError) as excinfo:
        WakeWordEngine(wakeword_model="hey_nova")

    assert ".onnx" in excinfo.value.user_message
