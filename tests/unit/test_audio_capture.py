from __future__ import annotations

import builtins

import pytest

from app.core.errors import VoiceError
from app.voice.audio_capture import MicrophoneRecorder


def test_missing_sounddevice_package_raises_voice_error(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise ImportError("no module named sounddevice")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError):
        MicrophoneRecorder()


def test_missing_native_portaudio_library_raises_voice_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # sounddevice itself can import fine as a Python package but raise
    # OSError at import time if libportaudio isn't installed on the system —
    # a real failure mode this sandbox hits directly (no audio hardware).
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError) as exc_info:
        MicrophoneRecorder()

    assert "PortAudio" in exc_info.value.user_message


from app.voice.audio_capture import ContinuousMicrophoneChunkSource  # noqa: E402


def test_continuous_chunk_source_missing_sounddevice_raises_voice_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise ImportError("no module named sounddevice")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError):
        ContinuousMicrophoneChunkSource()


def test_continuous_chunk_source_missing_portaudio_raises_voice_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sounddevice":
            raise OSError("PortAudio library not found")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError) as exc_info:
        ContinuousMicrophoneChunkSource()

    assert "PortAudio" in exc_info.value.user_message


# --- behavior tests using a fake sounddevice module ---


class FakeRawInputStream:
    instances: list[FakeRawInputStream] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        FakeRawInputStream.instances.append(self)

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True

    def feed(self, data: bytes) -> None:
        """Simulate the audio driver delivering a block of samples."""
        callback = self.kwargs["callback"]
        callback(data, len(data) // 2, None, None)  # type: ignore[operator]


@pytest.fixture
def fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> type[FakeRawInputStream]:
    import sys
    import types

    FakeRawInputStream.instances = []
    module = types.ModuleType("sounddevice")
    module.RawInputStream = FakeRawInputStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return FakeRawInputStream


def test_recorder_captures_only_while_recording(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    recorder = MicrophoneRecorder()
    recorder.start()
    stream = fake_sounddevice.instances[0]

    stream.feed(b"\x01\x02" * 4)
    audio = recorder.stop()
    stream.feed(b"\x09\x09" * 4)  # arrives after stop() -> must be ignored

    assert audio == b"\x01\x02" * 4


def test_recorder_reuses_one_stream_across_turns(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    recorder = MicrophoneRecorder()

    recorder.start()
    recorder.stop()
    recorder.start()
    recorder.stop()

    # The whole point of the persistent-stream design: only one stream is
    # ever opened, so only the first press pays hardware startup latency.
    assert len(fake_sounddevice.instances) == 1
    assert fake_sounddevice.instances[0].closed is False


def test_recorder_start_clears_previous_recording(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    recorder = MicrophoneRecorder()
    recorder.start()
    fake_sounddevice.instances[0].feed(b"\x01\x01" * 4)
    recorder.stop()

    recorder.start()
    fake_sounddevice.instances[0].feed(b"\x02\x02" * 4)

    assert recorder.stop() == b"\x02\x02" * 4


def test_recorder_close_releases_the_stream(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    recorder = MicrophoneRecorder()
    recorder.start()
    stream = fake_sounddevice.instances[0]

    recorder.close()

    assert stream.stopped is True
    assert stream.closed is True


def test_recorder_reopens_stream_after_close(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    recorder = MicrophoneRecorder()
    recorder.start()
    recorder.close()

    recorder.start()

    assert len(fake_sounddevice.instances) == 2


def test_recorder_close_without_start_is_a_no_op(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    MicrophoneRecorder().close()  # should not raise


def test_chunk_source_delivers_chunks_in_order(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    source = ContinuousMicrophoneChunkSource(chunk_seconds=1.0)
    source.start()
    stream = fake_sounddevice.instances[0]

    stream.feed(b"\x01\x00" * 4)
    stream.feed(b"\x02\x00" * 4)

    assert source.get_chunk() == b"\x01\x00" * 4
    assert source.get_chunk() == b"\x02\x00" * 4


def test_chunk_source_requests_blocksize_matching_chunk_seconds(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    source = ContinuousMicrophoneChunkSource(chunk_seconds=0.5, sample_rate=16_000)
    source.start()

    assert fake_sounddevice.instances[0].kwargs["blocksize"] == 8_000


def test_chunk_source_stop_unblocks_and_returns_none(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    source = ContinuousMicrophoneChunkSource()
    source.start()
    stream = fake_sounddevice.instances[0]

    source.stop()

    assert stream.closed is True
    assert source.get_chunk() is None


def test_drain_discards_queued_chunks(fake_sounddevice: type[FakeRawInputStream]) -> None:
    source = ContinuousMicrophoneChunkSource(chunk_seconds=1.0)
    source.start()
    stream = fake_sounddevice.instances[0]
    stream.feed(b"\x01\x00" * 4)
    stream.feed(b"\x02\x00" * 4)
    stream.feed(b"\x03\x00" * 4)

    source.drain()

    # New audio arriving after drain is delivered normally — the queue
    # wasn't broken, just emptied of the stale backlog.
    stream.feed(b"\x04\x00" * 4)
    assert source.get_chunk() == b"\x04\x00" * 4


def test_drain_on_empty_queue_does_not_block_or_raise(
    fake_sounddevice: type[FakeRawInputStream],
) -> None:
    source = ContinuousMicrophoneChunkSource()
    source.start()

    source.drain()  # should return immediately, not hang
