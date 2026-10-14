"""Microphone capture for push-to-talk recording.

`sounddevice` (PortAudio bindings) is imported lazily — this module is safe
to import on machines/containers with no audio hardware; only constructing
`MicrophoneRecorder` requires a working audio backend.
"""

from __future__ import annotations

import queue
from typing import Any, Protocol

from app.core.errors import VoiceError

SAMPLE_RATE = 16_000
CHANNELS = 1


class AudioRecorder(Protocol):
    """Interface the voice pipeline depends on — not tied to sounddevice."""

    def start(self) -> None:
        """Begin recording."""
        ...

    def stop(self) -> bytes:
        """Stop recording and return the captured audio as PCM16 mono bytes."""
        ...


class MicrophoneRecorder:
    """AudioRecorder backed by `sounddevice`, for push-to-talk capture.

    Keeps one audio stream open persistently across turns rather than
    opening a fresh one on every `start()`/closing it on every `stop()`.
    Opening a `RawInputStream` has real hardware/driver startup latency
    (tens to hundreds of milliseconds depending on the system) — doing that
    on every single press means the first fraction of a second of speech
    can be silently dropped before the stream is actually capturing, which
    is a very plausible cause of intermittent Whisper hallucinations
    ("Thank you.") on short commands that isn't a microphone hardware
    problem at all. With a persistent stream, only the very first `start()`
    of the whole session pays that latency; every subsequent press just
    flips a flag the callback already checks. Call `close()` to fully
    release the audio device at shutdown — `start()`/`stop()` never do.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'sounddevice' package is not installed (install the "
                "'voice' extra: `uv sync --extra voice`).",
                user_message="Microphone input isn't available yet.",
            ) from exc
        except OSError as exc:
            # sounddevice imports fine but fails at import time if the native
            # PortAudio library isn't present on the system (distinct from
            # the package not being pip-installed at all).
            raise VoiceError(
                f"sounddevice's native PortAudio library is missing: {exc}",
                user_message=(
                    "Microphone input isn't available — the PortAudio library "
                    "isn't installed on this system."
                ),
            ) from exc

        self._sd = sd
        self._sample_rate = sample_rate
        self._stream: Any | None = None
        self._frames: list[bytes] = []
        self._recording = False

    def _callback(self, indata: Any, frames: int, time_info: object, status: object) -> None:
        if self._recording:
            self._frames.append(bytes(indata))

    def _ensure_stream_open(self) -> None:
        if self._stream is not None:
            return
        stream = self._sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype="int16",
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def start(self) -> None:
        self._ensure_stream_open()
        self._frames = []
        self._recording = True

    def stop(self) -> bytes:
        self._recording = False
        return b"".join(self._frames)

    def close(self) -> None:
        """Fully release the audio device. Call once at shutdown, not
        between turns — `start()`/`stop()` reuse the same open stream."""
        self._recording = False
        if self._stream is not None:
            stream: Any = self._stream
            stream.stop()
            stream.close()
            self._stream = None


class InMemoryRecorder:
    """A fake `AudioRecorder` for tests and headless demos.

    Pre-load it with `set_next_recording(bytes)` before `start()`/`stop()`.
    """

    def __init__(self) -> None:
        self._next: bytes = b""

    def set_next_recording(self, audio_bytes: bytes) -> None:
        self._next = audio_bytes

    def start(self) -> None:
        return None

    def stop(self) -> bytes:
        return self._next


class ContinuousMicrophoneChunkSource:
    """Continuously captures fixed-size audio chunks for wake-word monitoring.

    Distinct from `MicrophoneRecorder` (which records start-to-stop, for
    push-to-talk): this reads the mic non-stop in the background and hands
    off one chunk at a time via `get_chunk()`, matching the
    `Callable[[], bytes | None]` shape `app.voice.wakeword.BackgroundListener.run`
    expects. `get_chunk()` blocks the calling thread until a chunk is ready
    (or `stop()` is called) — callers on the asyncio event loop must run it
    via `asyncio.to_thread`, exactly as `BackgroundListener.run` already does.
    """

    def __init__(self, chunk_seconds: float = 1.0, sample_rate: int = SAMPLE_RATE) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'sounddevice' package is not installed (install the "
                "'voice' extra: `uv sync --extra voice`).",
                user_message="Microphone input isn't available yet.",
            ) from exc
        except OSError as exc:
            raise VoiceError(
                f"sounddevice's native PortAudio library is missing: {exc}",
                user_message=(
                    "Microphone input isn't available — the PortAudio library "
                    "isn't installed on this system."
                ),
            ) from exc

        self._sd = sd
        self._sample_rate = sample_rate
        self._chunk_frames = max(1, int(chunk_seconds * sample_rate))
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._stream: Any | None = None
        self._stopped = False

    def _callback(self, indata: Any, frames: int, time_info: object, status: object) -> None:
        self._queue.put(bytes(indata))

    def start(self) -> None:
        self._stopped = False
        stream = self._sd.RawInputStream(
            samplerate=self._sample_rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=self._chunk_frames,
            callback=self._callback,
        )
        stream.start()
        self._stream = stream

    def stop(self) -> None:
        self._stopped = True
        if self._stream is not None:
            stream: Any = self._stream
            stream.stop()
            stream.close()
            self._stream = None
        # Unblock a caller currently waiting in get_chunk().
        self._queue.put(b"")

    def get_chunk(self) -> bytes | None:
        """Blocks until one chunk is available, or returns None after `stop()`."""
        if self._stopped:
            return None
        chunk = self._queue.get()
        if self._stopped and not chunk:
            return None
        return chunk

    def drain(self) -> None:
        """Discard any chunks queued but not yet consumed.

        The background audio callback keeps queuing chunks continuously,
        even while a caller is busy elsewhere (e.g. recording a multi-second
        voice command after a wake-word trigger — see
        `app.voice.wakeword.BackgroundListener.run_once`). Without draining,
        monitoring resumes by chewing through an ever-growing backlog of
        stale, real-time-drifted audio instead of picking back up at the
        present moment.
        """
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
