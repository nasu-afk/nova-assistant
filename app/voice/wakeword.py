"""Wake word / always-listening support (section 9, Phase 10).

Two pieces:

- `EnergyVAD`: a tiny, dependency-free voice-activity gate (RMS threshold
  over PCM16 audio) that decides whether a chunk of audio is worth sending
  to a wake-word model or STT at all. No ML dependency — always available.
- `WakeWordEngine`: wraps a real wake-word model (`openwakeword`) behind the
  `WakeWordProvider` protocol from `app.voice.base`. Imported lazily so this
  module loads fine without the model installed; only constructing
  `WakeWordEngine` requires it. `openwakeword`'s `tflite-runtime` dependency
  has no Python 3.12 wheel on Linux (which is why this project's own sandbox
  can't install or test it), but that dependency is Linux-only per
  `openwakeword`'s own package metadata — on Windows/macOS it only needs
  `onnxruntime`, which does have current wheels, so it should install
  normally there (see the `voice` extra in pyproject.toml).
- `BackgroundListener`: ties VAD + wake word + push-to-talk's existing
  `VoicePipeline` together into an always-on loop: keep sampling audio,
  gate it through VAD, run the wake-word model only on audio that passes
  the gate, and hand off to the normal STT/agent/TTS turn once "NOVA" is
  heard.
"""

from __future__ import annotations

import array
import asyncio
from collections.abc import Callable
from typing import Any

from app.core.errors import NovaError, VoiceError
from app.core.logging import get_logger
from app.voice.audio_capture import AudioRecorder
from app.voice.pipeline import VoicePipeline

logger = get_logger(__name__)


class EnergyVAD:
    """A minimal voice-activity detector based on RMS energy.

    Not a substitute for a real ML-based VAD (e.g. Silero VAD) in a
    production build, but it's dependency-free and good enough to avoid
    running a wake-word model on pure silence.
    """

    def __init__(self, threshold: int = 500) -> None:
        self.threshold = threshold

    def is_speech(self, pcm16_bytes: bytes) -> bool:
        if not pcm16_bytes:
            return False
        samples = array.array("h")
        samples.frombytes(pcm16_bytes[: len(pcm16_bytes) - (len(pcm16_bytes) % 2)])
        if not samples:
            return False
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        return bool(rms >= self.threshold)


class WakeWordEngine:
    """WakeWordProvider backed by `openwakeword`, imported lazily.

    `openwakeword` ships a small, fixed set of pretrained wake-phrase models
    — as of writing: `alexa`, `hey_mycroft`, `hey_jarvis`, `hey_rhasspy` (plus
    a couple of non-wake-word utility models like `weather`). There is no
    "nova" model, and no way to make it respond to an arbitrary phrase
    without training a custom model — that's a separate, much more involved
    process (openwakeword's own training notebook, ~100k synthetic
    text-to-speech clips) that this project doesn't attempt. Pick one of the
    four above via `wakeword_model` (default `hey_jarvis`) and say *that*
    phrase, not "Hey NOVA".

    `listen_for_wake_word` is intentionally not implemented against real
    hardware here — it operates on audio chunks handed to `score_chunk` by
    `BackgroundListener`, since a desktop app's mic loop is owned by the
    audio backend (see `app.voice.audio_capture`), not by the wake-word
    model itself.
    """

    PRETRAINED_MODELS = ("alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy")

    def __init__(self, wakeword_model: str = "hey_jarvis", threshold: float = 0.5) -> None:
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'openwakeword' package is not installed. Wake-word "
                "detection isn't available in this build.",
                user_message="Always-listening mode isn't set up yet.",
            ) from exc

        if wakeword_model not in self.PRETRAINED_MODELS:
            raise VoiceError(
                f"'{wakeword_model}' is not one of openwakeword's pretrained models "
                f"({', '.join(self.PRETRAINED_MODELS)}).",
                user_message=(
                    f"'{wakeword_model}' isn't a real wake word model — openwakeword only "
                    f"ships {', '.join(self.PRETRAINED_MODELS)}. Set NOVA_WAKE_WORD_MODEL to "
                    "one of those (say that phrase, not 'Hey NOVA')."
                ),
            )

        try:
            # Idempotent: only downloads the file(s) if not already cached
            # locally. Needed because Model() itself doesn't auto-download —
            # it just fails with "could not find pretrained model" if the
            # weights aren't present yet, which looks identical to typing
            # the name wrong and is confusing to debug from that error alone.
            download_models(model_names=[wakeword_model])
        except Exception as exc:  # noqa: BLE001 - network failure, disk, etc.
            raise VoiceError(
                f"Failed to download the '{wakeword_model}' wake-word model: {exc}",
                user_message=(
                    f"Couldn't download the '{wakeword_model}' wake-word model — check "
                    "your internet connection and try again."
                ),
            ) from exc

        self._model: Any = Model(wakeword_models=[wakeword_model])
        self._wakeword_model = wakeword_model
        self._threshold = threshold

    def score_chunk(self, pcm16_bytes: bytes) -> float:
        import numpy as np

        audio = np.frombuffer(pcm16_bytes, dtype=np.int16)
        predictions = self._model.predict(audio)
        return max(predictions.values()) if predictions else 0.0

    def is_wake_word(self, pcm16_bytes: bytes) -> bool:
        return self.score_chunk(pcm16_bytes) >= self._threshold

    async def listen_for_wake_word(self) -> None:  # pragma: no cover - needs real mic loop
        raise NotImplementedError(
            "listen_for_wake_word requires a running audio loop; use "
            "BackgroundListener instead of calling this directly."
        )


class BackgroundListener:
    """Continuously samples short audio chunks, gates them with VAD, checks
    for the wake word, and hands off to `VoicePipeline` once triggered.

    `chunk_source` is injected (rather than reading a real microphone
    directly) so this class is fully testable without audio hardware, and
    so it can be swapped between a real mic stream and a recorded fixture.
    """

    def __init__(
        self,
        recorder: AudioRecorder,
        vad: EnergyVAD,
        wake_word: Callable[[bytes], bool],
        pipeline: VoicePipeline,
        chunk_seconds: float = 1.0,
        command_seconds: float = 4.0,
        on_command_complete: Callable[[], None] | None = None,
    ) -> None:
        self._recorder = recorder
        self._vad = vad
        self._wake_word = wake_word
        self._pipeline = pipeline
        self._chunk_seconds = chunk_seconds
        # After the wake word triggers, NOVA records for this fixed window
        # to capture the command, then transcribes whatever was said in it.
        # This is a real, honest simplification — a fixed window either cuts
        # off a longer command or leaves dead air after a short one — rather
        # than true silence-based endpointing (detecting when the user stops
        # talking), which would need continuous VAD monitoring *during* the
        # recording itself, not just before it. That's a reasonable follow-up
        # if fixed-window turns out to feel wrong in practice.
        self._command_seconds = command_seconds
        # Called after a triggered command completes (success or failure) —
        # e.g. `ContinuousMicrophoneChunkSource.drain`, so wake-word
        # monitoring resumes at the present moment instead of working
        # through a backlog of audio queued during the command window.
        self._on_command_complete = on_command_complete
        self._running = False

    def stop(self) -> None:
        self._running = False

    async def run_once(self, chunk: bytes) -> str | None:
        """Process a single monitoring chunk. Returns the agent's reply text
        if the wake word triggered a full turn, else None."""
        if not self._vad.is_speech(chunk):
            return None
        if not self._wake_word(chunk):
            return None

        logger.info("wake_word_detected")
        self._pipeline.start_listening()
        try:
            await asyncio.sleep(self._command_seconds)
            return await self._pipeline.stop_listening_and_respond()
        except NovaError as exc:
            # A single failed turn (a transient network error, a provider
            # hiccup) must not take down the whole always-listening loop —
            # the user shouldn't have to restart NOVA over one bad request.
            # Compare app.main.run_voice_loop's handle_turn(), which already
            # handles this correctly for push-to-talk; this brings wake-word
            # mode to the same standard.
            logger.error("wake_word_turn_failed", error=exc.message)
            return None
        finally:
            if self._on_command_complete is not None:
                self._on_command_complete()

    async def run(self, chunk_source: Callable[[], bytes | None]) -> None:
        """Loop pulling chunks from `chunk_source` until `stop()` is called
        or the source returns None.

        `chunk_source` is called off the event loop thread (`asyncio.to_thread`)
        since a real implementation blocks waiting for the next chunk of
        audio to become available — calling it directly here would freeze
        the entire event loop (tool execution, TTS playback, everything)
        for up to `chunk_seconds` on every iteration.
        """
        self._running = True
        while self._running:
            chunk = await asyncio.to_thread(chunk_source)
            if chunk is None:
                break
            await self.run_once(chunk)
