"""Local Whisper-compatible speech-to-text provider.

Uses `faster-whisper` (CTranslate2-based Whisper) so transcription runs
fully offline/local-first. The heavy model-loading dependency is imported
lazily so this module can be imported (and its class referenced in wiring
code) even when the `voice` extra isn't installed — the error only surfaces
if someone actually tries to construct/use it without the dependency.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from app.core.errors import VoiceError
from app.voice.base import TranscriptionResult


class WhisperSTTProvider:
    """STTProvider backed by a local faster-whisper model."""

    def __init__(self, model_size: str = "base", device: str = "cpu") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'faster-whisper' package is not installed (install the "
                "'voice' extra: `uv sync --extra voice`).",
                user_message="Voice transcription isn't available yet.",
            ) from exc

        self._model: Any = WhisperModel(model_size, device=device, compute_type="int8")

    def _transcribe_sync(self, audio_bytes: bytes, sample_rate: int) -> TranscriptionResult:
        # faster-whisper accepts a file path or file-like object; wrap raw
        # PCM16 mono bytes as an in-memory WAV so no temp file is needed.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)
        buffer.seek(0)

        segments, info = self._model.transcribe(buffer)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            confidence=getattr(info, "language_probability", None),
        )

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="")
        return await asyncio.to_thread(self._transcribe_sync, audio_bytes, sample_rate)
