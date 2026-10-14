"""Speech-to-text via Groq's hosted Whisper endpoint.

An alternative to `WhisperSTTProvider` (Phase 1, `app/voice/whisper_stt.py`)
that needs no local model download — audio is sent to Groq's
`/audio/transcriptions` endpoint (`whisper-large-v3` or the faster
`whisper-large-v3-turbo`) using the same account/key as `GroqProvider`, so a
single free Groq key can drive both the LLM and STT halves of the voice
pipeline. The official `groq` SDK is imported lazily, matching every other
provider in this codebase.
"""

from __future__ import annotations

import asyncio
import io
import wave
from typing import Any

from app.core.errors import VoiceError
from app.voice.base import TranscriptionResult

DEFAULT_MODEL = "whisper-large-v3-turbo"


class GroqSTTProvider:
    """STTProvider backed by Groq's hosted Whisper transcription API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'groq' package is not installed (run `uv add groq`).",
                user_message="Voice transcription isn't available yet.",
            ) from exc

        self._client: Any = AsyncGroq(api_key=api_key)
        self._model = model

    @staticmethod
    def _to_wav_bytes(pcm16_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16_bytes)
        return buffer.getvalue()

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ) -> TranscriptionResult:
        if not audio_bytes:
            return TranscriptionResult(text="")

        wav_bytes = await asyncio.to_thread(self._to_wav_bytes, audio_bytes, sample_rate)

        try:
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=("audio.wav", wav_bytes),
                response_format="verbose_json",
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/network failure
            raise VoiceError(
                f"Groq transcription request failed: {exc}",
                user_message="I couldn't hear that clearly — could you try again?",
            ) from exc

        return TranscriptionResult(
            text=(response.text or "").strip(),
            language=getattr(response, "language", None),
        )
