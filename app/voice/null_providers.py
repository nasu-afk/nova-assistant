"""No-op voice providers.

Used when `NOVA_VOICE_PROVIDER=none` (the Phase 0 default), in headless
environments (CI, servers without a mic/speaker), and in tests. Satisfies
`STTProvider`/`TTSProvider` structurally so the rest of the pipeline never
needs to special-case "no voice configured".
"""

from __future__ import annotations

from app.voice.base import TranscriptionResult


class NullSTTProvider:
    """Always returns empty transcriptions. Never touches real audio."""

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ) -> TranscriptionResult:
        return TranscriptionResult(text="", confidence=0.0)


class NullTTSProvider:
    """Synthesizes nothing and speaks nothing. Safe default with no hardware."""

    async def synthesize(self, text: str) -> bytes:
        return b""

    async def speak(self, text: str) -> None:
        return None
