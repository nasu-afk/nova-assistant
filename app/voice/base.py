"""Voice provider abstractions.

Concrete implementations (Whisper-based STT, a system or cloud TTS voice, a
wake-word engine) are added starting in Phase 1 / Phase 10. Nothing in the
core agent should import a concrete voice library directly — only these
protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Normalized speech-to-text result."""

    text: str
    language: str | None = None
    confidence: float | None = None


class STTProvider(Protocol):
    """Speech-to-text provider interface."""

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ) -> TranscriptionResult:
        """Transcribe raw PCM audio into text."""
        ...


class TTSProvider(Protocol):
    """Text-to-speech provider interface."""

    async def synthesize(self, text: str) -> bytes:
        """Render `text` to playable audio bytes (e.g. WAV)."""
        ...

    async def speak(self, text: str) -> None:
        """Synthesize and immediately play `text` through the default output device."""
        ...


class WakeWordProvider(Protocol):
    """Wake-word ("NOVA") detection interface. Added in Phase 10."""

    async def listen_for_wake_word(self) -> None:
        """Block (asynchronously) until the configured wake word is detected."""
        ...
