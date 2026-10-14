"""Local text-to-speech provider using `pyttsx3` (offline, OS-native voices).

Imported lazily for the same reason as `WhisperSTTProvider` — this class can
be referenced in wiring/config code without the `voice` extra installed;
the ImportError only surfaces on first real use.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any

from app.core.errors import VoiceError


class Pyttsx3TTSProvider:
    """TTSProvider backed by the local `pyttsx3` engine."""

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        try:
            import pyttsx3
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'pyttsx3' package is not installed (install the 'voice' "
                "extra: `uv sync --extra voice`).",
                user_message="Speech output isn't available yet.",
            ) from exc

        self._pyttsx3 = pyttsx3
        self._rate = rate
        self._volume = volume

    def _new_engine(self) -> Any:
        engine = self._pyttsx3.init()
        engine.setProperty("rate", self._rate)
        engine.setProperty("volume", self._volume)
        return engine

    def _synthesize_sync(self, text: str) -> bytes:
        engine = self._new_engine()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "speech.wav"
            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
            return out_path.read_bytes() if out_path.exists() else b""

    def _speak_sync(self, text: str) -> None:
        engine = self._new_engine()
        engine.say(text)
        engine.runAndWait()

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        await asyncio.to_thread(self._speak_sync, text)
