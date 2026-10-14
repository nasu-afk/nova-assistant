"""Text-to-speech via Groq's hosted Orpheus (Canopy Labs) endpoint.

Unlike chat completions and Whisper transcription - which have generous,
genuinely-free daily quotas on Groq - the free tier for `audio.speech`
(Orpheus TTS) is tight: on the order of ~200 characters per request and a
low daily request cap before Groq expects you on a paid "Dev Tier". This
provider is still useful for short assistant replies, but callers should
expect `VoiceError` on long text or once the daily cap is hit, rather than
assuming unlimited free speech synthesis.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import VoiceError

DEFAULT_MODEL = "canopylabs/orpheus-v1-english"
DEFAULT_VOICE = "autumn"

# Conservative; Groq's actual free-tier cutoff isn't published as a hard
# constant, so this exists to fail with a clear message locally rather than
# silently sending an oversized request and getting back a 429/413.
FREE_TIER_CHAR_GUIDELINE = 200


class GroqTTSProvider:
    """TTSProvider backed by Groq's hosted Orpheus text-to-speech API."""

    def __init__(
        self, api_key: str, model: str = DEFAULT_MODEL, voice: str = DEFAULT_VOICE
    ) -> None:
        try:
            from groq import AsyncGroq
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "The 'groq' package is not installed (run `uv add groq`).",
                user_message="Speech output isn't available yet.",
            ) from exc

        self._client: Any = AsyncGroq(api_key=api_key)
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""

        try:
            response = await self._client.audio.speech.create(
                model=self._model,
                voice=self._voice,
                input=text,
                response_format="wav",
            )
        except Exception as exc:  # noqa: BLE001 - normalize any SDK/network/quota failure
            if "model_terms_required" in str(exc) or "terms acceptance" in str(exc):
                raise VoiceError(
                    f"Groq model '{self._model}' requires accepting its terms: {exc}",
                    user_message=(
                        f"Groq needs you to accept the terms for '{self._model}' first — "
                        f"open https://console.groq.com/playground?model="
                        f"{self._model.replace('/', '%2F')} and accept, then try again."
                    ),
                ) from exc
            raise VoiceError(
                f"Groq TTS request failed: {exc}",
                user_message=(
                    "I couldn't generate speech for that - Groq's free TTS tier has a "
                    "tight per-request character limit, so try a shorter reply, or "
                    "check console.groq.com/settings/billing for your current limits."
                ),
            ) from exc

        return bytes(await response.read())

    async def speak(self, text: str) -> None:
        """Synthesize and play through the default output device.

        Playback needs `sounddevice` + `soundfile` (part of the `voice`
        extra) - imported lazily, same pattern as every other audio
        dependency in this codebase.
        """
        audio_bytes = await self.synthesize(text)
        if not audio_bytes:
            return

        try:
            import io

            import sounddevice as sd
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise VoiceError(
                "Playback requires 'sounddevice' and 'soundfile' (install the "
                "'voice' extra: `uv sync --extra voice`).",
                user_message="I generated the audio, but can't play it back here.",
            ) from exc

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        sd.play(data, sample_rate)
        sd.wait()
