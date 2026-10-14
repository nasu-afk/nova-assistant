"""Builds the configured STT/TTS providers from `Settings`.

Mirrors `app.ai.factory.build_llm_provider`: one place maps
`NOVA_VOICE_PROVIDER` to concrete classes, so wiring code never imports
`WhisperSTTProvider`/`GroqSTTProvider`/etc. directly.

When `NOVA_VOICE_PROVIDER=groq`, both STT and TTS reuse `NOVA_LLM_API_KEY` -
the common case is one Groq key driving the LLM, transcription, and speech
all together (as set up in this project's README). If you want a separate
Groq key just for voice, set `NOVA_LLM_API_KEY` to that key and point
`NOVA_LLM_PROVIDER` elsewhere, or extend `Settings` with a dedicated field.
"""

from __future__ import annotations

from app.config.settings import Settings, VoiceProviderName
from app.core.errors import ConfigurationError
from app.voice.base import STTProvider, TTSProvider
from app.voice.null_providers import NullSTTProvider, NullTTSProvider


def build_stt_provider(settings: Settings) -> STTProvider:
    if settings.voice_provider == VoiceProviderName.NONE:
        return NullSTTProvider()

    if settings.voice_provider == VoiceProviderName.GROQ:
        if settings.llm_api_key is None:
            raise ConfigurationError(
                "NOVA_VOICE_PROVIDER=groq requires NOVA_LLM_API_KEY to be set.",
                user_message="I don't have a Groq API key configured for voice yet.",
            )
        from app.voice.groq_stt import GroqSTTProvider

        return GroqSTTProvider(api_key=settings.llm_api_key.get_secret_value())

    if settings.voice_provider == VoiceProviderName.WHISPER_LOCAL:
        from app.voice.whisper_stt import WhisperSTTProvider

        return WhisperSTTProvider()

    raise ConfigurationError(
        f"No STTProvider implementation is wired up for '{settings.voice_provider}' yet.",
        user_message="That voice provider isn't set up yet.",
    )


def build_tts_provider(settings: Settings) -> TTSProvider:
    if settings.voice_provider == VoiceProviderName.NONE:
        return NullTTSProvider()

    if settings.voice_provider == VoiceProviderName.GROQ:
        if settings.llm_api_key is None:
            raise ConfigurationError(
                "NOVA_VOICE_PROVIDER=groq requires NOVA_LLM_API_KEY to be set.",
                user_message="I don't have a Groq API key configured for voice yet.",
            )
        from app.voice.groq_tts import GroqTTSProvider

        return GroqTTSProvider(api_key=settings.llm_api_key.get_secret_value())

    if settings.voice_provider == VoiceProviderName.WHISPER_LOCAL:
        # Whisper is STT-only; local TTS falls back to pyttsx3.
        from app.voice.pyttsx3_tts import Pyttsx3TTSProvider

        return Pyttsx3TTSProvider()

    raise ConfigurationError(
        f"No TTSProvider implementation is wired up for '{settings.voice_provider}' yet.",
        user_message="That voice provider isn't set up yet.",
    )
