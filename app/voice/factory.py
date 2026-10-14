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


def _groq_voice_key(settings: Settings) -> str:
    """Key used for Groq STT/TTS: NOVA_VOICE_API_KEY, else the LLM key *only if*
    the LLM provider is Groq as well (never ship another vendor's key to Groq)."""
    from app.config.settings import LLMProviderName

    if settings.voice_api_key is not None:
        return settings.voice_api_key.get_secret_value()
    if settings.llm_api_key is not None and settings.llm_provider == LLMProviderName.GROQ:
        return settings.llm_api_key.get_secret_value()
    raise ConfigurationError(
        "NOVA_VOICE_PROVIDER=groq needs a Groq key: set NOVA_VOICE_API_KEY "
        "(or use NOVA_LLM_PROVIDER=groq so NOVA_LLM_API_KEY is reused).",
        user_message="I don't have a Groq API key configured for voice yet.",
    )


def build_stt_provider(settings: Settings) -> STTProvider:
    if settings.voice_provider == VoiceProviderName.NONE:
        return NullSTTProvider()

    if settings.voice_provider == VoiceProviderName.GROQ:
        from app.voice.groq_stt import GroqSTTProvider

        return GroqSTTProvider(api_key=_groq_voice_key(settings))

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
        from app.voice.groq_tts import GroqTTSProvider

        return GroqTTSProvider(api_key=_groq_voice_key(settings))

    if settings.voice_provider == VoiceProviderName.WHISPER_LOCAL:
        # Whisper is STT-only; local TTS falls back to pyttsx3.
        from app.voice.pyttsx3_tts import Pyttsx3TTSProvider

        return Pyttsx3TTSProvider()

    raise ConfigurationError(
        f"No TTSProvider implementation is wired up for '{settings.voice_provider}' yet.",
        user_message="That voice provider isn't set up yet.",
    )
