from __future__ import annotations

import pytest

from app.config.settings import Settings, VoiceProviderName
from app.core.errors import ConfigurationError
from app.voice.factory import build_stt_provider, build_tts_provider
from app.voice.groq_stt import GroqSTTProvider
from app.voice.groq_tts import GroqTTSProvider
from app.voice.null_providers import NullSTTProvider, NullTTSProvider


def test_build_stt_provider_defaults_to_null() -> None:
    settings = Settings(_env_file=None, voice_provider=VoiceProviderName.NONE)  # type: ignore[call-arg]
    assert isinstance(build_stt_provider(settings), NullSTTProvider)


def test_build_tts_provider_defaults_to_null() -> None:
    settings = Settings(_env_file=None, voice_provider=VoiceProviderName.NONE)  # type: ignore[call-arg]
    assert isinstance(build_tts_provider(settings), NullTTSProvider)


def test_build_stt_provider_returns_groq_when_key_present() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_api_key="gsk_fake",
    )
    assert isinstance(build_stt_provider(settings), GroqSTTProvider)


def test_build_tts_provider_returns_groq_when_key_present() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_api_key="gsk_fake",
    )
    assert isinstance(build_tts_provider(settings), GroqTTSProvider)


def test_build_stt_provider_raises_without_key_for_groq() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_api_key=None,
    )
    with pytest.raises(ConfigurationError):
        build_stt_provider(settings)


def test_build_tts_provider_raises_without_key_for_groq() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_api_key=None,
    )
    with pytest.raises(ConfigurationError):
        build_tts_provider(settings)
