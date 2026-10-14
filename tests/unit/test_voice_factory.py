from __future__ import annotations

import pytest

from app.config.settings import LLMProviderName, Settings, VoiceProviderName
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
        llm_provider=LLMProviderName.GROQ,
        llm_api_key="gsk_fake",
    )
    assert isinstance(build_stt_provider(settings), GroqSTTProvider)


def test_build_tts_provider_returns_groq_when_key_present() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_provider=LLMProviderName.GROQ,
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


def test_anthropic_llm_key_is_never_sent_to_groq_voice() -> None:
    """Reusing NOVA_LLM_API_KEY for Groq is only valid when the LLM *is* Groq;
    otherwise an Anthropic key would be shipped to a different vendor."""
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_provider=LLMProviderName.ANTHROPIC,
        llm_api_key="sk-ant-fake",
    )
    with pytest.raises(ConfigurationError, match="NOVA_VOICE_API_KEY"):
        build_stt_provider(settings)
    with pytest.raises(ConfigurationError, match="NOVA_VOICE_API_KEY"):
        build_tts_provider(settings)


def test_dedicated_voice_key_is_used_for_groq_voice() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        voice_provider=VoiceProviderName.GROQ,
        llm_provider=LLMProviderName.ANTHROPIC,
        llm_api_key="sk-ant-fake",
        voice_api_key="gsk_voice_key",
    )
    assert isinstance(build_stt_provider(settings), GroqSTTProvider)
    assert isinstance(build_tts_provider(settings), GroqTTSProvider)
