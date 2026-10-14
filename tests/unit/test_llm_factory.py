from __future__ import annotations

import pytest

from app.ai.anthropic_provider import AnthropicProvider
from app.ai.factory import build_llm_provider
from app.ai.groq_provider import GroqProvider
from app.config.settings import LLMProviderName, Settings
from app.core.errors import ConfigurationError


def test_build_llm_provider_returns_groq_provider() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider=LLMProviderName.GROQ,
        llm_model="llama-3.3-70b-versatile",
        llm_api_key="gsk_fake",
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, GroqProvider)


def test_build_llm_provider_returns_anthropic_provider() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider=LLMProviderName.ANTHROPIC,
        llm_model="claude-sonnet-4-6",
        llm_api_key="sk-ant-fake",
    )

    provider = build_llm_provider(settings)

    assert isinstance(provider, AnthropicProvider)


def test_build_llm_provider_raises_without_api_key() -> None:
    settings = Settings(_env_file=None, llm_api_key=None)  # type: ignore[call-arg]

    with pytest.raises(ConfigurationError):
        build_llm_provider(settings)


def test_build_llm_provider_raises_for_unimplemented_provider() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider=LLMProviderName.LOCAL,
        llm_api_key="anything",
    )

    with pytest.raises(ConfigurationError):
        build_llm_provider(settings)
