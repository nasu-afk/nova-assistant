"""Builds the configured `LLMProvider` from `Settings`.

This is the one place that maps `NOVA_LLM_PROVIDER` to a concrete class —
`app/main.py` and anything else that needs an LLM call `build_llm_provider`
rather than importing `AnthropicProvider`/`GroqProvider` directly, so adding
a new provider never means touching wiring code elsewhere.
"""

from __future__ import annotations

from app.ai.base import LLMProvider
from app.config.settings import LLMProviderName, Settings
from app.core.errors import ConfigurationError


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_api_key is None:
        raise ConfigurationError(
            "NOVA_LLM_API_KEY is not set.",
            user_message="I don't have a language model API key configured yet.",
        )
    api_key = settings.llm_api_key.get_secret_value()

    if settings.llm_provider == LLMProviderName.ANTHROPIC:
        from app.ai.anthropic_provider import AnthropicProvider

        return AnthropicProvider(api_key=api_key, model=settings.llm_model)

    if settings.llm_provider == LLMProviderName.GROQ:
        from app.ai.groq_provider import GroqProvider

        return GroqProvider(api_key=api_key, model=settings.llm_model)

    raise ConfigurationError(
        f"No LLMProvider implementation is wired up for '{settings.llm_provider}' yet.",
        user_message="That language model provider isn't set up yet.",
    )
