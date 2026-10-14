from __future__ import annotations

from pathlib import Path

from app.config.settings import LLMProviderName, LogLevel, Settings, get_settings


def test_settings_have_sane_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_name == "NOVA"
    assert settings.log_level == LogLevel.INFO
    assert settings.llm_provider == LLMProviderName.ANTHROPIC
    assert settings.require_confirmation_for_high_risk is True
    assert settings.require_confirmation_for_critical_risk is True


def test_settings_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("NOVA_APP_NAME", "NOVA-TEST")
    monkeypatch.setenv("NOVA_LOG_LEVEL", "DEBUG")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.app_name == "NOVA-TEST"
    assert settings.log_level == LogLevel.DEBUG


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b


def test_ensure_data_dir_creates_directory(tmp_path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path / "nova-data")  # type: ignore[call-arg]
    result = settings.ensure_data_dir()

    assert result.exists()
    assert result.is_dir()
    assert isinstance(result, Path)


def test_api_key_is_not_exposed_in_repr() -> None:
    settings = Settings(_env_file=None, llm_api_key="super-secret-key")  # type: ignore[call-arg]

    assert "super-secret-key" not in repr(settings.llm_api_key)
    assert settings.llm_api_key.get_secret_value() == "super-secret-key"
