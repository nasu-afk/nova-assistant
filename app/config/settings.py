"""Strongly typed application configuration.

All configuration is read from environment variables (optionally via a
`.env` file — see `.env.example`). Nothing here hard-codes a specific LLM,
voice, or database provider: the *names* of providers are configuration
values, and the actual provider classes are selected at wiring time (see
app/ai, app/voice) so they stay swappable per the project's engineering
principles.

Never commit a real `.env` file or real API keys.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProviderName(StrEnum):
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OPENAI = "openai"
    LOCAL = "local"


class VoiceProviderName(StrEnum):
    WHISPER_LOCAL = "whisper_local"
    GROQ = "groq"
    NONE = "none"


# Used when NOVA_LLM_MODEL is not set, so choosing a provider alone is enough
# (previously a Groq user who forgot the model got Anthropic's model name).
DEFAULT_MODELS: dict[LLMProviderName, str] = {
    LLMProviderName.ANTHROPIC: "claude-sonnet-4-6",
    LLMProviderName.GROQ: "llama-3.3-70b-versatile",
}


class Settings(BaseSettings):
    """Application-wide settings.

    Instantiate via `get_settings()` rather than constructing directly, so
    the whole app shares one validated, cached configuration object.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="NOVA_",
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "NOVA"
    environment: str = "development"
    log_level: LogLevel = LogLevel.INFO
    log_json: bool = False
    data_dir: Path = Path("data")

    # --- LLM provider ---
    llm_provider: LLMProviderName = LLMProviderName.ANTHROPIC
    # Empty = use the provider's default (see DEFAULT_MODELS).
    llm_model: str = ""
    llm_api_key: SecretStr | None = Field(default=None)

    # --- Voice ---
    voice_provider: VoiceProviderName = VoiceProviderName.NONE
    push_to_talk_key: str = "ctrl+space"
    # One of app.voice.wakeword.WakeWordEngine.PRETRAINED_MODELS — openwakeword
    # ships a fixed set of phrases; there is no "nova" model (see that class's
    # docstring). Say the actual phrase for whichever model you pick.
    # Key for the voice provider. Only falls back to NOVA_LLM_API_KEY when the LLM
    # provider is Groq too - sending an Anthropic key to Groq leaks it and 401s.
    voice_api_key: SecretStr | None = None
    # Keep every recording as data/debug_last_recording.wav. Off by default (it is
    # your voice); very-quiet recordings are still saved so the warning is actionable.
    debug_save_audio: bool = False
    # One of openwakeword's pretrained names (alexa, hey_mycroft, hey_jarvis,
    # hey_rhasspy) OR the path to a custom .onnx/.tflite model trained for your
    # own phrase (e.g. "Hey NOVA"; see README, "Wake word").
    wake_word_model: str = "hey_jarvis"
    # RMS gate (over a 1s chunk) a chunk must pass before the wake-word model
    # runs. Kept close to the "quiet recording" warning level (150) rather than
    # the old hard-coded 500, which silently swallowed normal-volume speech.
    wake_vad_threshold: int = 200

    # --- Database ---
    database_url: str = "sqlite+aiosqlite:///data/nova.db"

    # --- Browser ---
    browser_headless: bool = True

    # --- Security ---
    require_confirmation_for_high_risk: bool = True
    require_confirmation_for_critical_risk: bool = True
    command_allowlist: list[str] = Field(default_factory=list)
    command_denylist: list[str] = Field(
        # Specific patterns: the old bare "format" also blocked harmless things
        # like `git log --format=...`.
        default_factory=lambda: [
            "rm -rf /",
            "format c:",
            "mkfs",
            "del /f /s /q c:\\",
            "shutdown",
        ]
    )
    command_timeout_seconds: int = 30
    # run_command / git / npm / docker tools are powerful; off unless asked for.
    enable_developer_tools: bool = False
    # Root the file tools may touch. Empty = the user's home directory.
    workspace_root: str = ""

    @model_validator(mode="after")
    def _fill_default_model(self) -> Settings:
        if not self.llm_model.strip():
            self.llm_model = DEFAULT_MODELS.get(self.llm_provider, "")
        return self

    def ensure_data_dir(self) -> Path:
        """Create the data directory if it doesn't exist yet, and return it."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide cached Settings instance, creating it once."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Force settings to be re-read from the environment. Mainly for tests."""
    global _settings
    _settings = Settings()
    return _settings
