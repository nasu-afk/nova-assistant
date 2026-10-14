"""NOVA's exception hierarchy.

Every subsystem raises a specific, typed error rather than a bare Exception
or a string. This lets the agent explain failures to the user in natural
language (see section 23) without leaking stack traces, while logs retain
full technical detail (section 19/25).
"""

from __future__ import annotations


class NovaError(Exception):
    """Base class for every error NOVA raises intentionally.

    Attributes:
        message: A short technical description, safe to log.
        user_message: An optional natural-language explanation safe to speak
            or display to the end user. Falls back to `message` if unset.
    """

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.user_message = user_message or message


class ValidationError(NovaError):
    """Input to a tool, plan step, or API request failed schema validation."""


class ToolExecutionError(NovaError):
    """A tool ran but failed to complete its task."""


class ToolNotFoundError(NovaError):
    """A plan or agent step referenced a tool name that isn't registered."""


class PermissionDeniedError(NovaError):
    """The security layer refused to allow a requested operation."""


class ConfirmationRequiredError(NovaError):
    """A HIGH/CRITICAL risk action needs explicit user confirmation first.

    This is not a hard failure — the agent should surface a confirmation
    request to the user and resume the task once CONFIRMATION_RECEIVED
    fires, rather than treating this as a terminal error.
    """


class ProviderError(NovaError):
    """An external provider (LLM, STT, TTS, browser) failed or misbehaved."""


class NovaTimeoutError(NovaError):
    """An operation exceeded its allotted time budget."""


class BrowserError(NovaError):
    """A browser-automation tool failed."""


class VoiceError(NovaError):
    """The voice pipeline (VAD, STT, TTS, wake word) failed."""


class MemoryError_(NovaError):
    """The memory subsystem failed to store or retrieve data.

    Named `MemoryError_` (trailing underscore) to avoid shadowing the
    built-in `MemoryError`.
    """


class ConfigurationError(NovaError):
    """Application configuration is missing or invalid at startup."""
