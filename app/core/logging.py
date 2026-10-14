"""Structured logging configuration for NOVA.

Every important operation should log through `structlog`, with context
(task_id, tool_call_id, etc.) bound onto the logger rather than interpolated
into message strings, so that logs can be reconstructed and queried per
section 17 of the project spec.

Secrets, API keys, passwords, and raw sensitive user data must never be
passed to these loggers.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config.settings import LogLevel, Settings, get_settings

_configured = False


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog + stdlib logging once for the whole process.

    Safe to call multiple times; only the first call takes effect unless
    `force=True`-style re-configuration is added later.
    """
    global _configured
    if _configured:
        return

    settings = settings or get_settings()

    level = getattr(logging, LogLevel(settings.log_level).value)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str) -> Any:
    """Return a structlog logger bound to `name` (typically `__name__`).

    Configures logging on first use so callers never need to remember to
    call `configure_logging()` themselves.

    Returns `Any` rather than a specific structlog type because the actual
    wrapper class depends on runtime configuration (see `configure_logging`)
    and structlog itself types `get_logger` as returning `Any`.
    """
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)
