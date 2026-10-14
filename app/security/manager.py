"""Security subsystem (section 7).

`SecurityGate` is the concrete `PermissionGate` the Tool Registry calls —
it composes `PermissionManager` (risk-level policy) and `ConfirmationManager`
(interactive confirmation for HIGH/CRITICAL) into the one check the registry
needs, so tools and the registry never talk to either manager directly.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sqlalchemy import select

from app.core.enums import RiskLevel
from app.core.errors import ConfirmationRequiredError, PermissionDeniedError
from app.core.logging import get_logger
from app.tools.base import Tool

if TYPE_CHECKING:
    from app.database.engine import Database

logger = get_logger(__name__)

# A confirmation prompt handler: given a tool name + payload, return True to
# proceed, False to deny. The GUI/voice layer supplies the real
# implementation (Phase 9); tests and headless callers can supply a canned
# yes/no.
ConfirmationPrompt = Callable[[str, dict[str, Any]], Awaitable[bool]]


def summarize_payload(payload: Any, limit: int = 200) -> Any:
    """Shorten long strings so a confirmation dialog stays readable (a
    write_file call would otherwise dump the whole file into the prompt)."""
    if isinstance(payload, str):
        if len(payload) <= limit:
            return payload
        return f"{payload[:limit]}... (+{len(payload) - limit} more characters)"
    if isinstance(payload, dict):
        return {k: summarize_payload(v, limit) for k, v in payload.items()}
    if isinstance(payload, list):
        return [summarize_payload(v, limit) for v in payload]
    return payload


class PermissionManager:
    """Decides whether a risk level is allowed to run at all right now.

    This is deliberately simple in Phase 7: a per-risk-level on/off switch
    (e.g. "no CRITICAL actions while unattended"). Finer-grained,
    per-tool policy can be layered on top later without changing the
    registry's call site.
    """

    def __init__(
        self,
        blocked_risk_levels: set[RiskLevel] | None = None,
    ) -> None:
        self._blocked = blocked_risk_levels or set()

    def block(self, risk_level: RiskLevel) -> None:
        self._blocked.add(risk_level)

    def unblock(self, risk_level: RiskLevel) -> None:
        self._blocked.discard(risk_level)

    def check(self, tool: Tool) -> None:
        if tool.metadata.risk_level in self._blocked:
            raise PermissionDeniedError(
                f"{tool.metadata.name} is a {tool.metadata.risk_level.value}-risk "
                "action, which is currently disabled.",
                user_message=f"I'm not allowed to do that right now ({tool.metadata.name}).",
            )


class ConfirmationManager:
    """Gets explicit user confirmation for HIGH/CRITICAL risk tool calls."""

    def __init__(
        self,
        prompt: ConfirmationPrompt | None = None,
        require_for: set[RiskLevel] = frozenset({RiskLevel.HIGH, RiskLevel.CRITICAL}),  # type: ignore[assignment]
    ) -> None:
        self._prompt = prompt
        self._require_for = require_for

    async def confirm(self, tool: Tool, payload: dict[str, Any]) -> None:
        per_call = getattr(tool, "needs_confirmation", None)
        static_flag = (
            per_call(payload) if callable(per_call) else tool.metadata.requires_confirmation
        )
        needs_confirmation = static_flag or (tool.metadata.risk_level in self._require_for)
        if not needs_confirmation:
            return

        if self._prompt is None:
            raise ConfirmationRequiredError(
                f"{tool.metadata.name} requires confirmation but no confirmation "
                "prompt is configured.",
                user_message=f"I need your confirmation before I can {tool.metadata.name}.",
            )

        approved = await self._prompt(tool.metadata.name, summarize_payload(payload))
        if not approved:
            raise PermissionDeniedError(
                f"User declined confirmation for {tool.metadata.name}.",
                user_message="Okay, I won't do that.",
            )


class CommandPolicy:
    """Allowlist/denylist + timeout policy for the `run_command` developer tool.

    Kept separate from `PermissionManager` because it operates on raw
    command strings, not risk levels — a HIGH-risk tool call can still be an
    outright-denied command regardless of confirmation.
    """

    def __init__(
        self,
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        self.allowlist = allowlist or []
        self.denylist = denylist or []
        self.timeout_seconds = timeout_seconds

    def is_allowed(self, command: str) -> tuple[bool, str | None]:
        normalized = command.strip().lower()
        for denied in self.denylist:
            if denied.lower() in normalized:
                return False, f"Command matches a denylisted pattern: '{denied}'."

        if self.allowlist and not any(normalized.startswith(a.lower()) for a in self.allowlist):
            return False, "Command is not in the configured allowlist."

        return True, None


@dataclass
class AuditLogEntry:
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    tool_name: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # "allowed" | "denied" | "confirmed" | "error"
    detail: str = ""


class AuditLogger:
    """Append-only in-memory audit trail used by isolated tests and callers.

    The assembled NOVA runtime uses ``PersistentAuditLogger`` below so the
    same entries also survive process restarts.
    """

    def __init__(self) -> None:
        self._entries: list[AuditLogEntry] = []

    def record(
        self, tool_name: str, payload: dict[str, Any], outcome: str, detail: str = ""
    ) -> AuditLogEntry:
        entry = AuditLogEntry(tool_name=tool_name, payload=payload, outcome=outcome, detail=detail)
        self._entries.append(entry)
        logger.info("audit_entry", tool=tool_name, outcome=outcome, detail=detail)
        return entry

    def entries(self) -> list[AuditLogEntry]:
        return list(self._entries)

    async def record_async(
        self, tool_name: str, payload: dict[str, Any], outcome: str, detail: str = ""
    ) -> AuditLogEntry:
        return self.record(tool_name, payload, outcome, detail)


_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
}


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return normalized in _SENSITIVE_KEYS or normalized.endswith(
        ("_api_key", "_password", "_secret", "_token")
    )


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]" if _is_sensitive_key(key) else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_payload(item) for item in value]
    return value


class PersistentAuditLogger(AuditLogger):
    """Audit logger that keeps the in-memory view and appends to SQLite."""

    def __init__(self, db: Database) -> None:
        super().__init__()
        self._db = db

    async def record_async(
        self, tool_name: str, payload: dict[str, Any], outcome: str, detail: str = ""
    ) -> AuditLogEntry:
        from app.database.models import AuditRecord

        safe_payload = _redact_payload(payload)
        entry = self.record(tool_name, safe_payload, outcome, detail)
        record = AuditRecord(
            id=entry.id,
            timestamp=entry.timestamp,
            tool_name=tool_name,
            payload_json=json.dumps(safe_payload, default=str, sort_keys=True),
            outcome=outcome,
            detail=detail,
        )
        async with self._db.session() as session:
            session.add(record)
            await session.commit()
        return entry

    async def persisted_entries(self, limit: int = 100) -> list[AuditLogEntry]:
        from app.database.models import AuditRecord

        async with self._db.session() as session:
            result = await session.execute(
                select(AuditRecord).order_by(AuditRecord.timestamp.desc()).limit(limit)
            )
            records = result.scalars().all()
        return [
            AuditLogEntry(
                id=record.id,
                timestamp=record.timestamp,
                tool_name=record.tool_name,
                payload=json.loads(record.payload_json),
                outcome=record.outcome,
                detail=record.detail,
            )
            for record in records
        ]


class SecurityGate:
    """The concrete `PermissionGate` the Tool Registry calls.

    Composes PermissionManager + ConfirmationManager + AuditLogger. This
    class implements the synchronous `check(tool, payload)` the registry
    protocol expects for the risk-level gate, plus an async
    `check_and_confirm` the registry should call before proceeding when
    confirmation is required (see app/tools/registry.py wiring notes).
    """

    def __init__(
        self,
        permissions: PermissionManager | None = None,
        confirmations: ConfirmationManager | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self.permissions = permissions or PermissionManager()
        self.confirmations = confirmations or ConfirmationManager()
        self.audit = audit or AuditLogger()

    def check(self, tool: Tool, payload: dict[str, Any]) -> None:
        try:
            self.permissions.check(tool)
        except PermissionDeniedError as exc:
            self.audit.record(tool.metadata.name, payload, "denied", exc.message)
            raise
        self.audit.record(tool.metadata.name, payload, "allowed")

    async def _audit(
        self, tool_name: str, payload: dict[str, Any], outcome: str, detail: str = ""
    ) -> None:
        """Best-effort audit write: a failing audit store is logged loudly but
        must not block a tool call or mask the real denial/confirmation error
        (same policy the registry applies to outcome logging)."""
        try:
            await self.audit.record_async(tool_name, payload, outcome, detail)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "audit_persistence_failed", tool=tool_name, outcome=outcome, error=str(exc)
            )

    async def check_and_confirm(self, tool: Tool, payload: dict[str, Any]) -> None:
        """Full async gate: risk-level check, then confirmation if required.

        The audit trail records the *decision* (denied, or allowed/confirmed)
        after the prompt resolves, not "allowed" before the user answered.
        """
        name = tool.metadata.name
        try:
            self.permissions.check(tool)
        except PermissionDeniedError as exc:
            await self._audit(name, payload, "denied", exc.message)
            raise

        per_call = getattr(tool, "needs_confirmation", None)
        asked = per_call(payload) if callable(per_call) else tool.metadata.requires_confirmation
        try:
            await self.confirmations.confirm(tool, payload)
        except (ConfirmationRequiredError, PermissionDeniedError) as exc:
            await self._audit(name, payload, "denied", exc.message)
            raise
        await self._audit(name, payload, "allowed")
        if asked:
            await self._audit(name, payload, "confirmed")

    async def record_outcome(
        self, tool: Tool, payload: dict[str, Any], outcome: str, detail: str = ""
    ) -> None:
        await self.audit.record_async(tool.metadata.name, payload, outcome, detail)
