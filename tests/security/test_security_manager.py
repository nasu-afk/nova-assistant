from __future__ import annotations

from typing import Any, ClassVar

import pytest

from app.core.enums import RiskLevel
from app.core.errors import ConfirmationRequiredError, PermissionDeniedError
from app.security.manager import (
    AuditLogger,
    CommandPolicy,
    ConfirmationManager,
    PermissionManager,
    PersistentAuditLogger,
    SecurityGate,
)
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.registry import ToolRegistry


class LowRiskTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="low_tool", description="low risk", risk_level=RiskLevel.LOW
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, message="ok")


class HighRiskTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="high_tool",
        description="high risk",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, message="did the dangerous thing")


# --- PermissionManager ---


def test_permission_manager_allows_by_default() -> None:
    pm = PermissionManager()
    pm.check(LowRiskTool())  # should not raise


def test_permission_manager_blocks_a_risk_level() -> None:
    pm = PermissionManager()
    pm.block(RiskLevel.HIGH)

    with pytest.raises(PermissionDeniedError):
        pm.check(HighRiskTool())


def test_permission_manager_unblock_restores_access() -> None:
    pm = PermissionManager()
    pm.block(RiskLevel.HIGH)
    pm.unblock(RiskLevel.HIGH)

    pm.check(HighRiskTool())  # should not raise


# --- ConfirmationManager ---


@pytest.mark.asyncio
async def test_confirmation_not_required_for_low_risk() -> None:
    cm = ConfirmationManager()
    await cm.confirm(LowRiskTool(), {})  # should not raise, no prompt needed


@pytest.mark.asyncio
async def test_confirmation_required_but_no_prompt_raises() -> None:
    cm = ConfirmationManager()
    with pytest.raises(ConfirmationRequiredError):
        await cm.confirm(HighRiskTool(), {})


@pytest.mark.asyncio
async def test_confirmation_approved_proceeds() -> None:
    async def always_yes(tool_name: str, payload: dict[str, Any]) -> bool:
        return True

    cm = ConfirmationManager(prompt=always_yes)
    await cm.confirm(HighRiskTool(), {})  # should not raise


@pytest.mark.asyncio
async def test_confirmation_declined_raises_permission_denied() -> None:
    async def always_no(tool_name: str, payload: dict[str, Any]) -> bool:
        return False

    cm = ConfirmationManager(prompt=always_no)
    with pytest.raises(PermissionDeniedError):
        await cm.confirm(HighRiskTool(), {})


# --- CommandPolicy ---


def test_command_policy_blocks_denylisted_command() -> None:
    policy = CommandPolicy(denylist=["rm -rf /"])
    allowed, reason = policy.is_allowed("rm -rf / --no-preserve-root")

    assert allowed is False
    assert reason is not None


def test_command_policy_allows_non_denylisted_command() -> None:
    policy = CommandPolicy(denylist=["rm -rf /"])
    allowed, reason = policy.is_allowed("git status")

    assert allowed is True
    assert reason is None


def test_command_policy_enforces_allowlist_when_set() -> None:
    policy = CommandPolicy(allowlist=["git", "npm"])

    allowed, _ = policy.is_allowed("git status")
    blocked, reason = policy.is_allowed("curl evil.com")

    assert allowed is True
    assert blocked is False
    assert reason is not None


# --- AuditLogger ---


def test_audit_logger_records_entries() -> None:
    audit = AuditLogger()
    audit.record("delete_file", {"path": "/tmp/x"}, "allowed")

    entries = audit.entries()
    assert len(entries) == 1
    assert entries[0].tool_name == "delete_file"
    assert entries[0].outcome == "allowed"


# --- SecurityGate (integration with ToolRegistry) ---


@pytest.mark.asyncio
async def test_security_gate_allows_low_risk_without_confirmation() -> None:
    gate = SecurityGate()
    registry = ToolRegistry(permission_gate=gate)
    registry.register(LowRiskTool())

    result = await registry.invoke("low_tool", {})

    assert result.success is True
    assert [entry.outcome for entry in gate.audit.entries()] == ["allowed", "completed"]


@pytest.mark.asyncio
async def test_security_gate_blocks_high_risk_without_confirmation_prompt() -> None:
    gate = SecurityGate()
    registry = ToolRegistry(permission_gate=gate)
    registry.register(HighRiskTool())

    with pytest.raises(ConfirmationRequiredError):
        await registry.invoke("high_tool", {})


@pytest.mark.asyncio
async def test_security_gate_allows_high_risk_with_approved_confirmation() -> None:
    async def always_yes(tool_name: str, payload: dict[str, Any]) -> bool:
        return True

    gate = SecurityGate(confirmations=ConfirmationManager(prompt=always_yes))
    registry = ToolRegistry(permission_gate=gate)
    registry.register(HighRiskTool())

    result = await registry.invoke("high_tool", {})

    assert result.success is True
    outcomes = [e.outcome for e in gate.audit.entries()]
    assert "confirmed" in outcomes


@pytest.mark.asyncio
async def test_security_gate_blocked_risk_level_prevents_execution_entirely() -> None:
    permissions = PermissionManager()
    permissions.block(RiskLevel.HIGH)
    gate = SecurityGate(permissions=permissions)
    registry = ToolRegistry(permission_gate=gate)
    registry.register(HighRiskTool())

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("high_tool", {})

    outcomes = [entry.outcome for entry in gate.audit.entries()]
    assert "denied" in outcomes
    assert "error" in outcomes


@pytest.mark.asyncio
async def test_persistent_audit_log_survives_restart_and_redacts_secrets() -> None:
    from app.database.engine import Database

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    audit = PersistentAuditLogger(db)

    await audit.record_async(
        "login_tool",
        {"username": "hasan", "password": "do-not-store", "nested": {"token": "secret"}},
        "completed",
    )

    reloaded = await PersistentAuditLogger(db).persisted_entries()
    assert len(reloaded) == 1
    assert reloaded[0].payload == {
        "nested": {"token": "[REDACTED]"},
        "password": "[REDACTED]",
        "username": "hasan",
    }
    await db.dispose()


@pytest.mark.asyncio
async def test_registry_persists_allowed_and_completed_audit_events() -> None:
    from app.database.engine import Database

    db = Database("sqlite+aiosqlite:///:memory:")
    await db.init()
    audit = PersistentAuditLogger(db)
    gate = SecurityGate(audit=audit)
    registry = ToolRegistry(permission_gate=gate)
    registry.register(LowRiskTool())

    await registry.invoke("low_tool", {"value": 1})

    entries = await audit.persisted_entries()
    assert {entry.outcome for entry in entries} == {"allowed", "completed"}
    await db.dispose()
