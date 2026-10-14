from __future__ import annotations

from typing import Any, ClassVar

import pytest

from app.core.enums import RiskLevel
from app.core.errors import PermissionDeniedError, ToolNotFoundError, ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.registry import AllowAllGate, ToolRegistry


class GreetTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="greet",
        description="Greets someone by name.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "name" not in payload:
            raise ValidationError("greet requires 'name'")
        return payload

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, message=f"Hello, {payload['name']}!")


class DenyingGate:
    def check(self, tool: Tool, payload: dict[str, Any]) -> None:
        raise PermissionDeniedError(f"{tool.metadata.name} is not allowed")


@pytest.mark.asyncio
async def test_register_and_invoke_tool() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    result = await registry.invoke("greet", {"name": "Hasan"})

    assert result.success is True
    assert result.message == "Hello, Hasan!"


def test_get_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("does_not_exist")


@pytest.mark.asyncio
async def test_invoke_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        await registry.invoke("does_not_exist", {})


@pytest.mark.asyncio
async def test_invalid_payload_raises_validation_error() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    with pytest.raises(ValidationError):
        await registry.invoke("greet", {})


@pytest.mark.asyncio
async def test_permission_gate_blocks_execution() -> None:
    registry = ToolRegistry(permission_gate=DenyingGate())
    registry.register(GreetTool())

    with pytest.raises(PermissionDeniedError):
        await registry.invoke("greet", {"name": "Hasan"})


def test_list_metadata_returns_all_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())

    metadata = registry.list_metadata()

    assert len(metadata) == 1
    assert metadata[0].name == "greet"


def test_unregister_removes_tool() -> None:
    registry = ToolRegistry()
    registry.register(GreetTool())
    registry.unregister("greet")

    with pytest.raises(ToolNotFoundError):
        registry.get("greet")


def test_allow_all_gate_never_blocks() -> None:
    gate = AllowAllGate()
    gate.check(GreetTool(), {"name": "x"})  # should not raise
