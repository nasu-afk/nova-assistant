from __future__ import annotations

from typing import Any, ClassVar

import pytest

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult


class EchoTool(Tool):
    """A trivial mock tool used only to exercise the base interface."""

    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="echo",
        description="Echoes back the given text.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "text" not in payload:
            raise ValidationError("echo requires a 'text' field")
        return payload

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, data={"echo": payload["text"]}, message=payload["text"])


class AlwaysFailsTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="always_fails",
        description="A tool that always fails, for testing error paths.",
        risk_level=RiskLevel.HIGH,
        requires_confirmation=True,
    )

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        return ToolResult(success=False, error="simulated failure")


def test_tool_metadata_declares_risk_level() -> None:
    assert EchoTool.metadata.risk_level == RiskLevel.LOW
    assert AlwaysFailsTool.metadata.risk_level == RiskLevel.HIGH
    assert AlwaysFailsTool.metadata.requires_confirmation is True


def test_validate_input_raises_on_missing_field() -> None:
    tool = EchoTool()
    with pytest.raises(ValidationError):
        tool.validate_input({})


def test_validate_input_passes_through_valid_payload() -> None:
    tool = EchoTool()
    result = tool.validate_input({"text": "hello"})
    assert result == {"text": "hello"}


@pytest.mark.asyncio
async def test_execute_returns_structured_result() -> None:
    tool = EchoTool()
    payload = tool.validate_input({"text": "hello nova"})
    result = await tool.execute(payload)

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data["echo"] == "hello nova"


@pytest.mark.asyncio
async def test_format_result_success_uses_message() -> None:
    tool = EchoTool()
    result = await tool.execute({"text": "hi"})
    assert tool.format_result(result) == "hi"


def test_format_result_failure_uses_error() -> None:
    tool = AlwaysFailsTool()
    result = ToolResult(success=False, error="disk full")
    assert tool.format_result(result) == "disk full"


def test_format_result_falls_back_to_default_sentence() -> None:
    tool = EchoTool()
    assert tool.format_result(ToolResult(success=True)) == "Done."
    assert tool.format_result(ToolResult(success=False)) == "That didn't work."


def test_check_permission_default_allows() -> None:
    # Default implementation defers to the security layer (Phase 7) and
    # should not itself raise for an unremarkable payload.
    tool = EchoTool()
    tool.check_permission({"text": "hello"})  # should not raise
