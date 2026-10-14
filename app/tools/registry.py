"""Central Tool Registry.

This is the *only* path through which the agent invokes a tool. It wires
together validation, the security layer, execution, and audit logging so
individual tools stay simple and no caller can accidentally skip a check.
"""

from __future__ import annotations

from typing import Any, Protocol

from app.core.errors import ToolNotFoundError
from app.core.logging import get_logger
from app.tools.base import Tool, ToolMetadata, ToolResult

logger = get_logger(__name__)


class PermissionGate(Protocol):
    """What the registry needs from the security layer (Phase 7).

    Kept as a narrow Protocol so the registry can be tested without the
    real security subsystem, and so the security layer's public surface can
    grow without forcing changes here.
    """

    def check(self, tool: Tool, payload: dict[str, Any]) -> None:
        """Raise `PermissionDeniedError` or `ConfirmationRequiredError` to block."""
        ...


class AllowAllGate:
    """A permission gate that allows everything. Used before Phase 7 lands
    and in tests that don't care about security behavior."""

    def check(self, tool: Tool, payload: dict[str, Any]) -> None:
        return None


class ToolRegistry:
    """Registers tools by name and runs the validate -> permission -> execute
    pipeline for each invocation."""

    def __init__(self, permission_gate: PermissionGate | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._gate: PermissionGate = permission_gate or AllowAllGate()

    def register(self, tool: Tool) -> None:
        name = tool.metadata.name
        if name in self._tools:
            logger.warning("tool_reregistered", tool=name)
        self._tools[name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(f"No tool registered with name '{name}'")
        return tool

    def list_metadata(self) -> list[ToolMetadata]:
        """Return metadata for every registered tool (e.g. to hand to the LLM
        as its available-tools list)."""
        return [tool.metadata for tool in self._tools.values()]

    def list_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specs in a provider-agnostic canonical shape:
        {"name", "description", "parameters"} — each concrete `LLMProvider`
        translates this into its own native tool-calling format (Anthropic's
        `input_schema`, OpenAI/Groq's `function.parameters`, etc.)."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            }
            for t in self.list_metadata()
        ]

    async def _record_outcome(
        self,
        tool: Tool,
        payload: dict[str, Any],
        outcome: str,
        detail: str = "",
    ) -> None:
        recorder = getattr(self._gate, "record_outcome", None)
        if recorder is None:
            return
        try:
            await recorder(tool, payload, outcome, detail)
        except Exception as exc:  # noqa: BLE001 - an audit outage must not hide a tool result
            logger.error("audit_persistence_failed", tool=tool.metadata.name, error=str(exc))

    async def invoke(self, name: str, payload: dict[str, Any]) -> ToolResult:
        """Run the full validate -> permission -> execute -> log pipeline."""
        tool = self.get(name)

        logger.info("tool_requested", tool=name, payload=payload)
        try:
            validated = tool.validate_input(payload)
            confirm = getattr(self._gate, "check_and_confirm", None)
            if confirm is not None:
                await confirm(tool, validated)
            else:
                self._gate.check(tool, validated)
            tool.check_permission(validated)

            logger.info("tool_started", tool=name)
            result = await tool.execute(validated)
        except Exception as exc:  # noqa: BLE001 - convert to a logged, structured failure
            logger.error("tool_failed", tool=name, error=str(exc))
            await self._record_outcome(tool, payload, "error", str(exc))
            raise

        if result.success:
            logger.info("tool_completed", tool=name)
            await self._record_outcome(tool, validated, "completed", result.message)
        else:
            logger.warning("tool_completed_with_failure", tool=name, error=result.error)
            await self._record_outcome(tool, validated, "failed", result.error or "")

        return result
