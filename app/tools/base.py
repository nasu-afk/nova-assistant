"""Tool interface.

Every capability NOVA can execute (opening an app, reading a file, running a
shell command, driving a browser) is implemented as a `Tool` subclass and
registered with the Tool Registry (added in Phase 2). The LLM never touches
the operating system directly — it can only request a tool by name with a
JSON payload, which is then validated, permission-checked, and executed here.

This module intentionally contains no Windows/filesystem/browser/process
logic — see app/tools/windows, app/tools/filesystem, app/tools/browser,
app/tools/developer for concrete tools added in later phases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel

from app.core.enums import RiskLevel


class ToolResult(BaseModel):
    """Structured result every tool execution returns.

    Tools should prefer this over returning arbitrary strings so the agent,
    GUI, and logs can all consume results uniformly.
    """

    success: bool
    data: dict[str, Any] = {}
    message: str = ""
    error: str | None = None


class ToolMetadata(BaseModel):
    """Declarative description of a tool, used for registration and for
    presenting the tool list to the LLM."""

    name: str
    description: str
    risk_level: RiskLevel
    requires_confirmation: bool = False
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}


class Tool(ABC):
    """Base class for all NOVA tools.

    Subclasses must define `metadata` and implement `execute`. The registry
    (Phase 2) is responsible for calling `validate_input` and
    `check_permission` before `execute`, and `format_result` after — a
    caller should never invoke `execute` directly without going through
    that pipeline.
    """

    metadata: ClassVar[ToolMetadata]

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Validate/normalize `payload` against this tool's input schema.

        The default implementation returns the payload unchanged; tools with
        structured inputs should override this to parse into a Pydantic
        model and raise `app.core.errors.ValidationError` on failure.
        """
        return payload

    def needs_confirmation(self, payload: dict[str, Any]) -> bool:
        """Whether *this particular call* needs explicit user confirmation.

        Defaults to the static `metadata.requires_confirmation`; tools can
        override it for calls that are only risky some of the time (e.g.
        overwriting a file that already exists).
        """
        return self.metadata.requires_confirmation

    def check_permission(self, payload: dict[str, Any]) -> None:
        """Raise `PermissionDeniedError` if this call should not proceed.

        The default implementation defers entirely to the security layer's
        PermissionManager/ConfirmationManager (added in Phase 7), which
        wraps every tool call at the registry level. Individual tools only
        need to override this for tool-specific checks (e.g. a path that
        must stay inside an allowed directory).
        """
        return None

    @abstractmethod
    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        """Perform the tool's action and return a structured result."""
        ...

    def format_result(self, result: ToolResult) -> str:
        """Render a `ToolResult` as a short natural-language sentence.

        Used when the agent needs to narrate a tool result back to the user
        via speech or chat (section 23: NOVA should sound natural).
        """
        if result.success:
            return result.message or "Done."
        return result.error or "That didn't work."
