"""Structured planning models.

Plans are represented as Pydantic models, never as parsed strings — the LLM
is asked to emit plan steps as structured JSON (tool name + arguments), and
this module defines exactly what that JSON must look like.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.core.enums import TaskStatus


class PlanStep(BaseModel):
    """One step of a plan: either a tool call or a natural-language note."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result_summary: str | None = None

    @property
    def is_tool_step(self) -> bool:
        return self.tool_name is not None


class Plan(BaseModel):
    """An ordered sequence of steps produced for one user request."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING

    def next_pending_step(self) -> PlanStep | None:
        return next((s for s in self.steps if s.status == TaskStatus.PENDING), None)

    def is_complete(self) -> bool:
        return all(s.status == TaskStatus.COMPLETED for s in self.steps)

    def has_failure(self) -> bool:
        return any(s.status == TaskStatus.FAILED for s in self.steps)
