"""Shared enumerations used across NOVA subsystems.

Keeping these in one place (rather than duplicating string literals across
tools, security, and planning code) avoids "magic strings" per the project's
engineering principles.
"""

from __future__ import annotations

from enum import StrEnum


class RiskLevel(StrEnum):
    """How dangerous a tool invocation is.

    HIGH and CRITICAL operations must pass through the ConfirmationManager
    before execution (see app/security). LOW and MEDIUM may still require a
    permission check, but do not require an interactive confirmation.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(StrEnum):
    """Lifecycle state of a user-requested task, as tracked by the planner."""

    PENDING = "pending"
    PLANNING = "planning"
    RUNNING = "running"
    WAITING_FOR_CONFIRMATION = "waiting_for_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(StrEnum):
    """Internal event bus event names (see section 12 of the project spec)."""

    VOICE_STARTED = "voice_started"
    VOICE_STOPPED = "voice_stopped"
    TRANSCRIPT_CREATED = "transcript_created"
    TASK_CREATED = "task_created"
    PLAN_CREATED = "plan_created"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    CONFIRMATION_RECEIVED = "confirmation_received"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RESPONSE_READY = "response_ready"
