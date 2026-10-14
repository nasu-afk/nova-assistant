"""Turns a user request into a structured `Plan` by prompting the LLM for
JSON, never by parsing loose natural language.
"""

from __future__ import annotations

import json
from typing import Any

from app.ai.base import ChatMessage, LLMProvider
from app.core.errors import ValidationError
from app.core.logging import get_logger
from app.core.planning import Plan, PlanStep
from app.tools.base import ToolMetadata

logger = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are NOVA's task planner. Given a user request and a list of available \
tools, output ONLY a JSON object (no prose, no markdown fences) of the form:

{{
  "goal": "<short restatement of the user's goal>",
  "steps": [
    {{"description": "<what this step does>", "tool_name": "<tool name or null>",
     "tool_args": {{...}}}}
  ]
}}

Use "tool_name": null for a step that is just an internal note (e.g. "verify \
result") with no direct tool call. Only use tool names from this list:

{tools}

If the request needs no tools at all, return a plan with a single step and \
tool_name null.
"""


class PlanningError(ValidationError):
    """The LLM's plan response could not be parsed into a valid `Plan`."""


class Planner:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def _render_tools(self, tools: list[ToolMetadata]) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in tools) or "(no tools available)"

    def _parse(self, goal_hint: str, raw: str) -> Plan:
        cleaned = raw.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanningError(f"Planner returned invalid JSON: {exc}. Raw: {raw!r}") from exc

        try:
            steps = [
                PlanStep(
                    description=s["description"],
                    tool_name=s.get("tool_name"),
                    tool_args=s.get("tool_args", {}),
                )
                for s in data["steps"]
            ]
            return Plan(goal=data.get("goal", goal_hint), steps=steps)
        except (KeyError, TypeError) as exc:
            raise PlanningError(
                f"Planner JSON missing required fields: {exc}. Raw: {raw!r}"
            ) from exc

    async def create_plan(self, user_request: str, available_tools: list[ToolMetadata]) -> Plan:
        system = PLANNER_SYSTEM_PROMPT.format(tools=self._render_tools(available_tools))
        messages = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user_request),
        ]
        response = await self._llm.complete(messages, max_tokens=1024, temperature=0.0)

        plan = self._parse(user_request, response.content)
        logger.info("plan_created", goal=plan.goal, step_count=len(plan.steps))
        return plan
