from __future__ import annotations

from typing import Any

import pytest

from app.ai.base import ChatMessage, LLMResponse
from app.core.enums import RiskLevel
from app.core.planner import Planner, PlanningError
from app.tools.base import ToolMetadata


class ScriptedLLM:
    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.seen: list[ChatMessage] | None = None

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.seen = messages
        return LLMResponse(content=self._reply)


TOOLS = [
    ToolMetadata(name="open_application", description="Opens an app.", risk_level=RiskLevel.LOW)
]


@pytest.mark.asyncio
async def test_create_plan_parses_valid_json() -> None:
    llm = ScriptedLLM(
        """{
        "goal": "Open VS Code",
        "steps": [
            {
                "description": "Open VS Code",
                "tool_name": "open_application",
                "tool_args": {"name": "code"}
            }
        ]
        }"""
    )
    planner = Planner(llm)

    plan = await planner.create_plan("open VS Code", TOOLS)

    assert plan.goal == "Open VS Code"
    assert len(plan.steps) == 1
    assert plan.steps[0].tool_name == "open_application"
    assert plan.steps[0].tool_args == {"name": "code"}


@pytest.mark.asyncio
async def test_create_plan_strips_markdown_fences() -> None:
    reply = (
        '```json\n{"goal": "g", "steps": [{"description": "note", '
        '"tool_name": null}]}\n```'
    )
    llm = ScriptedLLM(reply)
    planner = Planner(llm)

    plan = await planner.create_plan("do a thing", TOOLS)

    assert plan.goal == "g"
    assert plan.steps[0].tool_name is None
    assert plan.steps[0].is_tool_step is False


@pytest.mark.asyncio
async def test_create_plan_raises_on_invalid_json() -> None:
    llm = ScriptedLLM("not json at all")
    planner = Planner(llm)

    with pytest.raises(PlanningError):
        await planner.create_plan("do a thing", TOOLS)


@pytest.mark.asyncio
async def test_create_plan_raises_on_missing_steps_field() -> None:
    llm = ScriptedLLM('{"goal": "g"}')
    planner = Planner(llm)

    with pytest.raises(PlanningError):
        await planner.create_plan("do a thing", TOOLS)


@pytest.mark.asyncio
async def test_available_tools_are_rendered_into_prompt() -> None:
    llm = ScriptedLLM('{"goal": "g", "steps": []}')
    planner = Planner(llm)

    await planner.create_plan("do a thing", TOOLS)

    assert llm.seen is not None
    system_message = llm.seen[0].content
    assert "open_application" in system_message


@pytest.mark.asyncio
async def test_empty_tool_list_still_produces_prompt() -> None:
    llm = ScriptedLLM('{"goal": "g", "steps": [{"description": "just chat", "tool_name": null}]}')
    planner = Planner(llm)

    plan = await planner.create_plan("just say hi", [])

    assert plan.steps[0].tool_name is None
