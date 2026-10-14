"""Screen understanding: "what's on my screen?" support.

Reuses `TakeScreenshotTool` for the actual capture (Phase 2) rather than
duplicating screenshot logic, then sends the image to a vision-capable
LLM call. The LLM call is expressed against the same `LLMProvider`
interface as everything else — no new provider abstraction needed, since
Anthropic's Messages API already accepts image content blocks.
"""

from __future__ import annotations

import asyncio
import base64
import os
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from app.ai.base import ChatMessage, LLMProvider
from app.core.enums import RiskLevel
from app.core.errors import ToolExecutionError, ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.windows.media import TakeScreenshotTool


class DescribeScreenTool(Tool):
    """Captures the screen and asks a vision-capable LLM to describe it."""

    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="describe_screen",
        description="Capture the screen and describe what's currently visible.",
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to look for on screen (optional)",
                }
            },
        },
    )

    def __init__(self, llm: LLMProvider, screenshot_tool: TakeScreenshotTool | None = None) -> None:
        self._llm = llm
        self._screenshot_tool = screenshot_tool or TakeScreenshotTool()

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = payload.get("question", "What's on the screen right now?")
        if not isinstance(question, str):
            raise ValidationError("describe_screen 'question' must be a string")
        return {"question": question}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        # A hard-coded /tmp path doesn't exist on Windows (NOVA's target OS); use a
        # real temp file, and delete it afterwards so screenshots don't pile up.
        fd, tmp_path = tempfile.mkstemp(prefix="nova_screen_", suffix=".png")
        os.close(fd)
        try:
            capture = await self._screenshot_tool.execute({"path": tmp_path})
            if not capture.success:
                return ToolResult(
                    success=False, error=f"Couldn't capture the screen: {capture.error}"
                )
            try:
                image_bytes = await asyncio.to_thread(Path(tmp_path).read_bytes)
            except OSError as exc:
                raise ToolExecutionError(f"Failed to read captured screenshot: {exc}") from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)  # noqa: ASYNC240 - tiny temp file

        image_b64 = base64.b64encode(image_bytes).decode()
        message = ChatMessage(
            role="user",
            content=payload["question"],
            metadata={"image_base64": image_b64, "media_type": "image/png"},
        )
        response = await self._llm.complete([message], max_tokens=500)

        return ToolResult(
            success=True, data={"description": response.content}, message=response.content
        )
