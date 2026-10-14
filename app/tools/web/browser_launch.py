"""Open a URL in the user's actual default browser.

Deliberately separate from `app.tools.browser` (the Playwright-driven
automation session for clicking/typing/scraping pages) and from
`open_application` (which launches a program by name, not a URL). Voice
requests like "open YouTube" or "look up the weather" want the user's
normal browser to pop up with that page - not a headless-automation
session, and not a guess at a browser executable name/path, which is
exactly what went wrong when an LLM tried to pass `"chrome <url>"` as a
program name to `open_application` (Windows' `os.startfile` treats that
whole string as one nonexistent file path and fails).

`webbrowser.open` is stdlib, cross-platform, and always launches whatever
the OS has configured as the default browser - no Chrome-specific path,
no extra dependency.
"""

from __future__ import annotations

import asyncio
import webbrowser
from typing import Any, ClassVar
from urllib.parse import urlparse

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult


class OpenWebpageTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="open_webpage",
        description=(
            "Open a URL in the user's default web browser for browsing or looking "
            "something up. For playing a specific song or video on YouTube, use "
            "play_youtube_video instead - it can actually click and start playback; "
            "this tool only opens the page. Never pass a URL to open_application."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open, e.g. 'https://youtube.com'",
                }
            },
            "required": ["url"],
        },
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = payload.get("url")
        if not url or not isinstance(url, str):
            raise ValidationError("open_webpage requires a 'url' string")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValidationError(f"open_webpage only supports http/https URLs, got: {url!r}")
        return {"url": url}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        url = payload["url"]
        try:
            opened = await asyncio.to_thread(webbrowser.open, url)
        except Exception as exc:  # noqa: BLE001 - no configured browser, etc.
            return ToolResult(success=False, error=f"Couldn't open a browser for '{url}': {exc}")

        if not opened:
            return ToolResult(success=False, error=f"No browser available to open '{url}'.")
        return ToolResult(
            success=True, message=f"Opened {url} in your browser.", data={"url": url}
        )
