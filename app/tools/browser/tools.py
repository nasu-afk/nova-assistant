"""Browser automation tools.

All tools in this file share one `BrowserSession` (passed in at
construction) so a multi-step plan acts on the same tab. Session lifecycle
(start/stop) is managed by whoever wires the Tool Registry together, not by
individual tool calls.
"""

from __future__ import annotations

from typing import Any, ClassVar
from urllib.parse import quote_plus, urlparse

from app.core.enums import RiskLevel
from app.core.errors import BrowserError, ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.browser.session import BrowserSession


class _BrowserTool(Tool):
    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    async def _ensure_started(self) -> Any:
        if not self._session.is_started:
            await self._session.start()
        return self._session.page


class OpenUrlTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="open_url",
        description="Open a URL in the browser, starting a session if needed.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = payload.get("url")
        if not url or not isinstance(url, str):
            raise ValidationError("open_url requires a 'url' string")
        # file:// would let the model read local files (then extract_text them),
        # bypassing the file tools' workspace confinement; javascript:/chrome: are
        # no better. Same rule as open_webpage.
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValidationError(f"open_url only supports http/https URLs, got: {url!r}")
        return {"url": url}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        try:
            await page.goto(payload["url"])
        except Exception as exc:  # noqa: BLE001 - normalize any Playwright failure
            raise BrowserError(f"Failed to open '{payload['url']}': {exc}") from exc
        url = payload["url"]
        return ToolResult(success=True, message=f"Opened {url}.", data={"url": url})


class NavigateTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="navigate",
        description="Go back, forward, or reload the current page.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        direction = payload.get("direction")
        if not isinstance(direction, str) or direction not in {"back", "forward", "reload"}:
            raise ValidationError("navigate requires 'direction' to be back/forward/reload")
        return {"direction": direction}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        direction = payload["direction"]
        try:
            if direction == "back":
                await page.go_back()
            elif direction == "forward":
                await page.go_forward()
            else:
                await page.reload()
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"Failed to navigate ({direction}): {exc}") from exc
        return ToolResult(success=True, message=f"Navigated: {direction}.")


class ClickTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="click",
        description="Click an element matching a CSS selector.",
        risk_level=RiskLevel.MEDIUM,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        selector = payload.get("selector")
        if not selector or not isinstance(selector, str):
            raise ValidationError("click requires a 'selector' string")
        return {"selector": selector}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        try:
            await page.click(payload["selector"])
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Couldn't click '{payload['selector']}': {exc}")
        return ToolResult(success=True, message=f"Clicked {payload['selector']}.")


class TypeTextTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="type",
        description="Type text into an element matching a CSS selector.",
        risk_level=RiskLevel.MEDIUM,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        selector, text = payload.get("selector"), payload.get("text")
        if not selector or not isinstance(selector, str) or not isinstance(text, str):
            raise ValidationError("type requires string 'selector' and 'text'")
        return {"selector": selector, "text": text}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        try:
            await page.fill(payload["selector"], payload["text"])
        except Exception as exc:  # noqa: BLE001
            selector = payload["selector"]
            return ToolResult(success=False, error=f"Couldn't type into '{selector}': {exc}")
        return ToolResult(success=True, message=f"Typed into {payload['selector']}.")


class ExtractTextTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="extract_text",
        description="Extract visible text content from the page, or one element.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"selector": payload.get("selector")}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        selector = payload.get("selector")
        try:
            if selector:
                text = await page.inner_text(selector)
            else:
                text = await page.inner_text("body")
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Couldn't extract text: {exc}")
        return ToolResult(success=True, data={"text": text}, message="Extracted page text.")


class BrowserScreenshotTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="browser_screenshot",
        description="Save a screenshot of the current page to a PNG file.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path", "page_screenshot.png")
        if not isinstance(path, str) or not path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValidationError("browser_screenshot only writes .png/.jpg files")
        return {"path": path}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        try:
            await page.screenshot(path=payload["path"])
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Couldn't capture page: {exc}")
        return ToolResult(success=True, message=f"Saved page screenshot to {payload['path']}.")


class SearchWebTool(_BrowserTool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="search_web",
        description="Search the web via DuckDuckGo and return the top result snippets.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query")
        if not query or not isinstance(query, str):
            raise ValidationError("search_web requires a 'query' string")
        return {"query": query}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        page = await self._ensure_started()
        url = f"https://duckduckgo.com/html/?q={quote_plus(payload['query'])}"
        try:
            await page.goto(url)
            text = await page.inner_text("body")
        except Exception as exc:  # noqa: BLE001
            raise BrowserError(f"Search failed: {exc}") from exc
        return ToolResult(success=True, data={"raw_text": text[:4000]}, message="Searched the web.")
