"""Search YouTube and play the first result - a single, reliable tool call.

Why this exists instead of exposing generic click/type primitives for the
LLM to drive: `open_webpage` (stdlib `webbrowser.open`) launches the user's
actual default browser as an untracked OS process - NOVA has no handle to
that window and cannot click inside it. And even with a Playwright-controlled
browser NOVA *does* have a handle to, asking the LLM to guess "click the
first result"'s CSS selector on a page it has never seen is unreliable by
construction. This tool instead hardcodes the one sequence that's actually
needed - go to YouTube search results, click the first video thumbnail,
optionally go fullscreen - as a single atomic action.

The browser this opens is Playwright-controlled and separate from the
user's regular browser (and from whatever `open_webpage` opened) - it needs
`playwright install chromium` to have been run once (see the `browser`
extra). It's launched non-headless (`headless=False`) so the user actually
sees the video playing, unlike `app.tools.browser.session.BrowserSession`'s
default of headless=True for automation/scraping use.

Ad handling: this only ever clicks YouTube's own native "Skip Ad" button
when it's already visible and skippable — the same thing a person would do
by hand. It does not block ad requests, modify the page to hide/remove ads,
or attempt to skip *unskippable* ads; that would cross from "automating a
click a user could make themselves" into ad-blocking, which is a different
thing this project isn't taking a position on.
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar
from urllib.parse import quote_plus

from app.core.enums import RiskLevel
from app.core.errors import BrowserError, ValidationError
from app.core.logging import get_logger
from app.tools.base import Tool, ToolMetadata, ToolResult
from app.tools.browser.session import BrowserSession

logger = get_logger(__name__)

# YouTube's search-result video links use this pattern as of writing. If
# YouTube changes its markup this selector will need updating - that's a
# real, known fragility of scraping a third-party site's DOM, not something
# that can be made robust from here.
_FIRST_RESULT_SELECTOR = "ytd-video-renderer a#video-title, a#video-title"

# YouTube has used a few different class names for the skip button over the
# years; trying all of them (comma = "or" in a CSS selector) makes this a
# little more resilient to markup changes than hardcoding just one.
_SKIP_AD_SELECTOR = (
    ".ytp-skip-ad-button, .ytp-ad-skip-button, "
    "button.ytp-ad-skip-button-modern, .ytp-ad-skip-button-slot"
)


class PlayYoutubeVideoTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="play_youtube_video",
        description=(
            "Search YouTube for a song/video and play the first result in a visible "
            "browser window, optionally fullscreen. Automatically clicks YouTube's own "
            "'Skip Ad' button whenever one appears, for as long as the video keeps "
            "playing. Use this instead of open_webpage when the user wants a video "
            "actually playing, not just a search page."
        ),
        risk_level=RiskLevel.LOW,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for on YouTube"},
                "fullscreen": {
                    "type": "boolean",
                    "description": "Whether to fullscreen the video once playing (default true)",
                },
            },
            "required": ["query"],
        },
    )

    def __init__(
        self, session: BrowserSession | None = None, ad_check_interval: float = 1.0
    ) -> None:
        # A dedicated non-headless session - separate from any shared
        # automation session - so the user actually sees the video window.
        self._session = session or BrowserSession(headless=False)
        self._ad_check_interval = ad_check_interval
        self._ad_skip_task: asyncio.Task[None] | None = None

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        query = payload.get("query")
        if not query or not isinstance(query, str):
            raise ValidationError("play_youtube_video requires a 'query' string")
        return {"query": query, "fullscreen": bool(payload.get("fullscreen", True))}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        if not self._session.is_started:
            await self._session.start()
        page = self._session.page

        query = payload["query"]
        search_url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        try:
            await page.goto(search_url)
            await page.wait_for_selector(_FIRST_RESULT_SELECTOR, timeout=10_000)
            await page.click(_FIRST_RESULT_SELECTOR)
        except Exception as exc:  # noqa: BLE001 - normalize any Playwright/selector failure
            raise BrowserError(
                f"Couldn't find or click a YouTube result for {query!r}: {exc}",
                user_message=(
                    f"I opened YouTube and searched for '{query}', but couldn't "
                    "click a result automatically - YouTube's page layout may have changed."
                ),
            ) from exc

        if payload["fullscreen"]:
            try:
                # Pressing "f" while the watch page is still loading goes nowhere.
                await page.wait_for_selector("video", timeout=10_000)
                await page.keyboard.press("f")
            except Exception:  # noqa: BLE001 - fullscreen is best-effort, never fails the tool
                pass

        # A new video is playing - stop watching for ads on whatever was
        # playing before and start a fresh watcher for this one.
        if self._ad_skip_task is not None and not self._ad_skip_task.done():
            self._ad_skip_task.cancel()
        self._ad_skip_task = asyncio.create_task(self._auto_skip_ads_loop(page))

        return ToolResult(
            success=True,
            message=f"Playing the first YouTube result for '{query}'. Ads will be auto-skipped.",
            data={"query": query},
        )

    async def aclose(self) -> None:
        """Stop the ad watcher and close the browser (called at NOVA shutdown)."""
        task = self._ad_skip_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._ad_skip_task = None
        try:
            await self._session.stop()
        except Exception:  # noqa: BLE001 - shutting down; nothing more to do
            pass

    async def _try_skip_ad_once(self, page: Any) -> bool:
        """Click the Skip Ad button if it's currently visible. Returns True
        if a click happened. Never raises — a missing/gone button, or the
        page having navigated away, is just "nothing to skip right now"."""
        try:
            button = await page.query_selector(_SKIP_AD_SELECTOR)
        except Exception:  # noqa: BLE001 - page may have navigated/closed
            return False
        if button is None:
            return False
        try:
            await button.click()
        except Exception:  # noqa: BLE001 - button may have disappeared between find and click
            return False
        logger.info("youtube_ad_skipped")
        return True

    async def _auto_skip_ads_loop(self, page: Any) -> None:
        """Poll for YouTube's Skip Ad button for as long as this video's tab
        is open, clicking it whenever it appears. Runs as a background task
        (see `execute`) so the tool call itself returns immediately rather
        than blocking for the video's entire runtime."""
        while True:
            try:
                await self._try_skip_ad_once(page)
                await asyncio.sleep(self._ad_check_interval)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - tab/browser closed, etc. - stop quietly
                return
