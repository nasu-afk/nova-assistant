"""Browser session, backed by Playwright.

Playwright itself (and its downloaded browser binary) is imported/launched
lazily — this module is safe to import without the `browser` extra or a
downloaded Chromium build installed; the error only surfaces on first real
use (`BrowserSession.start()`), as `BrowserError`.

Tools in this package (`open_url`, `navigate`, `click`, `type_text`,
`extract_text`, `screenshot`) all operate through one shared `BrowserSession`
so a multi-step plan ("open the page, click login, type the password") acts
on the same browser tab rather than opening a new one per tool call.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import BrowserError


class BrowserSession:
    """Owns one Playwright browser + page, lazily started on first use."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None

    @property
    def is_started(self) -> bool:
        """True only if there's a page AND it's still actually usable.

        Checks `page.is_closed()` (a real, synchronous Playwright API)
        rather than just "was `start()` ever called" — a page can die out
        from under us (the user closes the browser window, it crashes)
        without this object ever being told, and treating a dead page as
        still-started leads to every subsequent call failing with something
        like "Target page, context or browser has been closed" instead of
        transparently recovering.
        """
        if self._page is None:
            return False
        try:
            return not self._page.is_closed()
        except Exception:  # noqa: BLE001 - any failure here means "not usable"
            return False

    async def start(self) -> None:
        if self.is_started:
            return
        # Clean up any stale handles from a previous page/browser that died
        # externally before launching a fresh one — otherwise those old
        # references just leak.
        await self._cleanup_stale()

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - exercised only if dep missing
            raise BrowserError(
                "The 'playwright' package (and its browser binaries) are not "
                "installed. Install with the 'browser' extra and then run "
                "`playwright install chromium`.",
                user_message="Browser automation isn't set up yet.",
            ) from exc

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._page = await self._browser.new_page()

    async def _cleanup_stale(self) -> None:
        """Best-effort cleanup of a previous session's handles that may
        already be dead (e.g. the user closed the browser window) — errors
        here are swallowed since there's nothing meaningful to do about a
        handle that's already gone."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # noqa: BLE001
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001
                pass
        self._page = None
        self._browser = None
        self._playwright = None

    async def stop(self) -> None:
        await self._cleanup_stale()

    @property
    def page(self) -> Any:
        if not self.is_started:
            raise BrowserError("Browser session has not been started.")
        return self._page

    async def __aenter__(self) -> BrowserSession:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.stop()
