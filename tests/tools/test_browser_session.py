from __future__ import annotations

import pytest

from app.core.errors import BrowserError
from app.tools.browser.session import BrowserSession


class FakePage:
    def __init__(self) -> None:
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed


class FakeBrowser:
    def __init__(self) -> None:
        self.close_called = False
        self.fail_close = False

    async def close(self) -> None:
        self.close_called = True
        if self.fail_close:
            raise RuntimeError("already closed")


class FakePlaywright:
    def __init__(self) -> None:
        self.stop_called = False

    async def stop(self) -> None:
        self.stop_called = True


def test_is_started_false_before_start() -> None:
    session = BrowserSession()
    assert session.is_started is False


def test_is_started_true_after_manually_assigning_a_live_page() -> None:
    session = BrowserSession()
    session._page = FakePage()

    assert session.is_started is True


def test_is_started_false_once_page_reports_closed() -> None:
    session = BrowserSession()
    page = FakePage()
    session._page = page

    assert session.is_started is True
    page.closed = True
    assert session.is_started is False


def test_is_started_false_if_is_closed_raises() -> None:
    session = BrowserSession()

    class BrokenPage:
        def is_closed(self) -> bool:
            raise RuntimeError("connection lost")

    session._page = BrokenPage()

    assert session.is_started is False


def test_page_property_raises_if_not_started() -> None:
    session = BrowserSession()
    with pytest.raises(BrowserError):
        _ = session.page


@pytest.mark.asyncio
async def test_cleanup_stale_closes_old_handles_and_resets_state() -> None:
    session = BrowserSession()
    browser = FakeBrowser()
    playwright = FakePlaywright()
    session._page = FakePage()
    session._browser = browser
    session._playwright = playwright

    await session._cleanup_stale()

    assert browser.close_called is True
    assert playwright.stop_called is True
    assert session._page is None
    assert session._browser is None
    assert session._playwright is None


@pytest.mark.asyncio
async def test_cleanup_stale_swallows_errors_from_already_dead_handles() -> None:
    session = BrowserSession()
    browser = FakeBrowser()
    browser.fail_close = True
    session._browser = browser
    session._playwright = FakePlaywright()
    session._page = FakePage()

    await session._cleanup_stale()  # should not raise

    assert session._page is None


@pytest.mark.asyncio
async def test_stop_is_equivalent_to_cleanup_stale() -> None:
    session = BrowserSession()
    session._page = FakePage()
    session._browser = FakeBrowser()
    session._playwright = FakePlaywright()

    await session.stop()

    assert session.is_started is False


def test_missing_playwright_raises_browser_error_with_install_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "playwright.async_api":
            raise ImportError("no module named playwright")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    session = BrowserSession()
    with pytest.raises(BrowserError) as exc_info:
        asyncio.run(session.start())

    assert "playwright install chromium" in exc_info.value.message
