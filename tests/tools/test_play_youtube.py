from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest

from app.core.errors import BrowserError, ValidationError
from app.tools.browser.session import BrowserSession
from app.tools.web.youtube import PlayYoutubeVideoTool


class FakeKeyboard:
    def __init__(self) -> None:
        self.pressed: list[str] = []

    async def press(self, key: str) -> None:
        self.pressed.append(key)


class FakeElement:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.clicked = False

    async def click(self) -> None:
        if self.fail:
            raise RuntimeError("element vanished before click")
        self.clicked = True


class FakePage:
    def __init__(self) -> None:
        self.visited: list[str] = []
        self.clicked: list[str] = []
        self.waited_for: list[str] = []
        self.keyboard = FakeKeyboard()
        self.fail_wait = False
        self.fail_click = False
        self.skip_button: FakeElement | None = None
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def goto(self, url: str) -> None:
        if self.closed:
            raise RuntimeError("Page.goto: Target page, context or browser has been closed")
        self.visited.append(url)

    async def wait_for_selector(self, selector: str, timeout: int = 10_000) -> None:  # noqa: ASYNC109
        if self.fail_wait:
            raise TimeoutError("no results found")
        self.waited_for.append(selector)

    async def click(self, selector: str) -> None:
        if self.fail_click:
            raise RuntimeError("element not clickable")
        self.clicked.append(selector)

    async def query_selector(self, selector: str) -> FakeElement | None:
        return self.skip_button


class FakeBrowserSession(BrowserSession):
    def __init__(self, page: FakePage) -> None:
        super().__init__()
        self._fake_page = page

    async def start(self) -> None:
        # Simulates a real restart: a freshly-started session always gives
        # you a working page, even if the previous one had died.
        self._fake_page.closed = False
        self._page = self._fake_page

    @property
    def page(self) -> Any:
        return self._fake_page


@pytest.fixture
def session() -> FakeBrowserSession:
    return FakeBrowserSession(FakePage())


@pytest.fixture
async def tool(session: FakeBrowserSession):  # type: ignore[no-untyped-def]
    t = PlayYoutubeVideoTool(session, ad_check_interval=0.01)
    yield t
    # Clean up any background ad-skip task so tests don't leak pending tasks.
    if t._ad_skip_task is not None:
        t._ad_skip_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await t._ad_skip_task


def test_requires_query(tool: PlayYoutubeVideoTool) -> None:
    with pytest.raises(ValidationError):
        tool.validate_input({})


def test_fullscreen_defaults_to_true(tool: PlayYoutubeVideoTool) -> None:
    payload = tool.validate_input({"query": "test song"})
    assert payload["fullscreen"] is True


@pytest.mark.asyncio
async def test_searches_and_clicks_first_result(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    payload = tool.validate_input({"query": "Senorita ZNMD"})

    result = await tool.execute(payload)

    assert result.success is True
    assert "youtube.com/results" in session.page.visited[0]
    assert "Senorita+ZNMD" in session.page.visited[0]
    assert session.page.clicked  # clicked the first-result selector
    assert session.page.keyboard.pressed == ["f"]  # fullscreen attempted


@pytest.mark.asyncio
async def test_fullscreen_false_skips_keypress(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    payload = tool.validate_input({"query": "lofi", "fullscreen": False})

    await tool.execute(payload)

    assert session.page.keyboard.pressed == []


@pytest.mark.asyncio
async def test_no_results_found_raises_browser_error_with_helpful_message(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    session.page.fail_wait = True

    with pytest.raises(BrowserError) as exc_info:
        await tool.execute(tool.validate_input({"query": "asdkjaskldj"}))

    assert "asdkjaskldj" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_click_failure_raises_browser_error(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    session.page.fail_click = True

    with pytest.raises(BrowserError):
        await tool.execute(tool.validate_input({"query": "test"}))


@pytest.mark.asyncio
async def test_fullscreen_keypress_failure_does_not_fail_the_tool(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    async def broken_press(key: str) -> None:
        raise RuntimeError("no active player")

    session.page.keyboard.press = broken_press  # type: ignore[method-assign]

    result = await tool.execute(tool.validate_input({"query": "test"}))

    assert result.success is True


@pytest.mark.asyncio
async def test_starts_session_if_not_already_started(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    assert session.is_started is False

    await tool.execute(tool.validate_input({"query": "test"}))

    assert session.is_started is True


@pytest.mark.asyncio
async def test_recovers_automatically_when_browser_was_closed_externally(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    # First call succeeds normally.
    await tool.execute(tool.validate_input({"query": "first song"}))
    assert session.is_started is True

    # Simulate the user closing the browser window (or it crashing) between
    # calls — the session object doesn't know until it actually checks.
    session.page.closed = True
    assert session.is_started is False

    # The next call should transparently restart rather than failing with
    # "Target page, context or browser has been closed", which is exactly
    # what happened before BrowserSession.is_started checked page.is_closed().
    result = await tool.execute(tool.validate_input({"query": "second song"}))

    assert result.success is True
    assert session.is_started is True


# --- ad-skipping ---


@pytest.mark.asyncio
async def test_try_skip_ad_once_clicks_visible_skip_button(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    button = FakeElement()
    session.page.skip_button = button

    clicked = await tool._try_skip_ad_once(session.page)

    assert clicked is True
    assert button.clicked is True


@pytest.mark.asyncio
async def test_try_skip_ad_once_returns_false_when_no_button(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    session.page.skip_button = None

    clicked = await tool._try_skip_ad_once(session.page)

    assert clicked is False


@pytest.mark.asyncio
async def test_try_skip_ad_once_swallows_click_failure(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    session.page.skip_button = FakeElement(fail=True)

    clicked = await tool._try_skip_ad_once(session.page)  # should not raise

    assert clicked is False


@pytest.mark.asyncio
async def test_execute_starts_a_background_ad_skip_task(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    await tool.execute(tool.validate_input({"query": "test"}))

    assert tool._ad_skip_task is not None
    assert tool._ad_skip_task.done() is False


@pytest.mark.asyncio
async def test_ad_skip_loop_clicks_button_when_it_appears(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    await tool.execute(tool.validate_input({"query": "test"}))

    button = FakeElement()
    session.page.skip_button = button
    await asyncio.sleep(0.05)  # let the background loop poll at least once

    assert button.clicked is True


@pytest.mark.asyncio
async def test_execute_cancels_previous_ad_skip_task_for_new_video(
    tool: PlayYoutubeVideoTool, session: FakeBrowserSession
) -> None:
    await tool.execute(tool.validate_input({"query": "first song"}))
    first_task = tool._ad_skip_task

    await tool.execute(tool.validate_input({"query": "second song"}))
    await asyncio.sleep(0)  # let the cancellation actually propagate

    assert first_task is not None
    assert first_task.cancelled() or first_task.done()
    assert tool._ad_skip_task is not first_task
