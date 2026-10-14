from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import BrowserError, ValidationError
from app.tools.browser.session import BrowserSession
from app.tools.browser.tools import (
    ClickTool,
    ExtractTextTool,
    NavigateTool,
    OpenUrlTool,
    SearchWebTool,
    TypeTextTool,
)


class FakePage:
    def __init__(self) -> None:
        self.visited: list[str] = []
        self.filled: dict[str, str] = {}
        self.clicked: list[str] = []
        self.history: list[str] = []
        self.fail_selector: str | None = None
        self.closed = False

    def is_closed(self) -> bool:
        return self.closed

    async def goto(self, url: str) -> None:
        self.visited.append(url)

    async def go_back(self) -> None:
        self.history.append("back")

    async def go_forward(self) -> None:
        self.history.append("forward")

    async def reload(self) -> None:
        self.history.append("reload")

    async def click(self, selector: str) -> None:
        if selector == self.fail_selector:
            raise RuntimeError("element not found")
        self.clicked.append(selector)

    async def fill(self, selector: str, text: str) -> None:
        self.filled[selector] = text

    async def inner_text(self, selector: str) -> str:
        return f"text of {selector}"


class FakeBrowserSession(BrowserSession):
    """Bypasses real Playwright entirely — injects a FakePage instead."""

    def __init__(self, page: FakePage) -> None:
        super().__init__()
        self._fake_page = page

    async def start(self) -> None:
        self._page = self._fake_page

    @property
    def page(self) -> Any:
        return self._fake_page


@pytest.fixture
def session() -> FakeBrowserSession:
    return FakeBrowserSession(FakePage())


@pytest.mark.asyncio
async def test_open_url_navigates_and_starts_session(session: FakeBrowserSession) -> None:
    tool = OpenUrlTool(session)
    payload = tool.validate_input({"url": "https://example.com"})

    result = await tool.execute(payload)

    assert result.success is True
    assert session.page.visited == ["https://example.com"]


def test_open_url_requires_url() -> None:
    with pytest.raises(ValidationError):
        OpenUrlTool(FakeBrowserSession(FakePage())).validate_input({})


@pytest.mark.asyncio
async def test_navigate_back_forward_reload(session: FakeBrowserSession) -> None:
    tool = NavigateTool(session)

    await tool.execute(tool.validate_input({"direction": "back"}))
    await tool.execute(tool.validate_input({"direction": "forward"}))
    await tool.execute(tool.validate_input({"direction": "reload"}))

    assert session.page.history == ["back", "forward", "reload"]


def test_navigate_rejects_invalid_direction() -> None:
    with pytest.raises(ValidationError):
        NavigateTool(FakeBrowserSession(FakePage())).validate_input({"direction": "sideways"})


@pytest.mark.asyncio
async def test_click_succeeds(session: FakeBrowserSession) -> None:
    tool = ClickTool(session)
    result = await tool.execute(tool.validate_input({"selector": "#login"}))

    assert result.success is True
    assert session.page.clicked == ["#login"]


@pytest.mark.asyncio
async def test_click_missing_element_returns_failure_not_exception(
    session: FakeBrowserSession,
) -> None:
    session.page.fail_selector = "#missing"
    tool = ClickTool(session)

    result = await tool.execute(tool.validate_input({"selector": "#missing"}))

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_type_text_fills_field(session: FakeBrowserSession) -> None:
    tool = TypeTextTool(session)
    payload = tool.validate_input({"selector": "#email", "text": "hasan@example.com"})
    result = await tool.execute(payload)

    assert result.success is True
    assert session.page.filled["#email"] == "hasan@example.com"


@pytest.mark.asyncio
async def test_extract_text_defaults_to_body(session: FakeBrowserSession) -> None:
    tool = ExtractTextTool(session)
    result = await tool.execute(tool.validate_input({}))

    assert result.success is True
    assert result.data["text"] == "text of body"


@pytest.mark.asyncio
async def test_extract_text_with_selector(session: FakeBrowserSession) -> None:
    tool = ExtractTextTool(session)
    result = await tool.execute(tool.validate_input({"selector": "#main"}))

    assert result.data["text"] == "text of #main"


@pytest.mark.asyncio
async def test_search_web_goes_to_duckduckgo(session: FakeBrowserSession) -> None:
    tool = SearchWebTool(session)
    result = await tool.execute(tool.validate_input({"query": "nova assistant"}))

    assert result.success is True
    assert "duckduckgo.com" in session.page.visited[0]
    assert "nova+assistant" in session.page.visited[0]


@pytest.mark.asyncio
async def test_open_url_wraps_playwright_errors_as_browser_error() -> None:
    class ExplodingPage(FakePage):
        async def goto(self, url: str) -> None:
            raise RuntimeError("net::ERR_CONNECTION_REFUSED")

    session = FakeBrowserSession(ExplodingPage())
    tool = OpenUrlTool(session)

    with pytest.raises(BrowserError):
        await tool.execute(tool.validate_input({"url": "https://down.example"}))


def test_session_requires_start_before_page_access() -> None:
    session = BrowserSession()
    with pytest.raises(BrowserError):
        _ = session.page
