from __future__ import annotations

from unittest.mock import patch

import pytest

from app.core.errors import ValidationError
from app.tools.web.browser_launch import OpenWebpageTool


def test_requires_url() -> None:
    with pytest.raises(ValidationError):
        OpenWebpageTool().validate_input({})


def test_rejects_non_http_scheme() -> None:
    with pytest.raises(ValidationError):
        OpenWebpageTool().validate_input({"url": "file:///etc/passwd"})


def test_accepts_http_and_https() -> None:
    tool = OpenWebpageTool()
    assert tool.validate_input({"url": "http://example.com"})["url"] == "http://example.com"
    assert tool.validate_input({"url": "https://example.com"})["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_execute_opens_browser_successfully() -> None:
    tool = OpenWebpageTool()
    with patch("webbrowser.open", return_value=True) as mock_open:
        result = await tool.execute({"url": "https://youtube.com"})

    mock_open.assert_called_once_with("https://youtube.com")
    assert result.success is True
    assert result.data["url"] == "https://youtube.com"


@pytest.mark.asyncio
async def test_execute_reports_failure_when_no_browser_available() -> None:
    tool = OpenWebpageTool()
    with patch("webbrowser.open", return_value=False):
        result = await tool.execute({"url": "https://youtube.com"})

    assert result.success is False
    assert result.error is not None


@pytest.mark.asyncio
async def test_execute_wraps_exceptions_as_failure() -> None:
    tool = OpenWebpageTool()
    with patch("webbrowser.open", side_effect=RuntimeError("no display")):
        result = await tool.execute({"url": "https://youtube.com"})

    assert result.success is False
    assert "no display" in (result.error or "")
