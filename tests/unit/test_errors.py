from __future__ import annotations

import pytest

from app.core.errors import (
    ConfirmationRequiredError,
    NovaError,
    PermissionDeniedError,
    ToolExecutionError,
    ValidationError,
)


@pytest.mark.parametrize(
    "error_cls",
    [ValidationError, ToolExecutionError, PermissionDeniedError, ConfirmationRequiredError],
)
def test_all_nova_errors_are_nova_error_subclasses(error_cls: type[NovaError]) -> None:
    assert issubclass(error_cls, NovaError)
    assert issubclass(error_cls, Exception)


def test_nova_error_defaults_user_message_to_message() -> None:
    err = ToolExecutionError("stack trace: boom at line 42")

    assert err.message == "stack trace: boom at line 42"
    assert err.user_message == "stack trace: boom at line 42"


def test_nova_error_can_have_separate_user_message() -> None:
    err = ToolExecutionError(
        "Maven exited 1: java.lang.UnsupportedClassVersionError",
        user_message="I couldn't build the project because of a Java version mismatch.",
    )

    assert "UnsupportedClassVersionError" in err.message
    assert "UnsupportedClassVersionError" not in err.user_message
    assert str(err) == err.message
