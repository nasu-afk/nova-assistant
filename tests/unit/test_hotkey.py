from __future__ import annotations

import builtins

import pytest

from app.core.errors import VoiceError
from app.voice.hotkey import PushToTalkListener, parse_hotkey_string


def test_parse_hotkey_string_combo() -> None:
    assert parse_hotkey_string("ctrl+space") == frozenset({"ctrl", "space"})


def test_parse_hotkey_string_normalizes_aliases_and_case() -> None:
    assert parse_hotkey_string("Control + Space") == frozenset({"ctrl", "space"})
    assert parse_hotkey_string("option+a") == frozenset({"alt", "a"})
    assert parse_hotkey_string("win+shift") == frozenset({"cmd", "shift"})


def test_parse_hotkey_string_single_key() -> None:
    assert parse_hotkey_string("space") == frozenset({"space"})


def test_parse_hotkey_string_empty_raises() -> None:
    with pytest.raises(VoiceError):
        parse_hotkey_string("")


def test_press_full_combo_triggers_start() -> None:
    started = []
    listener = PushToTalkListener(
        "ctrl+space", on_start=lambda: started.append(1), on_stop=lambda: None
    )

    listener._handle_press("ctrl")
    assert started == []  # not yet - only one of two keys down
    listener._handle_press("space")

    assert started == [1]
    assert listener.is_active is True


def test_releasing_one_key_of_combo_triggers_stop() -> None:
    stopped = []
    listener = PushToTalkListener(
        "ctrl+space", on_start=lambda: None, on_stop=lambda: stopped.append(1)
    )
    listener._handle_press("ctrl")
    listener._handle_press("space")

    listener._handle_release("space")

    assert stopped == [1]
    assert listener.is_active is False


def test_start_does_not_fire_twice_while_held() -> None:
    calls = []
    listener = PushToTalkListener(
        "ctrl+space", on_start=lambda: calls.append(1), on_stop=lambda: None
    )
    listener._handle_press("ctrl")
    listener._handle_press("space")
    listener._handle_press("space")  # key-repeat while held

    assert calls == [1]


def test_unrelated_keys_are_ignored() -> None:
    started = []
    listener = PushToTalkListener(
        "ctrl+space", on_start=lambda: started.append(1), on_stop=lambda: None
    )

    listener._handle_press("a")
    listener._handle_press("shift")

    assert started == []
    assert listener.is_active is False


def test_single_key_hotkey_triggers_immediately() -> None:
    events = []
    listener = PushToTalkListener(
        "space", on_start=lambda: events.append("start"), on_stop=lambda: events.append("stop")
    )

    listener._handle_press("space")
    listener._handle_release("space")

    assert events == ["start", "stop"]


def test_start_raises_voice_error_without_pynput_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pynput" or name.startswith("pynput."):
            raise ImportError("pynput backend unavailable")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    listener = PushToTalkListener("ctrl+space", on_start=lambda: None, on_stop=lambda: None)

    with pytest.raises(VoiceError):
        listener.start()


def test_stop_without_start_is_a_no_op() -> None:
    listener = PushToTalkListener("ctrl+space", on_start=lambda: None, on_stop=lambda: None)
    listener.stop()  # should not raise
