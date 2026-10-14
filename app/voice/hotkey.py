"""Push-to-talk global hotkey support.

The combo-matching logic (`_parse_hotkey_string`, `_handle_press`,
`_handle_release`) is deliberately pure and pynput-free, because `pynput`
needs a live display/input backend just to *import* - on Linux it fails at
import time with no X server, and on a real desktop it hooks global input,
neither of which is exercisable in an automated test sandbox. Only
`start()`/`stop()`/the raw pynput callbacks touch the library, imported
lazily, matching every other hardware-facing provider in this codebase.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.core.errors import VoiceError
from app.core.logging import get_logger

logger = get_logger(__name__)

_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "space": "space",
    "cmd": "cmd",
    "super": "cmd",
    "win": "cmd",
}


def parse_hotkey_string(hotkey: str) -> frozenset[str]:
    """Normalize a hotkey string like 'ctrl+space' or 'Ctrl + Space' into a
    frozenset of lowercase key names, e.g. frozenset({'ctrl', 'space'})."""
    parts = [p.strip().lower() for p in hotkey.split("+") if p.strip()]
    if not parts:
        raise VoiceError(f"Empty push-to-talk hotkey: {hotkey!r}")
    return frozenset(_ALIASES.get(p, p) for p in parts)


class PushToTalkListener:
    """Tracks key state and fires `on_start`/`on_stop` when the configured
    combo is fully pressed / released.

    `on_start`/`on_stop` are plain callables (not coroutines) - the caller
    is responsible for bridging into async code (e.g.
    `asyncio.run_coroutine_threadsafe`) if needed, since the real pynput
    listener runs its callbacks on its own OS thread, not the asyncio loop.
    """

    def __init__(
        self, hotkey: str, on_start: Callable[[], None], on_stop: Callable[[], None]
    ) -> None:
        self._target = parse_hotkey_string(hotkey)
        self._pressed: set[str] = set()
        self._active = False
        self._on_start = on_start
        self._on_stop = on_stop
        self._listener: Any = None

    @property
    def is_active(self) -> bool:
        return self._active

    # --- pure logic, exercised directly in tests ---

    def _handle_press(self, name: str) -> None:
        self._pressed.add(name)
        if not self._active and self._target.issubset(self._pressed):
            self._active = True
            self._on_start()

    def _handle_release(self, name: str) -> None:
        self._pressed.discard(name)
        if self._active and not self._target.issubset(self._pressed):
            self._active = False
            self._on_stop()

    # --- pynput glue, only reachable once start() has succeeded ---

    @staticmethod
    def _normalize_pynput_key(key: Any, keyboard_module: Any) -> str:
        keyboard = keyboard_module
        if key in (keyboard.Key.ctrl_l, keyboard.Key.ctrl_r, keyboard.Key.ctrl):
            return "ctrl"
        if key in (keyboard.Key.shift_l, keyboard.Key.shift_r, keyboard.Key.shift):
            return "shift"
        if key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr, keyboard.Key.alt):
            return "alt"
        if key == keyboard.Key.space:
            return "space"
        if key in (keyboard.Key.cmd_l, keyboard.Key.cmd_r, keyboard.Key.cmd):
            return "cmd"
        char = getattr(key, "char", None)
        return char.lower() if char else str(key).lower()

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - exercised only if dep/display missing
            raise VoiceError(
                "Push-to-talk requires 'pynput' and a live display/input backend "
                "(install the 'voice' extra: `uv sync --extra voice`).",
                user_message="Push-to-talk isn't available in this environment.",
            ) from exc

        # pynput stops delivering events if a callback raises, which would
        # silently disable push-to-talk for the rest of the session.
        def on_press(key: Any) -> None:
            try:
                self._handle_press(self._normalize_pynput_key(key, keyboard))
            except Exception as exc:  # noqa: BLE001
                logger.error("push_to_talk_press_failed", error=str(exc))

        def on_release(key: Any) -> None:
            try:
                self._handle_release(self._normalize_pynput_key(key, keyboard))
            except Exception as exc:  # noqa: BLE001
                logger.error("push_to_talk_release_failed", error=str(exc))

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        logger.info("push_to_talk_listening", hotkey=sorted(self._target))

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
