from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox

from app.core.enums import EventType
from app.events.bus import Event, EventBus
from app.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def window(qapp: QApplication) -> MainWindow:
    return MainWindow()


def test_window_has_expected_title(window: MainWindow) -> None:
    assert window.windowTitle() == "NOVA"


def test_set_status_updates_label(window: MainWindow) -> None:
    window.set_status("Listening")
    assert window._status_label.text() == "LISTENING"


def test_set_mic_status_reflects_listening_state(window: MainWindow) -> None:
    window.set_mic_status(True)
    assert "listening" in window._mic_label.text()
    window.set_mic_status(False)
    assert "off" in window._mic_label.text()


def test_append_transcript_adds_text(window: MainWindow) -> None:
    window.append_transcript("open notepad")
    assert "open notepad" in window._transcript_view.toPlainText()


def test_append_conversation_shows_role_and_text(window: MainWindow) -> None:
    window.append_conversation("user", "hello")
    assert "hello" in window._conversation_view.toPlainText()


def test_set_task_shows_goal_and_progress(window: MainWindow) -> None:
    window.set_task("Open VS Code")
    assert "Open VS Code" in window._task_label.text()
    assert window._progress_bar.isHidden() is False

    window.set_task(None)
    assert window._task_label.text() == "No active task"
    assert window._progress_bar.isHidden() is True


def test_append_tool_activity_adds_list_item(window: MainWindow) -> None:
    window.append_tool_activity("Started: open_application")
    assert window._tool_activity_list.count() == 1
    assert window._tool_activity_list.item(0).text() == "Started: open_application"


def test_append_log_adds_list_item(window: MainWindow) -> None:
    window.append_log("nova_starting")
    assert window._log_list.count() == 1


def test_set_memory_snapshot_updates_view(window: MainWindow) -> None:
    window.set_memory_snapshot("favorite_editor: VS Code")
    assert "VS Code" in window._memory_view.toPlainText()


def test_confirmation_dialog_returns_true_on_yes(window: MainWindow) -> None:
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.Yes):
        assert window.show_confirmation_dialog("delete_file", {"path": "/tmp/x"}) is True


def test_confirmation_dialog_returns_false_on_no(window: MainWindow) -> None:
    with patch.object(QMessageBox, "question", return_value=QMessageBox.StandardButton.No):
        assert window.show_confirmation_dialog("delete_file", {"path": "/tmp/x"}) is False


@pytest.mark.asyncio
async def test_wire_events_updates_task_and_status(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.TASK_CREATED, payload={"goal": "Open VS Code"}))
    assert "Open VS Code" in window._task_label.text()
    assert window._status_label.text() == "WORKING"

    await bus.publish(Event(type=EventType.TASK_COMPLETED))
    assert window._task_label.text() == "No active task"
    assert window._status_label.text() == "IDLE"


@pytest.mark.asyncio
async def test_wire_events_updates_tool_activity(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.TOOL_STARTED, payload={"tool": "open_application"}))

    assert window._tool_activity_list.count() == 1
    assert "open_application" in window._tool_activity_list.item(0).text()


@pytest.mark.asyncio
async def test_wire_events_updates_mic_status(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.VOICE_STARTED))
    assert "listening" in window._mic_label.text()

    await bus.publish(Event(type=EventType.VOICE_STOPPED))
    assert "off" in window._mic_label.text()


@pytest.mark.asyncio
async def test_wire_events_appends_transcript(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.TRANSCRIPT_CREATED, payload={"text": "open notepad"}))

    assert "open notepad" in window._transcript_view.toPlainText()


@pytest.mark.asyncio
async def test_wire_events_handles_task_failure(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.TASK_FAILED))

    assert window._task_label.text() == "No active task"
    assert "FAILED" in window._status_label.text()


@pytest.mark.asyncio
async def test_wire_events_voice_started_updates_status_to_listening(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.VOICE_STARTED))

    assert window._status_label.text() == "LISTENING"


@pytest.mark.asyncio
async def test_wire_events_voice_stopped_updates_status_to_thinking(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.VOICE_STOPPED))

    assert window._status_label.text() == "THINKING"


@pytest.mark.asyncio
async def test_wire_events_transcript_also_appends_to_conversation(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.TRANSCRIPT_CREATED, payload={"text": "open notepad"}))

    assert "open notepad" in window._conversation_view.toPlainText()
    assert "You" in window._conversation_view.toPlainText()


@pytest.mark.asyncio
async def test_wire_events_response_ready_appends_conversation_and_sets_speaking(
    window: MainWindow,
) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(Event(type=EventType.RESPONSE_READY, payload={"text": "Opened Notepad."}))

    assert "Opened Notepad." in window._conversation_view.toPlainText()
    assert "NOVA" in window._conversation_view.toPlainText()
    assert window._status_label.text() == "SPEAKING"


@pytest.mark.asyncio
async def test_wire_events_tool_completed_prefers_tool_name_over_step_id(
    window: MainWindow,
) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(
        Event(type=EventType.TOOL_COMPLETED, payload={"tool": "open_application", "step_id": "s1"})
    )

    text = window._tool_activity_list.item(0).text()
    assert "open_application" in text
    assert "s1" not in text


@pytest.mark.asyncio
async def test_wire_events_tool_failed_shows_tool_name_and_error(window: MainWindow) -> None:
    bus = EventBus()
    window.wire_events(bus)

    await bus.publish(
        Event(type=EventType.TOOL_FAILED, payload={"tool": "delete_file", "error": "not found"})
    )

    text = window._tool_activity_list.item(0).text()
    assert "delete_file" in text
    assert "not found" in text


def test_create_application_builds_window_without_showing_or_exec() -> None:
    from app.ui.app import create_application

    application, window = create_application([])

    assert isinstance(window, MainWindow)
    assert application is not None


# --- visual styling behavior ---


def test_set_status_uppercases_text_for_hud_look(window: MainWindow) -> None:
    window.set_status("Listening")
    assert window._status_label.text() == "LISTENING"


def test_set_status_colors_by_state(window: MainWindow) -> None:
    window.set_status("Listening")
    assert "#00e5ff" in window._status_label.styleSheet()

    window.set_status("Thinking")
    assert "#ffb300" in window._status_label.styleSheet()

    window.set_status("Speaking")
    assert "#9d7bff" in window._status_label.styleSheet()


def test_set_status_error_states_turn_red(window: MainWindow) -> None:
    window.set_status("Microphone unavailable")
    assert "#ff4d6d" in window._status_label.styleSheet()

    window.set_status("Idle (last task failed)")
    assert "#ff4d6d" in window._status_label.styleSheet()


def test_status_dot_follows_status_color(window: MainWindow) -> None:
    window.set_status("Thinking")
    assert "#ffb300" in window._status_dot.styleSheet()


def test_unknown_status_falls_back_to_idle_color(window: MainWindow) -> None:
    window.set_status("Something unexpected")
    assert "#8ba0c2" in window._status_label.styleSheet()


def test_tool_activity_lines_are_color_coded(window: MainWindow) -> None:
    window.append_tool_activity("Started: open_application")
    window.append_tool_activity("Completed: open_application")
    window.append_tool_activity("Failed (open_application): boom")

    colors = [window._tool_activity_list.item(i).foreground().color().name() for i in range(3)]
    assert colors == ["#00e5ff", "#39ff9c", "#ff4d6d"]


def test_conversation_text_is_html_escaped(window: MainWindow) -> None:
    window.append_conversation("NOVA", "use <b>bold</b> & more")

    plain = window._conversation_view.toPlainText()
    assert "<b>bold</b>" in plain
    assert "& more" in plain


def test_conversation_roles_get_distinct_colors(window: MainWindow) -> None:
    window.append_conversation("You", "hi")
    window.append_conversation("NOVA", "hello")

    html_text = window._conversation_view.toHtml().lower()
    assert "#00e5ff" in html_text
    assert "#9d7bff" in html_text


def test_closing_window_emits_closed_signal(window: MainWindow) -> None:
    received: list[bool] = []
    window.closed.connect(lambda: received.append(True))

    window.show()
    window.close()

    assert received == [True]


def test_confirmation_dialog_shortens_huge_payloads(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    def fake_question(parent: object, title: str, text: str, *args: object) -> object:
        captured["text"] = text
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    window.show_confirmation_dialog("write_file", {"path": "a.txt", "content": "z" * 10_000})

    assert len(captured["text"]) < 600


# --------------------------------------------------------------- accessibility
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QShortcut  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402

import app.ui.main_window as mw  # noqa: E402


def test_on_screen_talk_button_emits_press_and_release(window: MainWindow) -> None:
    """The HOLD TO TALK button used to be connected to nothing at all."""
    events: list[str] = []
    window.talk_pressed.connect(lambda: events.append("down"))
    window.talk_released.connect(lambda: events.append("up"))
    window.show()

    QTest.mousePress(window._push_to_talk_button, Qt.MouseButton.LeftButton)
    QTest.mouseRelease(window._push_to_talk_button, Qt.MouseButton.LeftButton)

    assert events == ["down", "up"]


def test_talk_button_works_from_the_keyboard(window: MainWindow) -> None:
    events: list[str] = []
    window.talk_pressed.connect(lambda: events.append("down"))
    window.talk_released.connect(lambda: events.append("up"))
    window.show()
    window._push_to_talk_button.setFocus()

    QTest.keyPress(window._push_to_talk_button, Qt.Key.Key_Space)
    QTest.keyRelease(window._push_to_talk_button, Qt.Key.Key_Space)

    assert events == ["down", "up"]


def test_talk_button_is_tab_reachable_but_not_click_focusable(window: MainWindow) -> None:
    # Click-focus would make the global Ctrl+Space hotkey also press this button.
    assert window._push_to_talk_button.focusPolicy() == Qt.FocusPolicy.TabFocus


def test_typed_request_is_emitted_and_input_cleared(window: MainWindow) -> None:
    received: list[tuple[str, bool]] = []
    window.text_submitted.connect(lambda text, speak: received.append((text, speak)))

    window._input.setText("  open notepad  ")
    QTest.keyClick(window._input, Qt.Key.Key_Return)
    window._speak_checkbox.setChecked(True)
    window._input.setText("what time is it")
    window._send_button.click()

    assert received == [("open notepad", False), ("what time is it", True)]
    assert window._input.text() == ""


def test_blank_typed_request_is_ignored(window: MainWindow) -> None:
    received: list[object] = []
    window.text_submitted.connect(lambda *args: received.append(args))

    window._input.setText("   ")
    window._send_button.click()

    assert received == []


def test_every_interactive_widget_has_an_accessible_name(window: MainWindow) -> None:
    widgets = [
        window._push_to_talk_button,
        window._input,
        window._send_button,
        window._speak_checkbox,
        window._status_label,
        window._mic_label,
        window._transcript_view,
        window._conversation_view,
        window._tool_activity_list,
        window._log_list,
        window._memory_view,
        window._progress_bar,
        window._tabs,
    ]
    for widget in widgets:
        assert widget.accessibleName().strip(), type(widget).__name__
    # Controls the user acts on also explain themselves.
    for widget in (window._push_to_talk_button, window._input, window._speak_checkbox):
        assert widget.accessibleDescription() and widget.toolTip()


def test_status_and_replies_are_exposed_to_screen_readers(window: MainWindow) -> None:
    window.set_status("Thinking")
    assert window._status_label.accessibleName() == "Status: THINKING"

    window.append_conversation("NOVA", "It is noon.")
    assert "NOVA says: It is noon." in window._status_label.accessibleName()

    window.set_status("Idle")  # the next status change restores the normal name
    assert window._status_label.accessibleName() == "Status: IDLE"


def _luminance(hex_color: str) -> float:
    r, g, b = (int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4))

    def channel(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_palette_meets_wcag_aa_contrast() -> None:
    """Text colours must reach 4.5:1 on every surface they sit on (two of them
    were at 3.8-4.2:1 before)."""
    foregrounds = {
        "text": mw._TEXT,
        "dim": mw._TEXT_DIM,
        "cyan": mw._CYAN,
        "violet": mw._VIOLET,
        "success": mw._SUCCESS,
        **{f"status-{k}": v for k, v in mw._STATUS_COLORS.items()},
    }
    for name, color in foregrounds.items():
        for surface in (mw._BG, mw._PANEL, mw._PANEL_ALT):
            ratio = _contrast(color, surface)
            assert ratio >= 4.5, f"{name} {color} on {surface}: {ratio:.1f}:1"


def test_text_size_can_be_increased_decreased_and_reset(window: MainWindow) -> None:
    base = window.styleSheet()
    window.increase_text_size()
    assert window.text_scale == pytest.approx(1.1)
    assert "font-size: 14px" in window.styleSheet()  # 13px body text -> 14px
    assert window.styleSheet() != base

    for _ in range(30):
        window.increase_text_size()
    assert window.text_scale == mw.MAX_TEXT_SCALE
    for _ in range(60):
        window.decrease_text_size()
    assert window.text_scale == mw.MIN_TEXT_SCALE

    window.reset_text_size()
    assert window.text_scale == 1.0
    assert window.styleSheet() == mw._scaled_style_sheet(mw._STYLE_SHEET, 1.0)


def test_text_size_scales_every_label_including_headings(window: MainWindow) -> None:
    headings = [
        label
        for label in window.findChildren(QLabel)
        if label.text() in {"TRANSCRIPT", "CONVERSATION", "TOOL ACTIVITY"}
    ]
    assert headings
    window.show()
    before = {label.text(): label.font().pixelSize() for label in headings}
    window.set_text_scale(2.0)
    after = {label.text(): label.font().pixelSize() for label in headings}
    assert all(after[name] >= before[name] * 1.9 for name in before)  # 13px -> 26px


def test_keyboard_shortcuts_are_registered(window: MainWindow) -> None:
    keys = {s.key().toString() for s in window.findChildren(QShortcut)}
    assert {"Ctrl+L", "Ctrl+=", "Ctrl++", "Ctrl+-", "Ctrl+0"} <= keys


def test_focus_input_shortcut_target(window: MainWindow) -> None:
    window.show()
    window.focus_input()
    # (hasFocus() needs an *active* window, which a headless run never has;
    # focusWidget() is the widget that receives focus the moment it activates.)
    assert window.focusWidget() is window._input


def test_keyboard_focus_ring_is_styled() -> None:
    assert "QPushButton:focus" in mw._STYLE_SHEET and "QLineEdit:focus" in mw._STYLE_SHEET


def test_voice_unavailable_state_looks_disabled_and_says_why(window: MainWindow) -> None:
    window.set_voice_available(False, "no microphone was found")

    button = window._push_to_talk_button
    assert not button.isEnabled()
    assert "UNAVAILABLE" in button.text()
    assert button.graphicsEffect() is None  # no "active" glow on a dead control
    assert "no microphone was found" in button.accessibleDescription()
    assert "unavailable" in window._mic_label.text()

    window.set_voice_available(True)
    assert button.isEnabled() and button.text() == "HOLD TO TALK"
    assert button.graphicsEffect() is not None
