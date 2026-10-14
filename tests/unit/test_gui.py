from __future__ import annotations

from unittest.mock import patch

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

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
    assert "#7c4dff" in window._status_label.styleSheet()


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
    assert "#5f7396" in window._status_label.styleSheet()


def test_tool_activity_lines_are_color_coded(window: MainWindow) -> None:
    window.append_tool_activity("Started: open_application")
    window.append_tool_activity("Completed: open_application")
    window.append_tool_activity("Failed (open_application): boom")

    colors = [
        window._tool_activity_list.item(i).foreground().color().name() for i in range(3)
    ]
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
    assert "#7c4dff" in html_text
