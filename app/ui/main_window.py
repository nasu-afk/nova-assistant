"""NOVA's PySide6 desktop GUI.

The GUI is a *consumer* of the core system, never the other way around:
`MainWindow` only ever calls plain update methods (`set_status`,
`append_transcript`, ...) and subscribes them to the `EventBus`. Nothing in
`app/core`, `app/tools`, `app/security`, or `app/memory` imports this
module, so the whole application keeps working headlessly if the GUI is
never started (principle #14).

Visual style: a dark sci-fi "HUD" look (deep space background, cyan/violet
neon accents, monospace readouts) applied via a QSS stylesheet plus a
couple of `QGraphicsDropShadowEffect` glows — both are cosmetic layers on
top of the plain widgets above; nothing about the update methods' behavior
or the event wiring changes because of them.
"""

from __future__ import annotations

import html
import re
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAccessible, QAccessibleEvent, QColor, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.enums import EventType
from app.events.bus import Event, EventBus
from app.security.manager import summarize_payload

# Palette: deep-space background, cyan as the primary "system online" accent,
# violet as a secondary accent, and one status color per agent state so the
# whole window visibly shifts tone as NOVA moves through a turn.
_BG = "#05070d"
_PANEL = "#0b1220"
_PANEL_ALT = "#0e1626"
_BORDER = "#1c2b45"
_CYAN = "#00e5ff"
_VIOLET = "#9d7bff"  # lightened from #7c4dff: was 3.8-4.2:1, now >= 5.8:1
_TEXT = "#dce9ff"
_TEXT_DIM = "#8ba0c2"  # lightened from #5f7396: was 3.8-4.2:1, now >= 6.8:1
_SUCCESS = "#39ff9c"

_STATUS_COLORS = {
    "idle": _TEXT_DIM,
    "listening": "#00e5ff",
    "thinking": "#ffb300",
    "speaking": _VIOLET,
    "working": "#00e5ff",
    "error": "#ff4d6d",
}

_MONO_FONT_FAMILIES = "Cascadia Code, Consolas, 'Courier New', monospace"

_FONT_SIZE_RE = re.compile(r"font-size:\s*(\d+(?:\.\d+)?)px")
MIN_TEXT_SCALE, MAX_TEXT_SCALE = 0.8, 2.0


def _scaled_style_sheet(style_sheet: str, scale: float) -> str:
    """The stylesheet with every px font size multiplied by `scale`, so the whole
    window (not just one widget) grows for people who need larger text."""
    return _FONT_SIZE_RE.sub(
        lambda m: f"font-size: {round(float(m.group(1)) * scale)}px", style_sheet
    )


_STYLE_SHEET = f"""
QMainWindow, QWidget#centralPanel {{
    background-color: {_BG};
}}

QWidget {{
    color: {_TEXT};
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}}

QFrame#headerPanel {{
    background-color: {_PANEL};
    border: 1px solid {_BORDER};
    border-radius: 10px;
}}

QLabel#brandLabel {{
    color: {_CYAN};
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 4px;
}}

QLabel#statusLabel {{
    font-family: {_MONO_FONT_FAMILIES};
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
}}

QLabel#micLabel, QLabel#taskLabel {{
    font-family: {_MONO_FONT_FAMILIES};
    color: {_TEXT_DIM};
    font-size: 12px;
}}

QPushButton#pttButton {{
    background-color: {_PANEL_ALT};
    border: 1px solid {_CYAN};
    border-radius: 8px;
    color: {_CYAN};
    font-family: {_MONO_FONT_FAMILIES};
    font-weight: 700;
    letter-spacing: 2px;
    padding: 10px;
}}

QPushButton#pttButton:hover {{
    background-color: #0f1e33;
}}

QPushButton#pttButton:pressed {{
    background-color: {_CYAN};
    color: {_BG};
}}

QPushButton#pttButton:disabled {{
    background-color: {_PANEL};
    border: 1px dashed {_BORDER};
    color: {_TEXT_DIM};
}}

QTabWidget::pane {{
    border: 1px solid {_BORDER};
    border-radius: 8px;
    background-color: {_PANEL};
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {_TEXT_DIM};
    font-family: {_MONO_FONT_FAMILIES};
    padding: 8px 18px;
    border: none;
    letter-spacing: 1px;
}}

QTabBar::tab:selected {{
    color: {_CYAN};
    border-bottom: 2px solid {_CYAN};
}}

QTabBar::tab:hover {{
    color: {_TEXT};
}}

QLabel.sectionLabel {{
    color: {_VIOLET};
    font-family: {_MONO_FONT_FAMILIES};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding-top: 4px;
}}

QTextEdit, QListWidget {{
    background-color: {_PANEL_ALT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    color: {_TEXT};
    font-family: {_MONO_FONT_FAMILIES};
    font-size: 12px;
    padding: 6px;
    selection-background-color: {_CYAN};
    selection-color: {_BG};
}}

QListWidget::item {{
    padding: 3px 2px;
    border-bottom: 1px solid #10192b;
}}

QProgressBar#taskProgress {{
    background-color: {_PANEL_ALT};
    border: 1px solid {_BORDER};
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}}

QProgressBar#taskProgress::chunk {{
    background-color: {_CYAN};
    border-radius: 4px;
}}

QScrollBar:vertical {{
    background: {_PANEL};
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: {_BORDER};
    border-radius: 5px;
    min-height: 24px;
}}

QScrollBar::handle:vertical:hover {{
    background: {_CYAN};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QLineEdit {{
    background-color: {_PANEL_ALT};
    border: 1px solid {_BORDER};
    border-radius: 6px;
    padding: 8px 10px;
    color: {_TEXT};
    font-size: 13px;
    selection-background-color: {_VIOLET};
    selection-color: {_BG};
}}

QPushButton#sendButton {{
    background-color: {_PANEL_ALT};
    border: 1px solid {_CYAN};
    border-radius: 6px;
    color: {_CYAN};
    font-weight: 700;
    padding: 8px 16px;
}}

QPushButton#sendButton:hover {{
    background-color: #0f1e33;
}}

QCheckBox {{
    color: {_TEXT};
    spacing: 8px;
    font-size: 12px;
}}

/* Keyboard focus must be obvious (never rely on the OS default on a dark theme). */
QLineEdit:focus, QPushButton:focus, QCheckBox:focus, QTabBar::tab:focus,
QTextEdit:focus, QListWidget:focus {{
    border: 2px solid {_CYAN};
}}
"""


class MainWindow(QMainWindow):
    """The main NOVA application window."""

    # Emitted when the user closes the window, so the app can shut down
    # gracefully (stop listeners, close the DB/browser) instead of the Qt loop
    # just dying underneath running coroutines.
    closed = Signal()
    # On-screen controls, so the window is usable without the global hotkey:
    # hold-to-talk (pressed/released) and typed requests (text, read-aloud?).
    talk_pressed = Signal()
    talk_released = Signal()
    text_submitted = Signal(str, bool)

    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        self.closed.emit()
        super().closeEvent(event)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NOVA")
        self.resize(960, 640)
        self.setStyleSheet(_STYLE_SHEET)

        self._status_label = QLabel("IDLE")
        self._status_label.setObjectName("statusLabel")
        self._status_dot = QLabel("●")
        self._status_dot.setObjectName("statusDot")
        self._mic_label = QLabel("Microphone: off")
        self._mic_label.setObjectName("micLabel")
        self._transcript_view = QTextEdit(readOnly=True)
        self._conversation_view = QTextEdit(readOnly=True)
        self._task_label = QLabel("No active task")
        self._task_label.setObjectName("taskLabel")
        self._progress_bar = QProgressBar()
        self._progress_bar.setObjectName("taskProgress")
        self._progress_bar.setRange(0, 0)  # indeterminate until a real task starts
        self._progress_bar.setVisible(False)
        self._tool_activity_list = QListWidget()
        self._log_list = QListWidget()
        self._memory_view = QTextEdit(readOnly=True)
        self._push_to_talk_button = QPushButton("HOLD TO TALK")
        self._push_to_talk_button.setObjectName("pttButton")
        self._push_to_talk_button.setMinimumHeight(44)
        # Reachable with Tab (hold Space while focused to talk), but a mouse
        # click doesn't steal focus - otherwise the global Ctrl+Space hotkey would
        # also "press" the focused button and start two recordings.
        self._push_to_talk_button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        self._push_to_talk_button.pressed.connect(self._on_talk_pressed)
        self._push_to_talk_button.released.connect(self._on_talk_released)

        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a request and press Enter…")
        self._input.returnPressed.connect(self._submit_text)
        self._send_button = QPushButton("Send")
        self._send_button.setObjectName("sendButton")
        self._send_button.clicked.connect(self._submit_text)
        self._speak_checkbox = QCheckBox("Read replies aloud")

        self._text_scale = 1.0
        self._status_state = "idle"

        self._apply_glow(self._push_to_talk_button, QColor(_CYAN), radius=18)
        self._set_status_color("idle")

        self._build_layout()
        self._apply_accessibility_metadata()
        self._install_shortcuts()

    @staticmethod
    def _apply_glow(widget: QWidget, color: QColor, radius: int = 16) -> None:
        """A soft colored glow — Qt's QSS has no `box-shadow`, so a real
        neon-style glow needs a `QGraphicsEffect` instead of a stylesheet
        rule."""
        effect = QGraphicsDropShadowEffect(widget)
        effect.setColor(color)
        effect.setBlurRadius(radius)
        effect.setOffset(0, 0)
        widget.setGraphicsEffect(effect)

    def _set_status_color(self, state: str) -> None:
        self._status_state = state
        color = _STATUS_COLORS.get(state, _STATUS_COLORS["idle"])
        self._status_label.setStyleSheet(f"color: {color};")
        dot_size = round(16 * getattr(self, "_text_scale", 1.0))
        self._status_dot.setStyleSheet(f"color: {color}; font-size: {dot_size}px;")
        self._apply_glow(self._status_dot, QColor(color), radius=14)

    def _build_layout(self) -> None:
        central = QWidget()
        central.setObjectName("centralPanel")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        layout.addWidget(self._build_input_row())

        tabs = self._tabs = QTabWidget()
        tabs.addTab(self._build_conversation_tab(), "Conversation")
        tabs.addTab(self._build_activity_tab(), "Activity")
        tabs.addTab(self._memory_view, "Memory")
        tabs.addTab(self._log_list, "Logs")
        layout.addWidget(tabs)

        self.setCentralWidget(central)

        # Tab order follows how people use it: type, send, read-aloud, talk, then
        # the content tabs (the default order put the talk button first).
        self.setTabOrder(self._input, self._send_button)
        self.setTabOrder(self._send_button, self._speak_checkbox)
        self.setTabOrder(self._speak_checkbox, self._push_to_talk_button)
        self.setTabOrder(self._push_to_talk_button, tabs)

    def _build_input_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._input, 1)
        layout.addWidget(self._send_button)
        layout.addWidget(self._speak_checkbox)
        return row

    def _apply_accessibility_metadata(self) -> None:
        """Names/descriptions for screen readers and tooltips for everyone. Qt
        exposes these through the platform accessibility API (UI Automation on
        Windows, so Narrator/NVDA can read them)."""

        def describe(widget: QWidget, name: str, description: str = "") -> None:
            widget.setAccessibleName(name)
            if description:
                widget.setAccessibleDescription(description)
                widget.setToolTip(description)

        describe(self, "NOVA voice assistant")
        describe(
            self._push_to_talk_button,
            "Hold to talk",
            "Hold down to record, release to send. Hold Space while this button is "
            "focused, or use the global push-to-talk hotkey.",
        )
        describe(
            self._input,
            "Type a request",
            "Type a request for NOVA and press Enter. Shortcut: Ctrl+L.",
        )
        describe(self._send_button, "Send request", "Send the typed request to NOVA.")
        describe(
            self._speak_checkbox,
            "Read replies aloud",
            "When ticked, NOVA also speaks the reply to a typed request.",
        )
        describe(self._status_label, "Status: IDLE")
        describe(self._mic_label, "Microphone status")
        describe(self._transcript_view, "Transcript of what you said")
        describe(self._conversation_view, "Conversation history")
        describe(self._task_label, "Current task")
        describe(self._progress_bar, "Task progress")
        describe(self._tool_activity_list, "Tool activity")
        describe(self._log_list, "Logs")
        describe(self._memory_view, "Memory")
        describe(self._tabs, "Sections")
        # The glowing dot only duplicates the status text; hide it from readers.
        self._status_dot.setAccessibleName("")

    def _install_shortcuts(self) -> None:
        def bind(sequence: str, callback: Any) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)

        bind("Ctrl+L", self.focus_input)
        bind("Ctrl+=", self.increase_text_size)
        bind("Ctrl++", self.increase_text_size)
        bind("Ctrl+-", self.decrease_text_size)
        bind("Ctrl+0", self.reset_text_size)

    # --- On-screen controls ---

    def _on_talk_pressed(self) -> None:
        self.talk_pressed.emit()

    def _on_talk_released(self) -> None:
        self.talk_released.emit()

    def _submit_text(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self.text_submitted.emit(text, self._speak_checkbox.isChecked())

    def set_voice_available(self, available: bool, reason: str = "") -> None:
        """Switch the voice controls off (e.g. no microphone) while leaving typing
        fully working, and say why in a way screen readers pick up."""
        self._push_to_talk_button.setEnabled(available)
        if available:
            self._mic_label.setText("Microphone: off")
            self._push_to_talk_button.setText("HOLD TO TALK")
            self._push_to_talk_button.setAccessibleName("Hold to talk")
            self._apply_glow(self._push_to_talk_button, QColor(_CYAN), radius=18)
            return
        # Look disabled too (no glow, dashed outline, says why), not just *be* disabled.
        self._push_to_talk_button.setText("VOICE UNAVAILABLE — TYPE BELOW")
        self._push_to_talk_button.setGraphicsEffect(None)  # type: ignore[arg-type]  # Qt accepts None
        detail = (
            f"Voice input isn't available: {reason}" if reason else "Voice input isn't available."
        )
        self._mic_label.setText("Microphone: unavailable")
        self._push_to_talk_button.setAccessibleName("Hold to talk (unavailable)")
        self._push_to_talk_button.setAccessibleDescription(f"{detail} You can still type requests.")
        self._push_to_talk_button.setToolTip(f"{detail} You can still type requests.")

    def focus_input(self) -> None:
        self._input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._input.selectAll()

    # --- Text size (Ctrl+= / Ctrl+- / Ctrl+0) ---

    @property
    def text_scale(self) -> float:
        return self._text_scale

    def set_text_scale(self, scale: float) -> None:
        scale = round(min(MAX_TEXT_SCALE, max(MIN_TEXT_SCALE, scale)), 2)
        self._text_scale = scale
        self.setStyleSheet(_scaled_style_sheet(_STYLE_SHEET, scale))
        self._set_status_color(self._status_state)

    def increase_text_size(self) -> None:
        self.set_text_scale(self._text_scale + 0.1)

    def decrease_text_size(self) -> None:
        self.set_text_scale(self._text_scale - 0.1)

    def reset_text_size(self) -> None:
        self.set_text_scale(1.0)

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerPanel")
        outer = QVBoxLayout(header)
        outer.setContentsMargins(16, 12, 16, 14)
        outer.setSpacing(8)

        brand_row = QHBoxLayout()
        brand = QLabel("N O V A")
        brand.setObjectName("brandLabel")
        self._apply_glow(brand, QColor(_CYAN), radius=22)
        brand_row.addWidget(brand)
        brand_row.addStretch()
        brand_row.addWidget(self._status_dot)
        brand_row.addWidget(self._status_label)
        outer.addLayout(brand_row)

        outer.addWidget(self._mic_label)
        outer.addWidget(self._push_to_talk_button)
        return header

    def _build_conversation_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self._styled_section_label("TRANSCRIPT"))
        layout.addWidget(self._transcript_view)
        layout.addWidget(self._styled_section_label("CONVERSATION"))
        layout.addWidget(self._conversation_view)
        return widget

    def _build_activity_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        layout.addWidget(self._task_label)
        layout.addWidget(self._progress_bar)
        layout.addWidget(self._styled_section_label("TOOL ACTIVITY"))
        layout.addWidget(self._tool_activity_list)
        return widget

    @staticmethod
    def _styled_section_label(text: str) -> QLabel:
        label = QLabel(text)
        font = QFont("Cascadia Code", 9)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.5)
        label.setFont(font)
        label.setStyleSheet(f"color: {_VIOLET};")
        return label

    # --- Plain update methods: the only surface the rest of the app touches ---

    @staticmethod
    def _notify_accessibility(widget: QWidget, event: QAccessible.Event) -> None:
        """Tell screen readers something changed. Best effort: a no-op when no
        assistive technology is running (and in headless tests)."""
        try:
            if QAccessible.isActive():
                QAccessible.updateAccessibility(QAccessibleEvent(widget, event))
        except Exception:  # noqa: BLE001 - accessibility must never break the UI
            pass

    def set_status(self, text: str) -> None:
        self._status_label.setText(text.upper())
        self._status_label.setAccessibleName(f"Status: {text.upper()}")
        self._notify_accessibility(self._status_label, QAccessible.Event.NameChanged)
        state = text.strip().lower().split()[0] if text.strip() else "idle"
        if "error" in text.lower() or "unavailable" in text.lower() or "failed" in text.lower():
            state = "error"
        self._set_status_color(state)

    def set_mic_status(self, listening: bool) -> None:
        self._mic_label.setText(f"Microphone: {'listening' if listening else 'off'}")

    def append_transcript(self, text: str) -> None:
        self._transcript_view.append(text)

    def append_conversation(self, role: str, text: str) -> None:
        color = _CYAN if role.lower() in {"you", "user"} else _VIOLET
        label = f'<span style="color:{color};font-weight:600;">{html.escape(role)}:</span>'
        self._conversation_view.append(f"{label} {html.escape(text)}")
        if role.lower() not in {"you", "user"}:
            # Have a screen reader announce NOVA's reply as it arrives.
            self._status_label.setAccessibleName(f"{role} says: {text[:400]}")
            self._notify_accessibility(self._status_label, QAccessible.Event.Alert)

    def set_task(self, goal: str | None) -> None:
        self._task_label.setText(f"Task: {goal}" if goal else "No active task")
        self._progress_bar.setVisible(goal is not None)

    def append_tool_activity(self, line: str) -> None:
        item = QListWidgetItem(line)
        if line.startswith("Completed"):
            item.setForeground(QColor(_SUCCESS))
        elif line.startswith("Failed"):
            item.setForeground(QColor(_STATUS_COLORS["error"]))
        else:
            item.setForeground(QColor(_CYAN))
        self._tool_activity_list.addItem(item)

    def append_log(self, line: str) -> None:
        self._log_list.addItem(line)

    def set_memory_snapshot(self, text: str) -> None:
        self._memory_view.setPlainText(text)

    def show_confirmation_dialog(self, tool_name: str, payload: dict[str, Any]) -> bool:
        """Blocking Yes/No dialog for HIGH/CRITICAL tool confirmations.

        Must be called on the Qt main thread (e.g. via a Qt-integrated event
        loop such as `qasync`), since `QMessageBox` is not thread-safe.
        """
        reply = QMessageBox.question(
            self,
            "Confirm action",
            f"NOVA wants to run '{tool_name}' with:\n{summarize_payload(payload)}\n\nAllow this?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    # --- Event bus wiring ---

    def wire_events(self, bus: EventBus) -> None:
        """Subscribe this window's update methods to the shared EventBus."""

        async def on_tool_started(event: Event) -> None:
            self.append_tool_activity(f"Started: {event.payload.get('tool')}")

        async def on_tool_completed(event: Event) -> None:
            tool = event.payload.get("tool") or f"step {event.payload.get('step_id')}"
            self.append_tool_activity(f"Completed: {tool}")

        async def on_tool_failed(event: Event) -> None:
            tool = event.payload.get("tool") or f"step {event.payload.get('step_id')}"
            self.append_tool_activity(f"Failed ({tool}): {event.payload.get('error')}")

        async def on_task_created(event: Event) -> None:
            self.set_task(str(event.payload.get("goal")))
            self.set_status("Working")

        async def on_task_completed(event: Event) -> None:
            self.set_task(None)
            self.set_status("Idle")

        async def on_task_failed(event: Event) -> None:
            self.set_task(None)
            self.set_status("Idle (last task failed)")

        async def on_voice_started(event: Event) -> None:
            self.set_mic_status(True)
            self.set_status("Listening")

        async def on_voice_stopped(event: Event) -> None:
            self.set_mic_status(False)
            self.set_status("Thinking")

        async def on_transcript_created(event: Event) -> None:
            text = str(event.payload.get("text", ""))
            self.append_transcript(text)
            self.append_conversation("You", text)

        async def on_response_ready(event: Event) -> None:
            text = str(event.payload.get("text", ""))
            self.append_conversation("NOVA", text)
            self.set_status("Speaking")

        bus.subscribe(EventType.TOOL_STARTED, on_tool_started)
        bus.subscribe(EventType.TOOL_COMPLETED, on_tool_completed)
        bus.subscribe(EventType.TOOL_FAILED, on_tool_failed)
        bus.subscribe(EventType.TASK_CREATED, on_task_created)
        bus.subscribe(EventType.TASK_COMPLETED, on_task_completed)
        bus.subscribe(EventType.TASK_FAILED, on_task_failed)
        bus.subscribe(EventType.VOICE_STARTED, on_voice_started)
        bus.subscribe(EventType.VOICE_STOPPED, on_voice_stopped)
        bus.subscribe(EventType.TRANSCRIPT_CREATED, on_transcript_created)
        bus.subscribe(EventType.RESPONSE_READY, on_response_ready)
