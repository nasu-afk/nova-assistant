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
from typing import Any

from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
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

# Palette: deep-space background, cyan as the primary "system online" accent,
# violet as a secondary accent, and one status color per agent state so the
# whole window visibly shifts tone as NOVA moves through a turn.
_BG = "#05070d"
_PANEL = "#0b1220"
_PANEL_ALT = "#0e1626"
_BORDER = "#1c2b45"
_CYAN = "#00e5ff"
_VIOLET = "#7c4dff"
_TEXT = "#dce9ff"
_TEXT_DIM = "#5f7396"
_SUCCESS = "#39ff9c"

_STATUS_COLORS = {
    "idle": "#5f7396",
    "listening": "#00e5ff",
    "thinking": "#ffb300",
    "speaking": "#7c4dff",
    "working": "#00e5ff",
    "error": "#ff4d6d",
}

_MONO_FONT_FAMILIES = "Cascadia Code, Consolas, 'Courier New', monospace"

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
"""


class MainWindow(QMainWindow):
    """The main NOVA application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NOVA")
        self.resize(960, 640)
        self.setStyleSheet(_STYLE_SHEET)

        self._status_label = QLabel("Idle")
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

        self._apply_glow(self._push_to_talk_button, QColor(_CYAN), radius=18)
        self._set_status_color("idle")

        self._build_layout()

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
        color = _STATUS_COLORS.get(state, _STATUS_COLORS["idle"])
        self._status_label.setStyleSheet(f"color: {color};")
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 16px;")
        self._apply_glow(self._status_dot, QColor(color), radius=14)

    def _build_layout(self) -> None:
        central = QWidget()
        central.setObjectName("centralPanel")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        tabs = QTabWidget()
        tabs.addTab(self._build_conversation_tab(), "Conversation")
        tabs.addTab(self._build_activity_tab(), "Activity")
        tabs.addTab(self._memory_view, "Memory")
        tabs.addTab(self._log_list, "Logs")
        layout.addWidget(tabs)

        self.setCentralWidget(central)

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

    def set_status(self, text: str) -> None:
        self._status_label.setText(text.upper())
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
            f"NOVA wants to run '{tool_name}' with:\n{payload}\n\nAllow this?",
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
