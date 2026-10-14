"""GUI entrypoint.

Kept separate from `app/main.py` (the headless entrypoint) — starting the
GUI is an explicit choice (`python -m app.ui.app`), never a requirement for
the rest of NOVA to function.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.core.logging import configure_logging
from app.events.bus import EventBus
from app.ui.main_window import MainWindow


def create_application(argv: list[str] | None = None) -> tuple[QApplication, MainWindow]:
    """Build (but don't run) the QApplication + MainWindow, wired to a fresh
    EventBus. Split out from `main()` so tests can construct the same
    objects without calling `app.exec()`."""
    configure_logging()
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication(argv or sys.argv)
    window = MainWindow()
    window.wire_events(EventBus())
    return application, window


def main() -> int:  # pragma: no cover - exercised manually, needs a real display
    application, window = create_application()
    window.show()
    return application.exec()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
