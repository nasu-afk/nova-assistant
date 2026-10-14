"""Test-wide fixtures.

Forces Qt's offscreen platform plugin so PySide6 widgets can be constructed
and exercised in this headless sandbox (no X server / Wayland compositor).
Must be set before any PySide6 import happens.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
