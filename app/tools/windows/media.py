"""System volume and screenshot tools.

Both depend on OS-native or GUI libraries that aren't available in every
environment (headless CI, a container with no audio/display). Rather than
raising and crashing the agent's turn, these tools catch that case and
return a structured `ToolResult(success=False, ...)` explaining why —
per the "never hide failures, but never crash on an unsupported platform"
principle.
"""

from __future__ import annotations

import platform
from typing import Any, ClassVar

from app.core.enums import RiskLevel
from app.core.errors import ValidationError
from app.tools.base import Tool, ToolMetadata, ToolResult


class SetVolumeTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="set_volume",
        description="Set the system output volume to a percentage (0-100).",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        level = payload.get("level")
        if isinstance(level, bool) or not isinstance(level, int | float) or not (0 <= level <= 100):
            raise ValidationError("set_volume requires a numeric 'level' between 0 and 100")
        return {"level": int(level)}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        level = payload["level"]
        if platform.system() != "Windows":
            return ToolResult(
                success=False,
                error="Volume control is only implemented for Windows in this build.",
            )

        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            speakers = AudioUtilities.GetSpeakers()
            # Newer pycaw releases return a wrapper that exposes `.EndpointVolume`;
            # older ones return a raw COM device that needs `.Activate(...)`.
            volume = getattr(speakers, "EndpointVolume", None)
            if volume is None:
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
        except Exception as exc:  # noqa: BLE001 - normalize any driver/COM failure
            return ToolResult(success=False, error=f"Couldn't set volume: {exc}")

        return ToolResult(success=True, message=f"Volume set to {level}%.", data={"level": level})


class TakeScreenshotTool(Tool):
    metadata: ClassVar[ToolMetadata] = ToolMetadata(
        name="take_screenshot",
        description="Capture the screen and save it as a PNG file.",
        risk_level=RiskLevel.LOW,
    )

    def validate_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        path = payload.get("path", "screenshot.png")
        if not isinstance(path, str) or not path:
            raise ValidationError("take_screenshot 'path' must be a string")
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            # Never let a "screenshot" overwrite scripts/configs with image bytes.
            raise ValidationError("take_screenshot only writes .png/.jpg files")
        return {"path": path}

    async def execute(self, payload: dict[str, Any]) -> ToolResult:
        path = payload["path"]
        try:
            from PIL import ImageGrab
        except ImportError:
            return ToolResult(
                success=False,
                error="Screenshot support requires the 'Pillow' package to be installed.",
            )

        try:
            image = ImageGrab.grab()
            image.save(path)
        except Exception as exc:  # noqa: BLE001 - e.g. no display available
            return ToolResult(success=False, error=f"Couldn't capture screen: {exc}")

        return ToolResult(success=True, message=f"Saved screenshot to {path}.", data={"path": path})
