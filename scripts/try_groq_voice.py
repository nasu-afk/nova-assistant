"""Quick manual check: speak a line out loud using the configured voice provider.

Run with:
    uv run python scripts/try_groq_voice.py
"""

from __future__ import annotations

import asyncio

from app.config.settings import get_settings
from app.voice.factory import build_tts_provider


async def main() -> None:
    settings = get_settings()
    tts = build_tts_provider(settings)
    print(f"Using voice provider: {settings.voice_provider}")
    await tts.speak("Hello, I am NOVA.")
    print("Done — did you hear it?")


if __name__ == "__main__":
    asyncio.run(main())
