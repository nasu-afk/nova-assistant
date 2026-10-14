"""Push-to-talk voice pipeline: Microphone -> STT -> Agent -> TTS.

Wake-word / always-listening mode is added in Phase 10 on top of the same
`STTProvider`/`TTSProvider` interfaces — this class only implements the
"hold a key, speak, release" flow from section 9 of the project spec.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.agent import RespondingAgent
from app.core.enums import EventType
from app.core.logging import get_logger
from app.events.bus import Event, EventBus
from app.voice.audio_capture import AudioRecorder
from app.voice.base import STTProvider, TTSProvider

logger = get_logger(__name__)


class VoicePipeline:
    """Coordinates one push-to-talk turn: record -> transcribe -> reply -> speak.

    `events`, if given, publishes VOICE_STARTED/VOICE_STOPPED around
    recording and TRANSCRIPT_CREATED once transcription completes — the
    same `EventBus` a GUI's `MainWindow.wire_events()` already knows how to
    consume (see `app/ui/main_window.py`), so a GUI just needs to share the
    bus with this pipeline to get live status for free.
    """

    def __init__(
        self,
        recorder: AudioRecorder,
        stt: STTProvider,
        tts: TTSProvider,
        agent: RespondingAgent,
        on_audio_captured: Callable[[bytes], None] | None = None,
        events: EventBus | None = None,
    ) -> None:
        self._recorder = recorder
        self._stt = stt
        self._tts = tts
        self._agent = agent
        # Optional diagnostic hook, called with the raw recorded PCM16 bytes
        # right after recording stops (before transcription) — e.g. to log
        # its RMS energy or dump it to a .wav for manual inspection when
        # transcription looks wrong. See app/voice/debug.py.
        self._on_audio_captured = on_audio_captured
        self._events = events

    async def _publish(self, event_type: EventType, **payload: object) -> None:
        if self._events is not None:
            await self._events.publish(Event(type=event_type, payload=payload))

    async def publish_started(self) -> None:
        """Publish VOICE_STARTED. Call this from an async context (the
        caller's `run_coroutine_threadsafe`-bridged wrapper around
        `start_listening()`, not from a hotkey thread directly)."""
        await self._publish(EventType.VOICE_STARTED)

    def start_listening(self) -> None:
        """Call when the push-to-talk key is pressed.

        Deliberately synchronous and does not itself publish VOICE_STARTED —
        this is typically invoked directly from a hotkey listener's own OS
        thread (see `app.voice.hotkey.PushToTalkListener`), not the asyncio
        event loop's thread, so publishing an event here would need a
        thread-safe bridge this class has no business knowing about. The
        caller (see `app/main.py`), which already bridges `stop_listening_
        and_respond` via `asyncio.run_coroutine_threadsafe`, is responsible
        for publishing VOICE_STARTED itself if it wants that event.
        """
        logger.info("voice_recording_started")
        self._recorder.start()

    async def stop_listening_and_respond(self) -> str:
        """Call when the push-to-talk key is released.

        Stops recording, transcribes, gets the agent's reply, speaks it, and
        returns the reply text (useful for the GUI transcript view).
        """
        audio_bytes = self._recorder.stop()
        logger.info("voice_recording_stopped", bytes=len(audio_bytes))
        await self._publish(EventType.VOICE_STOPPED)

        if self._on_audio_captured is not None:
            self._on_audio_captured(audio_bytes)

        transcription = await self._stt.transcribe(audio_bytes)
        if not transcription.text.strip():
            logger.info("voice_transcription_empty")
            return ""

        logger.info("voice_transcript_created", text=transcription.text)
        await self._publish(EventType.TRANSCRIPT_CREATED, text=transcription.text)

        reply = await self._agent.respond(transcription.text)
        await self._tts.speak(reply)
        return reply
