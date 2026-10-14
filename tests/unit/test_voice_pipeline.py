from __future__ import annotations

import pytest

from app.core.agent import ConversationAgent
from app.voice.audio_capture import InMemoryRecorder
from app.voice.base import TranscriptionResult
from app.voice.null_providers import NullTTSProvider
from app.voice.pipeline import VoicePipeline


class FakeSTT:
    def __init__(self, text: str) -> None:
        self._text = text
        self.received_audio: bytes | None = None

    async def transcribe(
        self, audio_bytes: bytes, *, sample_rate: int = 16_000
    ) -> TranscriptionResult:
        self.received_audio = audio_bytes
        return TranscriptionResult(text=self._text)


class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> bytes:
        return text.encode()

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class FakeLLM:
    async def complete(self, messages, *, tools=None, max_tokens=1024, temperature=0.7):  # type: ignore[no-untyped-def]
        from app.ai.base import LLMResponse

        return LLMResponse(content="Sure, doing that now.")


@pytest.mark.asyncio
async def test_full_turn_records_transcribes_and_speaks() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(b"\x00\x01" * 100)
    stt = FakeSTT("open notepad")
    tts = FakeTTS()
    agent = ConversationAgent(FakeLLM())
    pipeline = VoicePipeline(recorder, stt, tts, agent)

    pipeline.start_listening()
    reply = await pipeline.stop_listening_and_respond()

    assert reply == "Sure, doing that now."
    assert stt.received_audio == b"\x00\x01" * 100
    assert tts.spoken == ["Sure, doing that now."]
    assert agent.history[0].content == "open notepad"


@pytest.mark.asyncio
async def test_empty_transcription_skips_agent_and_tts() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(b"")
    stt = FakeSTT("")
    tts = FakeTTS()
    agent = ConversationAgent(FakeLLM())
    pipeline = VoicePipeline(recorder, stt, tts, agent)

    pipeline.start_listening()
    reply = await pipeline.stop_listening_and_respond()

    assert reply == ""
    assert tts.spoken == []
    assert agent.history == []


@pytest.mark.asyncio
async def test_null_tts_provider_never_raises() -> None:
    tts = NullTTSProvider()
    await tts.speak("anything")
    assert await tts.synthesize("anything") == b""


@pytest.mark.asyncio
async def test_on_audio_captured_hook_receives_raw_bytes_before_transcription() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(b"\x01\x02" * 50)
    captured: list[bytes] = []
    pipeline = VoicePipeline(
        recorder,
        FakeSTT("hello"),
        FakeTTS(),
        ConversationAgent(FakeLLM()),
        on_audio_captured=captured.append,
    )

    pipeline.start_listening()
    await pipeline.stop_listening_and_respond()

    assert captured == [b"\x01\x02" * 50]


@pytest.mark.asyncio
async def test_pipeline_works_without_audio_hook() -> None:
    recorder = InMemoryRecorder()
    recorder.set_next_recording(b"\x00\x01" * 10)
    pipeline = VoicePipeline(recorder, FakeSTT("hi"), FakeTTS(), ConversationAgent(FakeLLM()))

    pipeline.start_listening()
    reply = await pipeline.stop_listening_and_respond()

    assert reply == "Sure, doing that now."
