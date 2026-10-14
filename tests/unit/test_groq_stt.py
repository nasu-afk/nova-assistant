from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.errors import VoiceError
from app.voice.groq_stt import GroqSTTProvider


@pytest.fixture
def provider() -> GroqSTTProvider:
    return GroqSTTProvider(api_key="gsk_fake_key_for_tests")


@pytest.mark.asyncio
async def test_transcribe_returns_text(provider: GroqSTTProvider) -> None:
    response = SimpleNamespace(text="open notepad", language="en")
    provider._client.audio.transcriptions.create = AsyncMock(return_value=response)

    result = await provider.transcribe(b"\x00\x01" * 100)

    assert result.text == "open notepad"
    assert result.language == "en"


@pytest.mark.asyncio
async def test_transcribe_strips_whitespace(provider: GroqSTTProvider) -> None:
    response = SimpleNamespace(text="  hello there  ", language="en")
    provider._client.audio.transcriptions.create = AsyncMock(return_value=response)

    result = await provider.transcribe(b"\x00\x01" * 100)

    assert result.text == "hello there"


@pytest.mark.asyncio
async def test_transcribe_empty_audio_skips_api_call(provider: GroqSTTProvider) -> None:
    mock_create = AsyncMock()
    provider._client.audio.transcriptions.create = mock_create

    result = await provider.transcribe(b"")

    assert result.text == ""
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_transcribe_wraps_sdk_errors_as_voice_error(provider: GroqSTTProvider) -> None:
    provider._client.audio.transcriptions.create = AsyncMock(
        side_effect=RuntimeError("network down")
    )

    with pytest.raises(VoiceError):
        await provider.transcribe(b"\x00\x01" * 100)


@pytest.mark.asyncio
async def test_transcribe_sends_correct_model_and_wraps_wav(provider: GroqSTTProvider) -> None:
    mock_create = AsyncMock(return_value=SimpleNamespace(text="hi", language="en"))
    provider._client.audio.transcriptions.create = mock_create

    await provider.transcribe(b"\x00\x01" * 100, sample_rate=16_000)

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "whisper-large-v3-turbo"
    filename, wav_bytes = kwargs["file"]
    assert filename == "audio.wav"
    assert wav_bytes[:4] == b"RIFF"


def test_missing_dependency_raises_voice_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "groq":
            raise ImportError("no module named groq")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError):
        GroqSTTProvider(api_key="gsk_fake")
