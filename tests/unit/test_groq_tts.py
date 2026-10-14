from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.core.errors import VoiceError
from app.voice.groq_tts import GroqTTSProvider


@pytest.fixture
def provider() -> GroqTTSProvider:
    return GroqTTSProvider(api_key="gsk_fake_key_for_tests")


@pytest.mark.asyncio
async def test_synthesize_returns_audio_bytes(provider: GroqTTSProvider) -> None:
    fake_response = AsyncMock()
    fake_response.read = AsyncMock(return_value=b"RIFF....WAVEfmt fake audio data")
    provider._client.audio.speech.create = AsyncMock(return_value=fake_response)

    audio = await provider.synthesize("Hello there.")

    assert audio.startswith(b"RIFF")


@pytest.mark.asyncio
async def test_synthesize_empty_text_skips_api_call(provider: GroqTTSProvider) -> None:
    mock_create = AsyncMock()
    provider._client.audio.speech.create = mock_create

    audio = await provider.synthesize("   ")

    assert audio == b""
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_synthesize_uses_configured_model_and_voice() -> None:
    provider = GroqTTSProvider(
        api_key="gsk_fake", model="canopylabs/orpheus-v1-english", voice="troy"
    )
    fake_response = AsyncMock()
    fake_response.read = AsyncMock(return_value=b"audio")
    mock_create = AsyncMock(return_value=fake_response)
    provider._client.audio.speech.create = mock_create

    await provider.synthesize("Hi")

    _, kwargs = mock_create.call_args
    assert kwargs["model"] == "canopylabs/orpheus-v1-english"
    assert kwargs["voice"] == "troy"
    assert kwargs["input"] == "Hi"


@pytest.mark.asyncio
async def test_synthesize_wraps_sdk_errors_with_helpful_message(provider: GroqTTSProvider) -> None:
    provider._client.audio.speech.create = AsyncMock(
        side_effect=RuntimeError("429 rate limit exceeded")
    )

    with pytest.raises(VoiceError) as exc_info:
        await provider.synthesize("A longer message than the free tier likes.")

    assert "character limit" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_synthesize_detects_terms_not_accepted(provider: GroqTTSProvider) -> None:
    provider._client.audio.speech.create = AsyncMock(
        side_effect=RuntimeError(
            "Error code: 400 - {'error': {'message': 'The model `x` requires terms "
            "acceptance', 'code': 'model_terms_required'}}"
        )
    )

    with pytest.raises(VoiceError) as exc_info:
        await provider.synthesize("Hi")

    assert "accept" in exc_info.value.user_message
    assert "playground" in exc_info.value.user_message


@pytest.mark.asyncio
async def test_speak_skips_playback_for_empty_audio(provider: GroqTTSProvider) -> None:
    fake_response = AsyncMock()
    fake_response.read = AsyncMock(return_value=b"")
    provider._client.audio.speech.create = AsyncMock(return_value=fake_response)

    await provider.speak("   ")  # empty text -> synthesize returns b"" -> no playback attempted


def test_missing_dependency_raises_voice_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "groq":
            raise ImportError("no module named groq")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(VoiceError):
        GroqTTSProvider(api_key="gsk_fake")
