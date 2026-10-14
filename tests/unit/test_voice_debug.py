from __future__ import annotations

import array
import wave
from pathlib import Path

from app.voice.debug import rms_energy, save_debug_wav


def _tone(amplitude: int = 10000, length: int = 800) -> bytes:
    samples = array.array("h", [amplitude if i % 2 == 0 else -amplitude for i in range(length)])
    return samples.tobytes()


def _silence(length: int = 800) -> bytes:
    return array.array("h", [0] * length).tobytes()


def test_rms_energy_of_silence_is_near_zero() -> None:
    assert rms_energy(_silence()) == 0.0


def test_rms_energy_of_loud_tone_is_high() -> None:
    assert rms_energy(_tone(amplitude=10000)) > 5000


def test_rms_energy_scales_with_amplitude() -> None:
    quiet = rms_energy(_tone(amplitude=500))
    loud = rms_energy(_tone(amplitude=10000))
    assert loud > quiet


def test_rms_energy_empty_bytes_is_zero() -> None:
    assert rms_energy(b"") == 0.0


def test_save_debug_wav_writes_playable_file(tmp_path: Path) -> None:
    path = tmp_path / "debug.wav"
    audio = _tone()

    save_debug_wav(path, audio, sample_rate=16_000)

    assert path.exists()
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16_000
        assert wav_file.readframes(wav_file.getnframes()) == audio


def test_save_debug_wav_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "debug.wav"
    save_debug_wav(path, _tone(amplitude=100))
    save_debug_wav(path, _tone(amplitude=200))

    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    assert frames == _tone(amplitude=200)
