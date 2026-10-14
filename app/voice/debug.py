"""Diagnostic helpers for the voice pipeline.

These exist to answer one question when voice input isn't working:
"did the microphone actually capture anything?" — a near-silent or all-zero
recording is the classic cause of Whisper hallucinating a stock phrase like
"Thank you." regardless of what was actually said, and it's very easy to
mistake that hallucination for a bug in NOVA's code rather than in the
audio capture itself (wrong default input device, muted mic, volume at 0).
"""

from __future__ import annotations

import array
import wave
from pathlib import Path


def rms_energy(pcm16_bytes: bytes) -> float:
    """Root-mean-square amplitude of raw PCM16 mono audio. ~0 means silence
    (or a broken capture path); real speech is usually in the low
    hundreds-to-thousands range depending on mic gain."""
    if not pcm16_bytes:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm16_bytes[: len(pcm16_bytes) - (len(pcm16_bytes) % 2)])
    if not samples:
        return 0.0
    return float((sum(s * s for s in samples) / len(samples)) ** 0.5)


def save_debug_wav(path: str | Path, pcm16_bytes: bytes, sample_rate: int = 16_000) -> None:
    """Write raw PCM16 mono audio to a playable .wav file for manual
    inspection (open it in any media player)."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16_bytes)
