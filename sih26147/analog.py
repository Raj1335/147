from __future__ import annotations

import io
import wave
from math import gcd

import numpy as np
from scipy import signal


def demodulate_analog(
    samples: np.ndarray,
    *,
    scheme: str,
    sample_rate_hz: float,
    cw_tone_hz: float = 700.0,
) -> np.ndarray:
    """Demodulate conventional AM, FM, or keyed-carrier CW to mono audio."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if x.size < 16 or not np.all(np.isfinite(x)):
        raise ValueError("Analog demodulation requires at least 16 finite IQ samples.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")

    if scheme == "AM":
        audio = np.abs(x)
        audio -= np.mean(audio)
    elif scheme == "FM":
        phase_steps = np.angle(x[1:] * np.conj(x[:-1]))
        audio = phase_steps * sample_rate_hz / (2.0 * np.pi)
        audio -= np.median(audio)
    elif scheme == "CW":
        if not np.isfinite(cw_tone_hz) or not 100 <= cw_tone_hz <= 2_000:
            raise ValueError("CW sidetone must be between 100 Hz and 2,000 Hz.")
        envelope = np.abs(x)
        low, high = np.percentile(envelope, [10, 90])
        if high - low <= np.finfo(float).eps * max(float(high), 1.0):
            raise ValueError("CW envelope has insufficient on/off keying contrast.")
        keyed = envelope > (low + high) / 2.0
        times = np.arange(keyed.size, dtype=np.float64) / sample_rate_hz
        audio = keyed * np.sin(2.0 * np.pi * cw_tone_hz * times)
    else:
        raise ValueError("Choose AM, FM, or CW.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Analog demodulation produced non-finite audio samples.")
    peak = float(np.max(np.abs(audio)))
    if peak <= np.finfo(float).tiny:
        raise ValueError("Capture produced no recoverable analog audio.")
    return (audio / peak).astype(np.float32)


def audio_to_wav(
    samples: np.ndarray,
    *,
    source_sample_rate_hz: float,
    output_sample_rate_hz: int = 12_000,
) -> bytes:
    """Resample mono audio with anti-alias filtering and encode PCM16 WAV."""
    audio = np.asarray(samples, dtype=np.float64).reshape(-1)
    if audio.size < 1 or not np.all(np.isfinite(audio)):
        raise ValueError("Audio must contain finite samples.")
    if not np.isfinite(source_sample_rate_hz) or source_sample_rate_hz <= 0:
        raise ValueError("Source sample rate must be positive.")
    if not isinstance(output_sample_rate_hz, (int, np.integer)) or output_sample_rate_hz < 1:
        raise ValueError("Output sample rate must be a positive integer.")

    source_rate = int(round(source_sample_rate_hz))
    if source_rate < 1:
        raise ValueError("Source sample rate must be at least one sample per second.")
    divisor = gcd(source_rate, int(output_sample_rate_hz))
    if source_rate != output_sample_rate_hz:
        audio = signal.resample_poly(
            audio,
            int(output_sample_rate_hz) // divisor,
            source_rate // divisor,
        )

    peak = float(np.max(np.abs(audio)))
    if peak <= np.finfo(float).tiny:
        raise ValueError("Cannot export silent audio.")
    pcm = np.rint(np.clip(audio / peak, -1.0, 1.0) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(int(output_sample_rate_hz))
        writer.writeframes(pcm.tobytes())
    return output.getvalue()
