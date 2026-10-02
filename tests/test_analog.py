from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from scipy import signal

from sih26147.analog import audio_to_wav, demodulate_analog


def test_am_demodulation_recovers_known_audio_tone() -> None:
    sample_rate = 48_000
    duration = 0.25
    time = np.arange(int(sample_rate * duration)) / sample_rate
    modulating = 0.6 * np.sin(2 * np.pi * 1_000 * time)
    received = (1.0 + modulating) * np.exp(1j * (2 * np.pi * 3_000 * time + 0.2))
    recovered = demodulate_analog(received, scheme="AM", sample_rate_hz=sample_rate)
    assert np.corrcoef(recovered, modulating)[0, 1] > 0.999


def test_fm_demodulation_recovers_known_audio_tone() -> None:
    sample_rate = 48_000
    duration = 0.25
    tone_hz = 700
    deviation_hz = 3_000
    time = np.arange(int(sample_rate * duration)) / sample_rate
    phase = 2 * np.pi * deviation_hz * np.cumsum(
        np.sin(2 * np.pi * tone_hz * time)
    ) / sample_rate
    received = np.exp(1j * phase)
    recovered = demodulate_analog(received, scheme="FM", sample_rate_hz=sample_rate)
    target = np.sin(2 * np.pi * tone_hz * time)[1:]
    assert np.corrcoef(recovered, target)[0, 1] > 0.999


def test_cw_demodulation_creates_tone_during_key_down() -> None:
    sample_rate = 12_000
    keyed = np.repeat(np.asarray([0.0, 1.0, 1.0, 0.0]), sample_rate // 4)
    time = np.arange(keyed.size) / sample_rate
    received = keyed * np.exp(1j * 2 * np.pi * 2_000 * time)
    recovered = demodulate_analog(received, scheme="CW", sample_rate_hz=sample_rate)
    assert np.max(np.abs(recovered[: sample_rate // 4])) == 0
    assert np.max(np.abs(recovered[sample_rate // 4 : sample_rate // 2])) > 0.9
    assert np.max(np.abs(recovered[sample_rate // 2 : 3 * sample_rate // 4])) > 0.9


def test_audio_export_resamples_to_mono_pcm16_wav() -> None:
    source_rate = 48_000
    time = np.arange(source_rate // 2) / source_rate
    audio = np.sin(2 * np.pi * 1_000 * time)
    encoded = audio_to_wav(
        audio,
        source_sample_rate_hz=source_rate,
        output_sample_rate_hz=12_000,
    )
    with wave.open(io.BytesIO(encoded), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 12_000
        decoded = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    assert decoded.size == 6_000
    frequencies, power = signal.periodogram(decoded.astype(float), fs=12_000)
    assert frequencies[np.argmax(power)] == pytest.approx(1_000, abs=5)


def test_analog_demodulation_rejects_invalid_signal() -> None:
    with pytest.raises(ValueError, match="Choose AM, FM, or CW"):
        demodulate_analog(np.ones(32), scheme="ASK", sample_rate_hz=8_000)
    with pytest.raises(ValueError, match="insufficient on/off"):
        demodulate_analog(np.ones(32), scheme="CW", sample_rate_hz=8_000)
