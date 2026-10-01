from __future__ import annotations

import io
import struct
import wave

import numpy as np
import pytest

from sih26147.ingest import CaptureError, read_capture


def _wav_bytes(samples: np.ndarray, rate: int = 48_000) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(samples.shape[1] if samples.ndim == 2 else 1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.astype("<i2").tobytes())
    return output.getvalue()


def _float_wav_bytes(samples: np.ndarray, rate: int = 48_000) -> bytes:
    channel_count = samples.shape[1] if samples.ndim == 2 else 1
    payload = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, channel_count, rate, rate * channel_count * 4, channel_count * 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_raw_ci16_reads_iq_and_selected_window() -> None:
    iq = np.asarray([1000, -1000, 2000, -2000, 3000, -3000], dtype="<i2")
    capture = read_capture(
        iq.tobytes(),
        source_name="recording.iq",
        raw_format="ci16",
        sample_rate_hz=2_000_000,
        first_sample=0,
        sample_limit=2,
    )
    assert capture.samples.tolist() == pytest.approx(
        [1000 / 32768 - 1j * 1000 / 32768, 2000 / 32768 - 1j * 2000 / 32768]
    )
    assert capture.first_sample == 0
    assert capture.truncated


def test_raw_cu8_centers_unsigned_components() -> None:
    capture = read_capture(
        bytes([128, 128, 255, 0]),
        source_name="capture.bin",
        raw_format="cu8",
        sample_rate_hz=1_000_000,
    )
    assert capture.samples[0] == 0
    assert capture.samples[1].real == pytest.approx(127 / 128)
    assert capture.samples[1].imag == pytest.approx(-1)


def test_raw_rejects_unmatched_iq_component() -> None:
    with pytest.raises(CaptureError, match="incomplete I/Q"):
        read_capture(
            b"\0\0\0",
            source_name="capture.iq",
            raw_format="ci16",
            sample_rate_hz=1_000_000,
        )


def test_wav_stereo_is_iq_and_uses_header_sample_rate() -> None:
    pcm = np.asarray([[16_384, -16_384], [8_192, 8_192]], dtype=np.int16)
    capture = read_capture(_wav_bytes(pcm), source_name="capture.wav")
    assert capture.sample_rate_hz == 48_000
    assert capture.source_format == "wav-pcm16-iq"
    assert capture.samples[0] == pytest.approx(0.5 - 0.5j)


def test_wav_sample_window_is_bounded() -> None:
    pcm = np.arange(20, dtype=np.int16)
    capture = read_capture(
        _wav_bytes(pcm),
        source_name="capture.wav",
        first_sample=5,
        sample_limit=4,
    )
    assert capture.samples.size == 4
    assert capture.first_sample == 5
    assert capture.truncated


def test_wav_zero_length_fields_stream_from_file_length() -> None:
    pcm = np.asarray([[16_384, -16_384], [8_192, 8_192]], dtype=np.int16)
    malformed_size_header = bytearray(_wav_bytes(pcm))
    malformed_size_header[4:8] = b"\0\0\0\0"
    malformed_size_header[40:44] = b"\0\0\0\0"
    capture = read_capture(bytes(malformed_size_header), source_name="large-capture.wav")
    assert capture.total_samples == 2
    assert capture.samples[0] == pytest.approx(0.5 - 0.5j)


def test_float32_wav_is_read_as_ieee_float() -> None:
    samples = np.asarray([[0.25, -0.5], [-0.125, 0.75]], dtype=np.float32)
    capture = read_capture(_float_wav_bytes(samples), source_name="float.wav")
    assert capture.source_format == "wav-float32-iq"
    assert capture.samples.tolist() == pytest.approx([0.25 - 0.5j, -0.125 + 0.75j])


def test_float_wav_rejects_non_finite_samples() -> None:
    samples = np.asarray([0.25, np.nan], dtype=np.float32)
    with pytest.raises(CaptureError, match="NaN or infinite"):
        read_capture(_float_wav_bytes(samples), source_name="invalid.wav")
