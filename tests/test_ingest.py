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


def _pcm_wav_bytes(
    payload: bytes,
    *,
    width: int = 2,
    channels: int = 1,
    rate: int = 48_000,
    block_align: int | None = None,
    format_tag: int = 1,
) -> bytes:
    alignment = channels * width if block_align is None else block_align
    fmt = struct.pack(
        "<HHIIHH",
        format_tag,
        channels,
        rate,
        rate * alignment,
        alignment,
        width * 8,
    )
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", len(body)) + body


class _NonSeekableBytesIO(io.BytesIO):
    def seekable(self) -> bool:
        return False


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


@pytest.mark.parametrize(
    ("source", "source_name", "options", "error"),
    [
        (b"\0" * 16, "capture.iq", {"first_sample": -1}, "First sample"),
        (b"\0" * 16, "capture.iq", {"sample_limit": 1}, "Sample limit"),
        (b"\0" * 16, "capture.iq", {"center_frequency_hz": -1}, "Center frequency"),
        (b"\0" * 16, "capture.txt", {}, "Choose a"),
        (b"\0" * 16, "capture.iq", {"raw_format": "bad"}, "Raw format"),
        (b"\0" * 16, "capture.iq", {}, "sample rate is required"),
        (b"\0" * 3, "capture.iq", {"raw_format": "ci16", "sample_rate_hz": 1}, "incomplete I/Q"),
        (b"\0" * 8, "capture.iq", {"raw_format": "cf32", "sample_rate_hz": 1}, "at least two"),
        (
            np.asarray([np.nan, 0, 0, 0], dtype="<f4").tobytes(),
            "capture.iq",
            {"raw_format": "cf32", "sample_rate_hz": 1},
            "NaN or infinite",
        ),
    ],
)
def test_raw_capture_rejects_invalid_inputs(
    source: bytes, source_name: str, options: dict[str, object], error: str
) -> None:
    with pytest.raises(CaptureError, match=error):
        read_capture(source, source_name=source_name, **options)


def test_raw_capture_rejects_non_seekable_stream_and_out_of_range_start() -> None:
    data = np.asarray([1, 2, 3, 4], dtype="<i2").tobytes()
    with pytest.raises(CaptureError, match="must be seekable"):
        read_capture(
            _NonSeekableBytesIO(data),
            source_name="capture.iq",
            raw_format="ci16",
            sample_rate_hz=1,
        )
    with pytest.raises(CaptureError, match="beyond the end"):
        read_capture(
            data,
            source_name="capture.iq",
            raw_format="ci16",
            sample_rate_hz=1,
            first_sample=2,
        )


@pytest.mark.parametrize(
    ("payload", "width", "expected"),
    [
        (bytes([0, 128, 255, 255]), 1, [-1.0, 0.0, 127 / 128, 127 / 128]),
        (
            bytes([0, 0, 128, 255, 255, 127]),
            3,
            [-1.0, 8_388_607 / 8_388_608],
        ),
        (
            np.asarray([-2_147_483_648, 2_147_483_647], dtype="<i4").tobytes(),
            4,
            [-1.0, 2_147_483_647 / 2_147_483_648],
        ),
    ],
)
def test_pcm_wav_supports_additional_sample_widths(
    payload: bytes, width: int, expected: list[float]
) -> None:
    capture = read_capture(
        _pcm_wav_bytes(payload, width=width),
        source_name="width.wav",
    )
    assert capture.samples.real.tolist() == pytest.approx(expected)


def test_pcm_wav_rejects_invalid_channel_count_sample_rate_and_width() -> None:
    stereo_data = _pcm_wav_bytes(b"\0" * 12, channels=3)
    with pytest.raises(CaptureError, match="mono or stereo"):
        read_capture(stereo_data, source_name="channels.wav")
    zero_rate = _pcm_wav_bytes(b"\0" * 4, rate=0)
    with pytest.raises(CaptureError, match="invalid sample rate"):
        read_capture(zero_rate, source_name="rate.wav")
    unsupported_width = _pcm_wav_bytes(b"\0" * 12, width=6)
    with pytest.raises(CaptureError, match="Unsupported PCM sample width"):
        read_capture(unsupported_width, source_name="width.wav")


def test_pcm_wav_rejects_bad_format_and_out_of_range_start() -> None:
    compressed = _pcm_wav_bytes(b"\0" * 4, format_tag=6)
    with pytest.raises(CaptureError, match="Compressed WAV"):
        read_capture(compressed, source_name="compressed.wav")
    valid = _wav_bytes(np.arange(4, dtype=np.int16))
    with pytest.raises(CaptureError, match="beyond the end"):
        read_capture(valid, source_name="past-end.wav", first_sample=4)
