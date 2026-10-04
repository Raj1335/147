from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


SUPPORTED_RAW_FORMATS = ("cf32", "ci16", "cu8")
DEFAULT_SAMPLE_LIMIT = 1_000_000


class CaptureError(ValueError):
    """Raised when a capture cannot be interpreted safely."""


@dataclass(frozen=True)
class Capture:
    samples: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float | None
    source_name: str
    source_format: str
    total_samples: int | None
    first_sample: int
    truncated: bool


def _read_pcm_samples(stream: BinaryIO | wave.Wave_read, width: int, count: int) -> np.ndarray:
    raw = stream.readframes(count) if isinstance(stream, wave.Wave_read) else stream.read(count * width)
    return _decode_pcm_bytes(raw, width)


def _decode_pcm_bytes(raw: bytes, width: int) -> np.ndarray:
    if not raw:
        return np.empty(0, dtype=np.float32)
    if width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        return values / 128.0
    if width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return values / 32768.0
    if width == 3:
        triples = np.frombuffer(raw, dtype=np.uint8)
        packed = triples[: len(triples) // 3 * 3].reshape(-1, 3)
        signed_values = (
            packed[:, 0].astype(np.int32)
            | (packed[:, 1].astype(np.int32) << 8)
            | (packed[:, 2].astype(np.int32) << 16)
        )
        signed_values = (signed_values ^ 0x800000) - 0x800000
        return signed_values.astype(np.float32) / 8388608.0
    if width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32)
        return values / 2147483648.0
    raise CaptureError(f"Unsupported PCM sample width: {width} bytes.")


def read_capture(
    source: BinaryIO | bytes,
    *,
    source_name: str,
    raw_format: str = "cf32",
    sample_rate_hz: float | None = None,
    center_frequency_hz: float | None = None,
    first_sample: int = 0,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
) -> Capture:
    """Read a bounded sample window from PCM WAV or interleaved raw IQ."""
    if first_sample < 0:
        raise CaptureError("First sample must be zero or greater.")
    if sample_limit < 2:
        raise CaptureError("Sample limit must be at least two.")
    if center_frequency_hz is not None and (
        not np.isfinite(center_frequency_hz) or center_frequency_hz < 0
    ):
        raise CaptureError("Center frequency must be a finite value of zero or greater.")
    if isinstance(source, bytes):
        stream: BinaryIO = io.BytesIO(source)
    else:
        stream = source
    stream.seek(0)

    suffix = source_name.lower().rsplit(".", 1)[-1] if "." in source_name else ""
    if suffix == "wav":
        return _read_wav(
            stream,
            source_name=source_name,
            first_sample=first_sample,
            sample_limit=sample_limit,
            center_frequency_hz=center_frequency_hz,
        )
    if suffix not in {"iq", "bin", "dat"}:
        raise CaptureError("Choose a .wav, .iq, .bin, or .dat capture.")
    if raw_format not in SUPPORTED_RAW_FORMATS:
        raise CaptureError(f"Raw format must be one of {', '.join(SUPPORTED_RAW_FORMATS)}.")
    if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise CaptureError("A positive sample rate is required for raw IQ data.")

    dtype, bytes_per_component, scale, offset = {
        "cf32": ("<f4", 4, 1.0, 0.0),
        "ci16": ("<i2", 2, 32768.0, 0.0),
        "cu8": ("u1", 1, 128.0, 128.0),
    }[raw_format]
    if not stream.seekable():
        raise CaptureError("Raw IQ input must be seekable so its size and sample alignment can be validated.")
    byte_count = _stream_size(stream)
    if byte_count % (2 * bytes_per_component):
        raise CaptureError("Raw IQ file ends with an incomplete I/Q sample pair.")
    component_count = byte_count // bytes_per_component
    total_samples = component_count // 2
    if total_samples < 2:
        raise CaptureError("Raw IQ file must contain at least two I/Q sample pairs.")
    if first_sample >= total_samples:
        raise CaptureError("First sample is beyond the end of the raw IQ file.")

    count = min(sample_limit, total_samples - first_sample)
    stream.seek(first_sample * 2 * bytes_per_component)
    raw = stream.read(count * 2 * bytes_per_component)
    values = np.frombuffer(raw, dtype=np.dtype(dtype))
    values = (values.astype(np.float32) - offset) / scale
    if raw_format == "cf32" and not np.all(np.isfinite(values)):
        raise CaptureError("Raw float IQ data contains NaN or infinite values.")
    samples = (values[0::2] + 1j * values[1::2]).astype(np.complex64)
    if samples.size < 2:
        raise CaptureError("Raw IQ file did not contain a complete sample pair.")
    return Capture(
        samples=samples,
        sample_rate_hz=float(sample_rate_hz),
        center_frequency_hz=center_frequency_hz,
        source_name=source_name,
        source_format=raw_format,
        total_samples=total_samples,
        first_sample=first_sample,
        truncated=first_sample + samples.size < total_samples,
    )


def _stream_size(stream: BinaryIO) -> int:
    current = stream.tell()
    stream.seek(0, io.SEEK_END)
    size = stream.tell()
    stream.seek(current)
    return size


def _read_wav(
    stream: BinaryIO,
    *,
    source_name: str,
    first_sample: int,
    sample_limit: int,
    center_frequency_hz: float | None,
) -> Capture:
    if _wav_format_tag(stream) == 3:
        return _read_float_wav(
            stream,
            source_name=source_name,
            first_sample=first_sample,
            sample_limit=sample_limit,
            center_frequency_hz=center_frequency_hz,
        )
    try:
        reader = wave.open(stream, "rb")
    except (wave.Error, EOFError) as exc:
        try:
            return _read_zero_size_pcm_wav(
                stream,
                source_name=source_name,
                first_sample=first_sample,
                sample_limit=sample_limit,
                center_frequency_hz=center_frequency_hz,
            )
        except CaptureError as fallback_error:
            raise CaptureError(
                f"Could not read PCM WAV header ({exc}); file-length fallback failed: {fallback_error}"
            ) from exc
    with reader:
        if reader.getcomptype() != "NONE":
            raise CaptureError("Compressed WAV is not supported; provide uncompressed PCM WAV.")
        channels = reader.getnchannels()
        rate = reader.getframerate()
        width = reader.getsampwidth()
        total = reader.getnframes()
        if channels not in (1, 2):
            raise CaptureError("WAV must be mono or stereo (stereo is interpreted as I/Q).")
        if rate <= 0:
            raise CaptureError("WAV header has an invalid sample rate.")
        if total < 2:
            return _read_zero_size_pcm_wav(
                stream,
                source_name=source_name,
                first_sample=first_sample,
                sample_limit=sample_limit,
                center_frequency_hz=center_frequency_hz,
            )
        if first_sample >= total:
            raise CaptureError("First sample is beyond the end of the WAV file.")
        reader.setpos(first_sample)
        count = min(sample_limit, total - first_sample)
        values = _read_pcm_samples(reader, width, count * channels)
        frames = values.size // channels
        frame_values = np.asarray(values[: frames * channels], dtype=np.float32).reshape(frames, channels)
        if channels == 2:
            samples = (frame_values[:, 0] + 1j * frame_values[:, 1]).astype(np.complex64)
            source_format = f"wav-pcm{width * 8}-iq"
        else:
            samples = frame_values[:, 0].astype(np.complex64)
            source_format = f"wav-pcm{width * 8}-mono"
    if samples.size < 2:
        raise CaptureError("WAV did not contain enough complete frames.")
    return Capture(
        samples=samples,
        sample_rate_hz=float(rate),
        center_frequency_hz=center_frequency_hz,
        source_name=source_name,
        source_format=source_format,
        total_samples=total,
        first_sample=first_sample,
        truncated=first_sample + samples.size < total,
    )


def _read_zero_size_pcm_wav(
    stream: BinaryIO,
    *,
    source_name: str,
    first_sample: int,
    sample_limit: int,
    center_frequency_hz: float | None,
) -> Capture:
    """Read PCM WAV recordings whose RIFF/data length fields are zero but bytes follow."""
    stream.seek(12)
    fmt: bytes | None = None
    data_offset: int | None = None
    declared_data_size: int | None = None
    while True:
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        chunk_offset = stream.tell()
        if chunk_id == b"fmt ":
            if chunk_size < 16 or chunk_size > 4096:
                raise CaptureError("WAV format chunk has an invalid size.")
            fmt = stream.read(chunk_size)
        elif chunk_id == b"data":
            data_offset = chunk_offset
            declared_data_size = chunk_size
            break
        stream.seek(chunk_offset + chunk_size + (chunk_size & 1))
    if fmt is None or data_offset is None or declared_data_size is None:
        raise CaptureError("WAV file is missing a format or data chunk.")
    format_tag, channels, rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    if format_tag == 0xFFFE and len(fmt) >= 40:
        format_tag = struct.unpack_from("<H", fmt, 24)[0]
    width = bits // 8
    if (
        format_tag != 1
        or channels not in (1, 2)
        or bits not in (8, 16, 24, 32)
        or block_align != channels * width
        or rate <= 0
    ):
        raise CaptureError("Zero-length WAV headers are supported only for PCM mono/stereo recordings.")
    data_size = declared_data_size
    if data_size in (0, 0xFFFFFFFF):
        data_size = _stream_size(stream) - data_offset
    if data_size < 0 or data_size % block_align:
        raise CaptureError("WAV data chunk has an invalid length.")
    total = data_size // block_align
    if total < 2:
        raise CaptureError("WAV header has too few frames.")
    if first_sample >= total:
        raise CaptureError("First sample is beyond the end of the WAV file.")
    count = min(sample_limit, total - first_sample)
    stream.seek(data_offset + first_sample * block_align)
    values = _decode_pcm_bytes(stream.read(count * block_align), width)
    frames = values.size // channels
    frame_values = np.asarray(values[: frames * channels], dtype=np.float32).reshape(frames, channels)
    if channels == 2:
        samples = (frame_values[:, 0] + 1j * frame_values[:, 1]).astype(np.complex64)
        source_format = f"wav-pcm{bits}-iq"
    else:
        samples = frame_values[:, 0].astype(np.complex64)
        source_format = f"wav-pcm{bits}-mono"
    if samples.size < 2:
        raise CaptureError("WAV did not contain enough complete frames.")
    return Capture(
        samples=samples,
        sample_rate_hz=float(rate),
        center_frequency_hz=center_frequency_hz,
        source_name=source_name,
        source_format=source_format,
        total_samples=total,
        first_sample=first_sample,
        truncated=first_sample + samples.size < total,
    )


def _wav_format_tag(stream: BinaryIO) -> int:
    stream.seek(0)
    header = stream.read(12)
    if len(header) != 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:] != b"WAVE":
        stream.seek(0)
        raise CaptureError("Expected a little-endian RIFF/RF64 WAVE file.")
    while True:
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        if chunk_id == b"fmt ":
            if chunk_size < 16 or chunk_size > 4096:
                break
            fmt = stream.read(chunk_size)
            stream.seek(chunk_size & 1, io.SEEK_CUR)
            tag = int(struct.unpack_from("<H", fmt)[0])
            if tag == 0xFFFE and chunk_size >= 40:
                tag = int(struct.unpack_from("<H", fmt, 24)[0])
            stream.seek(0)
            return tag
        stream.seek(chunk_size + (chunk_size & 1), io.SEEK_CUR)
    stream.seek(0)
    return 1


def _read_float_wav(
    stream: BinaryIO,
    *,
    source_name: str,
    first_sample: int,
    sample_limit: int,
    center_frequency_hz: float | None,
) -> Capture:
    stream.seek(12)
    fmt: bytes | None = None
    data_offset: int | None = None
    data_size: int | None = None
    while True:
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            break
        chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
        chunk_offset = stream.tell()
        if chunk_id == b"fmt ":
            if chunk_size < 16 or chunk_size > 4096:
                raise CaptureError("WAV format chunk has an invalid size.")
            fmt = stream.read(chunk_size)
        elif chunk_id == b"data":
            data_offset = chunk_offset
            data_size = chunk_size
        stream.seek(chunk_offset + chunk_size + (chunk_size & 1))
        if fmt is not None and data_offset is not None:
            break
    if fmt is None or data_offset is None or data_size is None:
        raise CaptureError("WAV file is missing a format or data chunk.")
    tag, channels, rate, _, block_align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 0xFFFE and len(fmt) >= 40:
        tag = int(struct.unpack_from("<H", fmt, 24)[0])
    if tag != 3 or bits != 32 or channels not in (1, 2) or block_align != channels * 4:
        raise CaptureError("Only float32 IEEE-float WAV or uncompressed PCM WAV is supported.")
    total = data_size // block_align
    if rate <= 0 or total < 2:
        raise CaptureError("WAV header has an invalid sample rate or too few frames.")
    if first_sample >= total:
        raise CaptureError("First sample is beyond the end of the WAV file.")
    count = min(sample_limit, total - first_sample)
    stream.seek(data_offset + first_sample * block_align)
    raw = stream.read(count * block_align)
    values = np.frombuffer(raw, dtype="<f4")
    if not np.all(np.isfinite(values)):
        raise CaptureError("WAV contains NaN or infinite samples.")
    frames = values.size // channels
    frame_values = np.asarray(values[: frames * channels], dtype=np.float32).reshape(frames, channels)
    samples = (
        (frame_values[:, 0] + 1j * frame_values[:, 1]).astype(np.complex64)
        if channels == 2
        else frame_values[:, 0].astype(np.complex64)
    )
    if samples.size < 2:
        raise CaptureError("WAV did not contain enough complete frames.")
    return Capture(
        samples=samples,
        sample_rate_hz=float(rate),
        center_frequency_hz=center_frequency_hz,
        source_name=source_name,
        source_format=f"wav-float32-{'iq' if channels == 2 else 'mono'}",
        total_samples=total,
        first_sample=first_sample,
        truncated=first_sample + samples.size < total,
    )
