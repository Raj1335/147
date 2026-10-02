from __future__ import annotations

import numpy as np

from .estimators import ModulationEstimate, _constellations, _samples


def demodulate(
    samples: np.ndarray,
    *,
    scheme: str,
    samples_per_symbol: int,
    fsk_deviation_hz: float | None = None,
    sample_rate_hz: float | None = None,
) -> np.ndarray:
    """Hard-decision educational demodulator; assumes timing/carrier are already corrected."""
    if not isinstance(samples_per_symbol, (int, np.integer)) or samples_per_symbol < 1:
        raise ValueError("Samples per symbol must be a positive integer.")
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite samples.")
    if x.size < samples_per_symbol:
        raise ValueError("Capture is shorter than one requested symbol.")
    symbols = x[samples_per_symbol // 2 :: samples_per_symbol]
    symbols = symbols[: min(len(symbols), 1_000_000)]
    if scheme == "BPSK":
        return (symbols.real < 0).astype(np.uint8)
    if scheme in {"QPSK", "8PSK"}:
        order = 4 if scheme == "QPSK" else 8
        phase = np.mod(np.angle(symbols), 2.0 * np.pi)
        labels = np.rint(phase * order / (2.0 * np.pi)).astype(int) % order
        return _integers_to_bits(labels, int(np.log2(order)))
    if scheme == "16QAM":
        symbols /= max(float(np.sqrt(np.mean(np.abs(symbols) ** 2))), 1e-12)
        levels = np.asarray([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        i_levels = levels[np.argmin(np.abs(symbols.real[:, None] - levels), axis=1)]
        q_levels = levels[np.argmin(np.abs(symbols.imag[:, None] - levels), axis=1)]
        i_labels = np.searchsorted(levels, i_levels)
        q_labels = np.searchsorted(levels, q_levels)
        i_bits = _integers_to_bits(i_labels, 2).reshape(-1, 2)
        q_bits = _integers_to_bits(q_labels, 2).reshape(-1, 2)
        return np.column_stack((i_bits, q_bits)).reshape(-1)
    if scheme == "2FSK":
        if (
            fsk_deviation_hz is None
            or not np.isfinite(fsk_deviation_hz)
            or fsk_deviation_hz <= 0
        ):
            raise ValueError("2FSK requires a positive tone deviation in Hz.")
        if sample_rate_hz is None or not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("A positive sample rate is required for FSK.")
        phase_step = np.angle(x[1:] * np.conj(x[:-1]))
        instantaneous_frequency = phase_step * sample_rate_hz / (2.0 * np.pi)
        symbol_freq = instantaneous_frequency[samples_per_symbol // 2 :: samples_per_symbol]
        symbol_freq = symbol_freq[: len(symbols)]
        low_tone_distance = np.abs(symbol_freq + fsk_deviation_hz)
        high_tone_distance = np.abs(symbol_freq - fsk_deviation_hz)
        return (high_tone_distance < low_tone_distance).astype(np.uint8)
    raise ValueError("Supported hard-decision modes: BPSK, QPSK, 8PSK, 16QAM, 2FSK.")


def demodulate_from_estimate(
    samples: np.ndarray,
    estimate: ModulationEstimate,
    sample_rate_hz: float,
) -> np.ndarray:
    """Recover carrier/timing and demap a supported estimated PSK/QAM constellation."""
    if estimate.modulation not in {"BPSK", "QPSK", "8PSK", "16QAM", "2FSK"}:
        raise ValueError("Automatic hard demodulation supports 2FSK/BPSK/QPSK/8PSK/16QAM.")
    if estimate.modulation == "2FSK":
        if (
            estimate.samples_per_symbol is None
            or estimate.timing_offset_samples is None
            or estimate.fsk_deviation_hz is None
        ):
            raise ValueError("The automatic 2FSK estimate lacks timing or tone-deviation parameters.")
        x = _samples(samples)
        if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("Sample rate must be positive.")
        phase = np.angle(x[1:] * np.conj(x[:-1]))
        instantaneous_frequency = phase * sample_rate_hz / (2.0 * np.pi)
        spacing = estimate.samples_per_symbol
        count = min(1_000_000, int((len(x) - estimate.timing_offset_samples) / spacing))
        if count < 1:
            raise ValueError("No symbols remain after applying the estimated timing phase.")
        output = np.empty(count, dtype=np.uint8)
        for index in range(count):
            start = int(round(estimate.timing_offset_samples + index * spacing))
            end = min(
                len(instantaneous_frequency),
                int(round(estimate.timing_offset_samples + (index + 1) * spacing)),
            )
            mean_frequency = float(np.mean(instantaneous_frequency[start:end]))
            output[index] = mean_frequency >= estimate.frequency_offset_hz
        return output
    if (
        estimate.samples_per_symbol is None
        or estimate.timing_offset_samples is None
        or estimate.frequency_offset_hz is None
        or estimate.phase_rotation_rad is None
    ):
        raise ValueError("The automatic estimate lacks a valid carrier/timing solution.")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    x = _samples(samples)
    indices = np.arange(x.size, dtype=np.float64)
    x = x * np.exp(-2j * np.pi * estimate.frequency_offset_hz * indices / sample_rate_hz)
    spacing = estimate.samples_per_symbol
    timing_offset = estimate.timing_offset_samples
    count = min(
        1_000_000,
        max(0, int(np.floor((x.size - 1 - timing_offset) / spacing)) + 1),
    )
    if count < 1:
        raise ValueError("No symbols remain after applying the estimated timing phase.")
    times = timing_offset + np.arange(count) * spacing
    symbols = np.interp(times, indices, x.real) + 1j * np.interp(times, indices, x.imag)
    symbols *= np.exp(1j * estimate.phase_rotation_rad)
    symbols /= max(float(np.sqrt(np.mean(np.abs(symbols) ** 2))), 1e-12)
    if estimate.modulation == "16QAM":
        levels = np.asarray([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(10.0)
        i_index = np.argmin(np.abs(symbols.real[:, None] - levels), axis=1)
        q_index = np.argmin(np.abs(symbols.imag[:, None] - levels), axis=1)
        i_bits = _integers_to_bits(i_index, 2).reshape(-1, 2)
        q_bits = _integers_to_bits(q_index, 2).reshape(-1, 2)
        return np.column_stack(
            (i_bits, q_bits)
        ).reshape(-1)
    constellation = _constellations()[estimate.modulation]
    decisions = np.argmin(np.abs(symbols[:, None] - constellation[None, :]), axis=1)
    width = int(np.log2(constellation.size))
    return _integers_to_bits(decisions, width)


def correlate_bitstreams(reference: str, candidate: str, max_offset: int = 4096) -> dict[str, float | int]:
    """Find the lowest BER over a bounded non-negative alignment offset."""
    if max_offset < 0:
        raise ValueError("Maximum alignment offset must be zero or greater.")
    a = _parse_bits(reference)
    b = _parse_bits(candidate)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both bit streams must contain at least one bit.")
    best = (1.0, 0, 0)
    limit = min(max_offset, b.size - 1)
    for offset in range(limit + 1):
        count = min(a.size, b.size - offset)
        errors = int(np.count_nonzero(a[:count] != b[offset : offset + count]))
        ber = errors / count
        if ber < best[0]:
            best = (ber, offset, count)
            if ber == 0:
                break
    return {"bit_error_rate": best[0], "offset_bits": best[1], "compared_bits": best[2]}


def bits_to_bytes(bits: np.ndarray) -> bytes:
    raw = np.asarray(bits).reshape(-1)
    if raw.size == 0 or not np.all(np.isin(raw, (0, 1))):
        raise ValueError("Bit stream must contain only zeroes and ones.")
    normalized = raw.astype(np.uint8)
    if normalized.size % 8:
        raise ValueError("Bit count must be a multiple of eight.")
    return np.packbits(normalized, bitorder="big").tobytes()


def _parse_bits(bits: str) -> np.ndarray:
    cleaned = "".join(bits.split())
    if not cleaned or set(cleaned) - {"0", "1"}:
        raise ValueError("Bit streams must contain only 0 and 1.")
    return np.fromiter((char == "1" for char in cleaned), dtype=np.uint8)


def _integers_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint32).reshape(-1)
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint32)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
