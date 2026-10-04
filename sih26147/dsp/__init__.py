from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .carrier import estimate_carrier_offset
from .filters import rrc_taps
from .mapping import qpsk_gray_indices
from .timing import estimate_timing_offset

_QPSK_BIT_SHIFTS = np.asarray([1, 0], dtype=np.int64)


@dataclass(frozen=True)
class DemodConfig:
    scheme: str = "QPSK"
    symbol_rate_hz: float = 6000.0
    sample_rate_hz: float | None = None
    samples_per_symbol: int | None = None
    rolloff: float = 0.35
    gray: bool = True
    differential: bool = False
    carrier_offset_hz: float | None = None
    timing_offset_samples: float | None = None
    phase_reference_bits: tuple[int, ...] | None = None


@dataclass
class DemodResult:
    symbols: np.ndarray
    bits: np.ndarray
    llrs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    timing_trace: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    freq_trace: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    lock: bool = False
    evm: float | None = None
    warnings: tuple[str, ...] = ()


def demodulate_chain(iq: np.ndarray, fs: float, cfg: DemodConfig) -> DemodResult:
    """Matched-filter and hard-decision QPSK with coarse carrier/timing recovery."""
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be positive.")
    x = np.asarray(iq, dtype=np.complex128).reshape(-1)
    if x.size == 0:
        raise ValueError("Demodulation requires at least one IQ sample.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite samples.")

    symbol_rate_hz = float(cfg.symbol_rate_hz)
    if not np.isfinite(symbol_rate_hz) or symbol_rate_hz <= 0:
        raise ValueError("Symbol rate must be positive.")
    if cfg.samples_per_symbol is None:
        sps = max(2, int(round(fs / symbol_rate_hz)))
    elif isinstance(cfg.samples_per_symbol, (int, np.integer)):
        sps = int(cfg.samples_per_symbol)
    else:
        raise ValueError("samples_per_symbol must be an integer.")
    if sps < 2:
        raise ValueError("At least two samples per symbol are required.")
    if not np.isclose(fs / sps, symbol_rate_hz, rtol=0.01):
        raise ValueError("This receiver requires an integer samples-per-symbol ratio.")
    if cfg.sample_rate_hz is not None and not np.isclose(cfg.sample_rate_hz, fs):
        raise ValueError("Configured and supplied sample rates do not match.")
    if cfg.differential:
        raise ValueError("Differential QPSK is not supported by this receiver.")
    if cfg.timing_offset_samples is not None and (
        not np.isfinite(cfg.timing_offset_samples)
        or not 0.0 <= cfg.timing_offset_samples < sps
    ):
        raise ValueError("Timing offset must be finite and lie within one symbol.")
    if cfg.phase_reference_bits is not None and (
        not cfg.phase_reference_bits
        or len(cfg.phase_reference_bits) % 2
        or any(bit not in (0, 1) for bit in cfg.phase_reference_bits)
    ):
        raise ValueError("Phase-reference bits must be a non-empty, even-length binary sequence.")

    baseline = x - np.mean(x)
    scale = max(np.std(baseline), 1e-12)
    normalized = baseline / scale

    if cfg.scheme != "QPSK":
        raise ValueError(f"Unsupported scheme in demodulate_chain: {cfg.scheme!r}")

    carrier_hz = cfg.carrier_offset_hz
    if carrier_hz is None:
        carrier_hz = estimate_carrier_offset(normalized, fs, scheme=cfg.scheme)
    if not np.isfinite(carrier_hz):
        raise ValueError("Carrier offset must be finite.")
    qpsk_values = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * np.arange(4)))
    sample_indices = np.arange(normalized.size, dtype=np.float64)
    corrected = normalized * np.exp(-2j * np.pi * carrier_hz * sample_indices / fs)
    taps = rrc_taps(sps, cfg.rolloff)
    matched = np.convolve(corrected, taps, mode="same")
    group_delay = (taps.size - 1) // 2
    matched_indices = np.arange(matched.size, dtype=np.float64)
    best_timing = cfg.timing_offset_samples
    best_symbols: np.ndarray | None = None
    best_score = np.inf

    timing_candidates = (
        [float(cfg.timing_offset_samples)]
        if cfg.timing_offset_samples is not None
        else np.linspace(0.0, sps, 33, endpoint=False)
    )
    for timing_offset in timing_candidates:
        symbol_count = int(
            np.floor((normalized.size - 1 - 2 * group_delay - timing_offset) / sps) + 1
        )
        if symbol_count < 8:
            continue
        positions = group_delay + timing_offset + np.arange(symbol_count) * sps
        valid = (positions >= 0) & (positions <= matched.size - 1)
        positions = positions[valid]
        symbols = np.interp(positions, matched_indices, matched.real)
        symbols = symbols + 1j * np.interp(
            positions, matched_indices, matched.imag
        )
        symbols /= max(float(np.sqrt(np.mean(np.abs(symbols) ** 2))), 1e-12)

        # QPSK carrier recovery has a 90-degree ambiguity without a known preamble.
        fourth_moment_phase = float(np.angle(np.mean(symbols**4)))
        phase = (fourth_moment_phase - np.pi) / 4.0
        phase = (phase + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
        symbols *= np.exp(-1j * phase)
        nearest = np.argmin(np.abs(symbols[:, None] - qpsk_values[None, :]), axis=1)
        score = float(np.mean(np.abs(symbols - qpsk_values[nearest]) ** 2))
        if score < best_score:
            best_score = score
            best_timing = float(timing_offset)
            best_symbols = symbols.copy()

    if best_symbols is None:
        raise ValueError("No valid symbol alignment was found for the requested modulation.")
    symbols = best_symbols
    phase_resolved = cfg.phase_reference_bits is not None
    if phase_resolved:
        reference = np.asarray(cfg.phase_reference_bits, dtype=np.uint8)
        symbol_count = reference.size // 2
        if symbol_count > symbols.size:
            raise ValueError("Phase-reference bits exceed the recovered symbol count.")
        best_rotation = 0
        best_errors = reference.size + 1
        rotation_errors: list[int] = []
        for rotation in range(4):
            candidate = symbols * np.exp(-1j * rotation * np.pi / 2.0)
            candidate_indices = np.argmin(
                np.abs(candidate[:, None] - qpsk_values[None, :]), axis=1
            )
            if cfg.gray:
                candidate_bits = qpsk_gray_indices(candidate_indices)
            else:
                candidate_bits = (
                    (candidate_indices[:, None] >> _QPSK_BIT_SHIFTS) & 1
                ).astype(np.uint8).reshape(-1)
            errors = int(np.count_nonzero(candidate_bits[: reference.size] != reference))
            rotation_errors.append(errors)
            if errors < best_errors:
                best_rotation = rotation
                best_errors = errors
        second_best_errors = sorted(rotation_errors)[1]
        if best_errors > reference.size * 0.2 or (
            second_best_errors - best_errors < max(2, reference.size * 0.1)
        ):
            raise ValueError("Phase-reference bits do not reliably resolve the QPSK ambiguity.")
        symbols *= np.exp(-1j * best_rotation * np.pi / 2.0)
    nearest = np.argmin(np.abs(symbols[:, None] - qpsk_values[None, :]), axis=1)
    if cfg.gray:
        bits = qpsk_gray_indices(nearest)
    else:
        bits = ((nearest[:, None] >> _QPSK_BIT_SHIFTS) & 1).astype(np.uint8).reshape(-1)
    evm = float(np.sqrt(np.mean(np.abs(symbols - qpsk_values[nearest]) ** 2)))

    return DemodResult(
        symbols=symbols,
        bits=bits,
        timing_trace=np.asarray([best_timing], dtype=np.float64),
        freq_trace=np.asarray([carrier_hz], dtype=np.float64),
        lock=bool(evm < 0.5),
        evm=evm,
        warnings=(
            ()
            if phase_resolved
            else ("QPSK phase ambiguity remains unresolved without phase-reference bits.",)
        ),
    )


__all__ = [
    "DemodConfig",
    "DemodResult",
    "demodulate_chain",
    "estimate_carrier_offset",
    "estimate_timing_offset",
    "rrc_taps",
]
