from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import DemodConfig, rrc_taps
from .ingest import Capture


@dataclass(frozen=True)
class SyntheticDemo:
    capture: Capture
    reference_bits: np.ndarray
    demod_config: DemodConfig
    snr_db: float


def generate_qpsk_demo(seed: int = 26147) -> SyntheticDemo:
    """Generate a labeled synthetic QPSK capture for the app demo, never measured data."""
    sample_rate_hz = 48_000.0
    symbol_rate_hz = 6_000.0
    samples_per_symbol = 8
    rolloff = 0.35
    snr_db = 15.0
    timing_offset_samples = 3
    carrier_offset_hz = 150.0
    phase_offset_rad = 0.8

    rng = np.random.default_rng(seed)
    reference_bits = rng.integers(0, 2, size=1024, dtype=np.uint8)
    bit_pairs = reference_bits.reshape(-1, 2)
    gray_to_phase_index = np.asarray([[0, 1], [3, 2]], dtype=np.int64)
    phase_indices = gray_to_phase_index[bit_pairs[:, 0], bit_pairs[:, 1]]
    symbols = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * phase_indices))

    upsampled = np.zeros(symbols.size * samples_per_symbol, dtype=np.complex128)
    upsampled[::samples_per_symbol] = symbols
    pulse = rrc_taps(samples_per_symbol, rolloff=rolloff)
    shaped = np.convolve(upsampled, pulse, mode="full")
    delayed = np.concatenate(
        (np.zeros(timing_offset_samples, dtype=np.complex128), shaped)
    )
    time_s = np.arange(delayed.size, dtype=np.float64) / sample_rate_hz
    carrier = np.exp(
        1j * (2.0 * np.pi * carrier_offset_hz * time_s + phase_offset_rad)
    )
    signal = delayed * carrier
    noise_power = float(np.mean(np.abs(signal) ** 2) / (10.0 ** (snr_db / 10.0)))
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(signal.size) + 1j * rng.standard_normal(signal.size)
    )
    samples = (signal + noise).astype(np.complex64)

    config = DemodConfig(
        scheme="QPSK",
        symbol_rate_hz=symbol_rate_hz,
        sample_rate_hz=sample_rate_hz,
        samples_per_symbol=samples_per_symbol,
        rolloff=rolloff,
        gray=True,
        carrier_offset_hz=carrier_offset_hz,
        phase_reference_bits=tuple(int(bit) for bit in reference_bits[:64]),
    )
    capture = Capture(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        center_frequency_hz=None,
        source_name="synthetic-qpsk-demo.iq",
        source_format="synthetic-cf32-iq",
        total_samples=int(samples.size),
        first_sample=0,
        truncated=False,
    )
    return SyntheticDemo(
        capture=capture,
        reference_bits=reference_bits,
        demod_config=config,
        snr_db=snr_db,
    )
