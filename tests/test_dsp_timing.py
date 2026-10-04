from __future__ import annotations

import numpy as np
import pytest

from sih26147.dsp import estimate_carrier_offset, estimate_timing_offset


def test_timing_estimator_finds_qpsk_symbol_centers() -> None:
    rng = np.random.default_rng(11)
    symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, size=256)))
    samples = np.zeros(symbols.size * 8, dtype=np.complex128)
    samples[3::8] = symbols

    assert estimate_timing_offset(samples, 48_000.0, 6_000.0, 8) == pytest.approx(
        3.0, abs=0.5
    )


def test_timing_estimator_validates_rates_and_samples_per_symbol() -> None:
    samples = np.ones(32, dtype=np.complex128)
    with pytest.raises(ValueError, match="must be positive"):
        estimate_timing_offset(samples, 0.0, 6_000.0, 8)
    with pytest.raises(ValueError, match="at least two"):
        estimate_timing_offset(samples, 48_000.0, 6_000.0, 1)


def test_carrier_estimator_finds_qpsk_frequency_offset() -> None:
    rng = np.random.default_rng(23)
    symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, size=1024)))
    baseband = np.repeat(symbols, 8)
    time = np.arange(baseband.size) / 48_000.0
    offset_signal = baseband * np.exp(2j * np.pi * 150.0 * time)

    assert estimate_carrier_offset(offset_signal, 48_000.0) == pytest.approx(150.0, abs=2.0)


def test_carrier_estimator_handles_short_and_invalid_inputs() -> None:
    assert estimate_carrier_offset(np.ones(16), 48_000.0) == 0.0
    with pytest.raises(ValueError, match="Sample rate"):
        estimate_carrier_offset(np.ones(64), 0.0)
