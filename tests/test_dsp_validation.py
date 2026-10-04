from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sih26147.dsp import DemodConfig, demodulate_chain, rrc_taps
from sih26147.dsp.mapping import gray_bits_from_symbols


@pytest.mark.parametrize(
    ("samples", "sample_rate_hz", "config", "error"),
    [
        (np.ones(8), 0.0, DemodConfig(), "Sample rate"),
        (np.empty(0), 48_000.0, DemodConfig(), "at least one IQ sample"),
        (np.full(8, np.nan), 48_000.0, DemodConfig(), "NaN or infinite"),
        (np.ones(8), 48_000.0, replace(DemodConfig(), symbol_rate_hz=0), "Symbol rate"),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), samples_per_symbol=0),
            "At least two samples",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), samples_per_symbol=8.0),
            "must be an integer",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), samples_per_symbol=8, symbol_rate_hz=5_000),
            "integer samples-per-symbol",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), sample_rate_hz=44_100),
            "sample rates do not match",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), differential=True),
            "Differential QPSK",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), timing_offset_samples=8),
            "Timing offset",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), phase_reference_bits=(1, 0, 1)),
            "even-length binary sequence",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), scheme="BPSK"),
            "Unsupported scheme",
        ),
        (
            np.ones(8),
            48_000.0,
            replace(DemodConfig(), carrier_offset_hz=np.inf),
            "Carrier offset",
        ),
    ],
)
def test_demodulate_chain_rejects_unsupported_or_invalid_inputs(
    samples: np.ndarray, sample_rate_hz: float, config: DemodConfig, error: str
) -> None:
    with pytest.raises(ValueError, match=error):
        demodulate_chain(samples, sample_rate_hz, config)


def test_rrc_filter_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="at least two"):
        rrc_taps(1)
    with pytest.raises(ValueError, match="Rolloff"):
        rrc_taps(8, rolloff=1.0)
    with pytest.raises(ValueError, match="span"):
        rrc_taps(8, span=0)


def test_gray_qpsk_mapping_has_expected_quadrant_order() -> None:
    symbols = np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4)))
    assert gray_bits_from_symbols(symbols).tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
