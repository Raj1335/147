from __future__ import annotations

import numpy as np

from sih26147.dsp import DemodConfig, demodulate_chain, rrc_taps


def _qpsk_symbols(gray_bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(gray_bits, dtype=np.uint8).reshape(-1, 2)
    index = np.where((bits[:, 0] == 1) & (bits[:, 1] == 0), 3, np.where((bits[:, 0] == 1) & (bits[:, 1] == 1), 2, bits[:, 0] * 2 + bits[:, 1]))
    phases = np.pi / 4.0 + np.pi / 2.0 * index
    return np.exp(1j * phases)


def test_dsp_demod_chain_recovers_rrc_qpsk_with_offset() -> None:
    rng = np.random.default_rng(7)
    fs = 48_000.0
    symbol_rate_hz = 6_000.0
    samples_per_symbol = 8
    timing_offset_samples = int(round(0.37 * samples_per_symbol))
    bits = rng.integers(0, 2, size=512 * 2, endpoint=False).astype(np.uint8)
    symbols = _qpsk_symbols(bits.reshape(-1, 2))
    root = np.zeros(symbols.size * samples_per_symbol, dtype=np.complex128)
    root[0::samples_per_symbol] = symbols
    pulse = rrc_taps(samples_per_symbol, rolloff=0.35)
    shaped = np.convolve(root, pulse, mode="full")
    shifted = np.concatenate((np.zeros(timing_offset_samples, dtype=np.complex128), shaped))
    t = np.arange(shifted.size, dtype=np.float64) / fs
    carrier = np.exp(1j * 2.0 * np.pi * 150.0 * t)
    signal = shifted * carrier * np.exp(1j * 0.8)
    noise_power = np.mean(np.abs(signal) ** 2) / (10 ** (15.0 / 10.0))
    signal = signal + (rng.normal(0.0, np.sqrt(noise_power / 2.0), shifted.size) + 1j * rng.normal(0.0, np.sqrt(noise_power / 2.0), shifted.size))

    result = demodulate_chain(
        signal,
        fs,
        DemodConfig(
            scheme="QPSK",
            symbol_rate_hz=symbol_rate_hz,
            sample_rate_hz=fs,
            samples_per_symbol=samples_per_symbol,
            rolloff=0.35,
            gray=True,
            differential=False,
            phase_reference_bits=tuple(int(bit) for bit in bits[:64]),
        ),
    )
    decoded = np.asarray(result.bits, dtype=np.uint8).reshape(-1)
    assert decoded.size == bits.size
    ber = np.mean(decoded != bits)
    assert ber < 0.02
    assert abs(result.freq_trace[0] - 150.0) < 25.0
    assert abs(result.timing_trace[0] - timing_offset_samples) < 0.5
    assert result.llrs.size == 0
