from __future__ import annotations

import numpy as np

from sih26147.demod import demodulate_from_estimate
from sih26147.estimators import classify_modulation, estimate_symbol_rate


def _fixture(modulation: str, *, seed: int = 26147) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    count = 512
    if modulation == "16QAM":
        bits = rng.integers(0, 2, (count, 4), dtype=np.uint8)
        levels = np.asarray([-3, -1, 1, 3]) / np.sqrt(10)
        symbols = (
            levels[2 * bits[:, 0] + bits[:, 1]]
            + 1j * levels[2 * bits[:, 2] + bits[:, 3]]
        )
        bits = bits.reshape(-1)
    else:
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]
        bits = rng.integers(0, 2, count * int(np.log2(order)), dtype=np.uint8)
        labels = np.packbits(bits.reshape(-1, int(np.log2(order))), axis=1, bitorder="big")
        labels = labels[:, 0] >> (8 - int(np.log2(order)))
        symbols = np.exp(2j * np.pi * labels / order)
    return symbols, bits


def test_symbol_rate_and_modulation_estimate_for_psk_and_qam() -> None:
    sample_rate = 64_000
    samples_per_symbol = 8
    for modulation in ("BPSK", "QPSK", "8PSK", "16QAM"):
        symbols, _ = _fixture(modulation)
        samples = np.repeat(symbols, samples_per_symbol)
        rate, clock_score = estimate_symbol_rate(samples, sample_rate)
        estimate = classify_modulation(samples, sample_rate)
        assert rate == sample_rate / samples_per_symbol
        assert clock_score > 0.35
        assert estimate.modulation == modulation
        assert estimate.score > 0.25


def test_estimator_and_auto_demodulator_recover_qpsk_with_carrier_offset() -> None:
    sample_rate = 64_000
    samples_per_symbol = 8
    symbols, bits = _fixture("QPSK")
    baseband = np.repeat(symbols, samples_per_symbol)
    carrier_offset = 1_000.0
    time = np.arange(baseband.size) / sample_rate
    received = baseband * np.exp(1j * (2 * np.pi * carrier_offset * time + 0.37))
    estimate = classify_modulation(received, sample_rate)
    assert estimate.modulation == "QPSK"
    assert abs(estimate.frequency_offset_hz - carrier_offset) < 100
    decoded = demodulate_from_estimate(received, estimate, sample_rate)
    assert np.array_equal(decoded, bits)


def test_estimator_and_auto_demodulator_recover_bpsk_with_carrier_offset() -> None:
    sample_rate = 64_000
    samples_per_symbol = 8
    rng = np.random.default_rng(12)
    bits = rng.integers(0, 2, 512, dtype=np.uint8)
    baseband = np.repeat(1.0 - 2.0 * bits, samples_per_symbol).astype(np.complex128)
    time = np.arange(baseband.size) / sample_rate
    received = baseband * np.exp(1j * (2 * np.pi * 1_000 * time + 0.37))
    estimate = classify_modulation(received, sample_rate)
    assert estimate.modulation == "BPSK"
    decoded = demodulate_from_estimate(received, estimate, sample_rate)
    assert np.array_equal(decoded, bits)


def test_estimator_and_auto_demodulator_recover_16qam_with_carrier_offset() -> None:
    sample_rate = 64_000
    samples_per_symbol = 8
    symbols, bits = _fixture("16QAM")
    baseband = np.repeat(symbols, samples_per_symbol)
    carrier_offset = 1_000.0
    time = np.arange(baseband.size) / sample_rate
    received = baseband * np.exp(1j * (2 * np.pi * carrier_offset * time + 0.37))
    estimate = classify_modulation(received, sample_rate)
    assert estimate.modulation == "16QAM"
    decoded = demodulate_from_estimate(received, estimate, sample_rate)
    assert np.array_equal(decoded, bits)


def test_tone_capture_does_not_receive_a_false_modulation_label() -> None:
    sample_rate = 64_000
    time = np.arange(4096) / sample_rate
    tone = np.exp(2j * np.pi * 3_000 * time)
    estimate = classify_modulation(tone, sample_rate)
    assert estimate.modulation == "Unknown"


def test_2fsk_symbol_rate_and_auto_demodulation() -> None:
    sample_rate = 64_000
    samples_per_symbol = 8
    rng = np.random.default_rng(47)
    bits = rng.integers(0, 2, 512, dtype=np.uint8)
    deviation = 4_000.0
    frequency = np.repeat((2 * bits.astype(np.int16) - 1) * int(deviation), samples_per_symbol)
    phase = np.cumsum(2.0 * np.pi * frequency / sample_rate)
    samples = np.exp(1j * phase)
    estimate = classify_modulation(samples, sample_rate)
    assert estimate.modulation == "2FSK"
    assert estimate.symbol_rate_hz == sample_rate / samples_per_symbol
    assert abs(estimate.fsk_deviation_hz - deviation) < 100
    decoded = demodulate_from_estimate(samples, estimate, sample_rate)
    assert np.array_equal(decoded, bits[1 : 1 + decoded.size])
