from __future__ import annotations

import numpy as np
import pytest

from sih26147.decode import deinterleave, ldpc_decode, rs_decode, viterbi_decode
from sih26147.demod import bits_to_bytes, correlate_bitstreams, demodulate


def _conv_encode(bits: np.ndarray, constraint_length: int, generators: tuple[int, ...]) -> np.ndarray:
    state = 0
    mask = (1 << constraint_length) - 1
    output: list[int] = []
    for bit in bits:
        register = ((state << 1) | int(bit)) & mask
        output.extend((register & polynomial).bit_count() & 1 for polynomial in generators)
        state = register & ((1 << (constraint_length - 1)) - 1)
    return np.asarray(output, dtype=np.uint8)


def test_bpsk_hard_decision_recovers_clean_symbols() -> None:
    source_bits = np.asarray([0, 1, 1, 0, 1], dtype=np.uint8)
    symbols = np.repeat(1.0 - 2.0 * source_bits, 4).astype(np.complex64)
    assert np.array_equal(
        demodulate(symbols, scheme="BPSK", samples_per_symbol=4),
        source_bits,
    )


def test_2fsk_hard_decision_recovers_known_tones() -> None:
    source_bits = np.asarray([0, 1, 1, 0], dtype=np.uint8)
    samples_per_symbol = 8
    sample_rate = 64_000
    deviation = 4_000
    tone = np.repeat((2 * source_bits.astype(int) - 1) * deviation, samples_per_symbol)
    phase = np.cumsum(2.0 * np.pi * tone / sample_rate)
    samples = np.exp(1j * phase)
    decoded = demodulate(
        samples,
        scheme="2FSK",
        samples_per_symbol=samples_per_symbol,
        fsk_deviation_hz=deviation,
        sample_rate_hz=sample_rate,
    )
    assert np.array_equal(decoded, source_bits)


def test_viterbi_decodes_terminated_rate_half_code() -> None:
    payload = np.asarray([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    terminated = np.concatenate((payload, np.zeros(6, dtype=np.uint8)))
    encoded = _conv_encode(terminated, 7, (0o171, 0o133))
    assert np.array_equal(viterbi_decode(encoded), payload)


def test_block_and_pseudorandom_deinterleave_are_inverses() -> None:
    payload = np.arange(24, dtype=np.uint8) % 2
    interleaved = payload.reshape(4, 6).T.reshape(-1)
    assert np.array_equal(
        deinterleave(interleaved, method="block", rows=4, columns=6),
        payload,
    )
    order = np.random.default_rng(42).permutation(payload.size)
    coded = payload[order]
    assert np.array_equal(deinterleave(coded, method="pseudorandom", seed=42), payload)


def test_diagonal_deinterleave_is_inverse_of_diagonal_scan() -> None:
    rows, columns = 4, 6
    payload = np.arange(rows * columns, dtype=np.uint8) % 2
    order = [
        row * columns + column
        for diagonal in range(rows + columns - 1)
        for row in range(rows)
        for column in range(columns)
        if row + column == diagonal
    ]
    assert np.array_equal(
        deinterleave(payload[order], method="diagonal", rows=rows, columns=columns),
        payload,
    )


def test_convolutional_deinterleave_with_branch_delay_convention() -> None:
    payload = np.asarray([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
    branches = 3
    branch_streams = [payload[index::branches].tolist() for index in range(branches)]
    padded = [[0] * index + stream for index, stream in enumerate(branch_streams)]
    rounds = max(map(len, padded))
    interleaved = np.asarray(
        [padded[branch][round_index] if round_index < len(padded[branch]) else 0
         for round_index in range(rounds) for branch in range(branches)],
        dtype=np.uint8,
    )
    assert np.array_equal(
        deinterleave(
            interleaved,
            method="convolutional",
            branches=branches,
            original_length=payload.size,
        ),
        payload,
    )


def test_ldpc_stops_when_parity_syndrome_is_zero() -> None:
    matrix = np.asarray([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    decoded, converged, iterations = ldpc_decode(np.asarray([8.0, 8.0, 8.0]), matrix)
    assert converged
    assert iterations == 1
    assert np.all((matrix @ decoded) % 2 == 0)


def test_ldpc_degree_one_check_corrects_single_variable() -> None:
    decoded, converged, _ = ldpc_decode(np.asarray([-8.0]), np.asarray([[1]], dtype=np.uint8))
    assert converged
    assert decoded.tolist() == [0]


def test_reed_solomon_decodes_clean_codeword() -> None:
    from reedsolo import RSCodec

    payload = b"SIH26147"
    encoded = RSCodec(8).encode(payload)
    assert rs_decode(encoded, parity_symbols=8) == payload


def test_bitstream_correlation_reports_shift() -> None:
    result = correlate_bitstreams("1011001", "001011001", max_offset=4)
    assert result == {"bit_error_rate": 0.0, "offset_bits": 2, "compared_bits": 7}


def test_demodulator_requires_valid_timing() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        demodulate(np.ones(8), scheme="BPSK", samples_per_symbol=0)


def test_nonbinary_bit_values_are_not_silently_truncated() -> None:
    with pytest.raises(ValueError, match="only 0 and 1"):
        viterbi_decode(np.asarray([0.5, 1.0]))
    with pytest.raises(ValueError, match="only zeroes and ones"):
        bits_to_bytes(np.asarray([0, 2, 0, 1, 0, 1, 0, 1]))
