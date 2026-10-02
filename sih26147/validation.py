from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decode import viterbi_decode
from .demod import demodulate


@dataclass(frozen=True)
class ReferenceValidation:
    payload_bits: int
    coded_bits: int
    demodulated_ber: float
    decoded_ber: float
    passed: bool


def run_reference_validation(
    *,
    seed: int = 26147,
    payload_bits: int = 512,
    samples_per_symbol: int = 8,
) -> ReferenceValidation:
    """Exercise a known-answer BPSK → convolutional code → Viterbi pipeline."""
    if payload_bits < 32:
        raise ValueError("Reference payload must contain at least 32 bits.")
    if samples_per_symbol < 1:
        raise ValueError("Samples per symbol must be positive.")

    source = np.random.default_rng(seed).integers(0, 2, payload_bits, dtype=np.uint8)
    terminated = np.concatenate((source, np.zeros(6, dtype=np.uint8)))
    coded = _convolutional_encode(terminated, constraint_length=7, generators=(0o171, 0o133))
    symbols = np.repeat(1.0 - 2.0 * coded.astype(np.float64), samples_per_symbol)
    received = symbols.astype(np.complex128)
    demodulated = demodulate(
        received,
        scheme="BPSK",
        samples_per_symbol=samples_per_symbol,
    )
    decoded = viterbi_decode(
        demodulated,
        constraint_length=7,
        generators=(0o171, 0o133),
        terminated=True,
    )
    demodulated_ber = float(np.mean(demodulated != coded))
    decoded_ber = float(np.mean(decoded != source))
    return ReferenceValidation(
        payload_bits=payload_bits,
        coded_bits=int(coded.size),
        demodulated_ber=demodulated_ber,
        decoded_ber=decoded_ber,
        passed=decoded_ber == 0.0 and demodulated_ber == 0.0,
    )


def _convolutional_encode(
    bits: np.ndarray,
    *,
    constraint_length: int,
    generators: tuple[int, ...],
) -> np.ndarray:
    state = 0
    mask = (1 << constraint_length) - 1
    output: list[int] = []
    for bit in bits:
        register = ((state << 1) | int(bit)) & mask
        output.extend((register & polynomial).bit_count() & 1 for polynomial in generators)
        state = register & ((1 << (constraint_length - 1)) - 1)
    return np.asarray(output, dtype=np.uint8)
