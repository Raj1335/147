from __future__ import annotations

import numpy as np


def deinterleave(
    bits: np.ndarray,
    *,
    method: str,
    rows: int = 1,
    columns: int = 1,
    branches: int = 1,
    seed: int = 0,
    original_length: int | None = None,
) -> np.ndarray:
    """Invert the documented interleavers used by this application."""
    data = _validate_bits(bits)
    if method == "block":
        if rows < 1 or columns < 1 or rows * columns != data.size:
            raise ValueError("Block deinterleaving requires rows × columns = bit count.")
        return data.reshape(columns, rows).T.reshape(-1)
    if method == "diagonal":
        if rows < 1 or columns < 1 or rows * columns != data.size:
            raise ValueError("Diagonal deinterleaving requires rows × columns = bit count.")
        matrix = np.empty((rows, columns), dtype=np.uint8)
        permutation = _diagonal_permutation(rows, columns)
        matrix.reshape(-1)[permutation] = data
        return matrix.reshape(-1)
    if method == "pseudorandom":
        order = np.random.default_rng(seed).permutation(data.size)
        result = np.empty_like(data)
        result[order] = data
        return result
    if method == "convolutional":
        if branches < 1:
            raise ValueError("Branch count must be positive.")
        if original_length is None:
            raise ValueError("Convolutional deinterleaving requires the original bit count to remove padding.")
        return _convolutional_deinterleave(data, branches, original_length)
    raise ValueError("Choose block, diagonal, pseudorandom, or convolutional.")


def viterbi_decode(
    bits: np.ndarray,
    *,
    constraint_length: int = 7,
    generators: tuple[int, ...] = (0o171, 0o133),
    terminated: bool = True,
) -> np.ndarray:
    """Hard-decision Viterbi decoder for a rate 1/n convolutional code."""
    received = _validate_bits(bits)
    if constraint_length < 2 or constraint_length > 12:
        raise ValueError("Constraint length must be between 2 and 12.")
    if not generators:
        raise ValueError("At least one generator polynomial is required.")
    n = len(generators)
    if received.size % n:
        raise ValueError("Coded bit count must be divisible by the number of generators.")
    state_count = 1 << (constraint_length - 1)
    full_mask = (1 << constraint_length) - 1
    if any(poly <= 0 or poly > full_mask for poly in generators):
        raise ValueError("Generator polynomial does not fit the selected constraint length.")
    received_symbols = received.reshape(-1, n)
    if len(received_symbols) * state_count > 10_000_000:
        raise ValueError("Viterbi block is too large for the selected constraint length; decode shorter frames.")
    infinity = np.iinfo(np.int32).max // 4
    metrics = np.full(state_count, infinity, dtype=np.int32)
    metrics[0] = 0
    predecessors = np.zeros((len(received_symbols), state_count), dtype=np.int32)
    decisions = np.zeros((len(received_symbols), state_count), dtype=np.uint8)

    for time, observed in enumerate(received_symbols):
        next_metrics = np.full(state_count, infinity, dtype=np.int32)
        for state in range(state_count):
            if metrics[state] == infinity:
                continue
            for bit in (0, 1):
                register = ((state << 1) | bit) & full_mask
                next_state = register & (state_count - 1)
                expected = np.fromiter(
                    ((register & poly).bit_count() & 1 for poly in generators),
                    dtype=np.uint8,
                    count=n,
                )
                metric = metrics[state] + int(np.count_nonzero(expected != observed))
                if metric < next_metrics[next_state]:
                    next_metrics[next_state] = metric
                    predecessors[time, next_state] = state
                    decisions[time, next_state] = bit
        metrics = next_metrics

    state = 0 if terminated else int(np.argmin(metrics))
    decoded = np.empty(len(received_symbols), dtype=np.uint8)
    for time in range(len(received_symbols) - 1, -1, -1):
        decoded[time] = decisions[time, state]
        state = predecessors[time, state]
    return decoded[:-constraint_length + 1] if terminated else decoded


def rs_decode(encoded: bytes, *, parity_symbols: int = 32) -> bytes:
    """Decode a Reed-Solomon codeword using reedsolo when installed."""
    try:
        from reedsolo import RSCodec
    except ImportError as exc:
        raise RuntimeError("Install the optional reedsolo package to use Reed-Solomon decoding.") from exc
    if not encoded:
        raise ValueError("Reed-Solomon input cannot be empty.")
    codec = RSCodec(parity_symbols)
    decoded, _, _ = codec.decode(encoded)
    return bytes(decoded)


def ldpc_decode(
    llrs: np.ndarray,
    parity_check: np.ndarray,
    *,
    max_iterations: int = 50,
) -> tuple[np.ndarray, bool, int]:
    """Sum-product LDPC decoder for an explicit user-supplied parity-check matrix."""
    raw_matrix = np.asarray(parity_check)
    values = np.asarray(llrs, dtype=np.float64).reshape(-1)
    if raw_matrix.ndim != 2 or raw_matrix.shape[1] != values.size or raw_matrix.size == 0:
        raise ValueError("Parity-check matrix columns must match the LLR count.")
    if not np.all(np.isin(raw_matrix, (0, 1))):
        raise ValueError("Parity-check matrix must contain only zeroes and ones.")
    h = raw_matrix.astype(np.uint8)
    if max_iterations < 1:
        raise ValueError("Maximum iterations must be positive.")
    checks, variables = h.shape
    if not np.any(h):
        raise ValueError("Parity-check matrix has no edges.")
    if not np.all(np.isfinite(values)):
        raise ValueError("LLR values must be finite.")
    q = np.zeros((checks, variables), dtype=np.float64)
    r = np.zeros_like(q)
    q[h == 1] = np.broadcast_to(values, (checks, variables))[h == 1]
    posterior = values.copy()
    estimate = np.zeros(variables, dtype=np.uint8)
    for iteration in range(1, max_iterations + 1):
        for check in range(checks):
            indices = np.flatnonzero(h[check])
            if indices.size == 0:
                continue
            incoming = np.clip(q[check, indices], -30.0, 30.0)
            tanh_values = np.tanh(incoming / 2.0)
            for position, variable in enumerate(indices):
                others = np.delete(tanh_values, position)
                product = float(np.prod(others)) if others.size else 1.0
                r[check, variable] = 2.0 * np.arctanh(np.clip(product, -0.999999, 0.999999))
        posterior = values + np.sum(r, axis=0)
        estimate = (posterior < 0).astype(np.uint8)
        if not np.any((h @ estimate) % 2):
            return estimate, True, iteration
        for check in range(checks):
            indices = np.flatnonzero(h[check])
            if indices.size:
                q[check, indices] = posterior[indices] - r[check, indices]
    return estimate, False, max_iterations


def _validate_bits(bits: np.ndarray) -> np.ndarray:
    raw = np.asarray(bits).reshape(-1)
    if raw.size == 0 or not np.all(np.isin(raw, (0, 1))):
        raise ValueError("Expected a non-empty bit array containing only 0 and 1.")
    return raw.astype(np.uint8)


def _diagonal_permutation(rows: int, columns: int) -> np.ndarray:
    return np.asarray(
        [row * columns + column for diagonal in range(rows + columns - 1)
         for row in range(rows) for column in range(columns)
         if row + column == diagonal],
        dtype=np.int64,
    )


def _convolutional_deinterleave(bits: np.ndarray, branches: int, original_length: int) -> np.ndarray:
    if bits.size % branches:
        raise ValueError("Convolutional interleaver stream must include complete branch rounds.")
    if original_length < 1 or original_length > bits.size:
        raise ValueError("Original bit count must be within the interleaved stream length.")
    branch_streams = [bits[index::branches].tolist() for index in range(branches)]
    delays = [index for index in range(branches)]
    for branch, delay in enumerate(delays):
        if len(branch_streams[branch]) <= delay:
            raise ValueError("Not enough bits to flush the convolutional deinterleaver.")
        branch_streams[branch] = branch_streams[branch][delay:]
    output: list[int] = []
    for offset in range(max(map(len, branch_streams))):
        for stream in branch_streams:
            if offset < len(stream):
                output.append(stream[offset])
    if original_length > len(output):
        raise ValueError("Original bit count is inconsistent with the selected branch delays.")
    return np.asarray(output[:original_length], dtype=np.uint8)
