from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


QPSK_GRAY = np.asarray(
    [
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0],
    ],
    dtype=np.uint8,
)


def qpsk_gray_indices(indices: np.ndarray) -> NDArray[np.uint8]:
    if np.asarray(indices).size == 0:
        return np.empty(0, dtype=np.uint8)
    idx = np.asarray(indices, dtype=int).reshape(-1)
    bits = np.array(QPSK_GRAY[idx].reshape(-1), dtype=np.uint8, copy=True)
    return bits


def gray_bits_from_symbols(symbols: np.ndarray) -> NDArray[np.uint8]:
    """Map each QPSK symbol to Gray-coded bits using the standard quadrant order."""
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    qpsk_values = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * np.arange(4)))
    nearest = np.argmin(np.abs(symbols[:, None] - qpsk_values[None, :]), axis=1)
    return qpsk_gray_indices(nearest)
