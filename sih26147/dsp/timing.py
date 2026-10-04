from __future__ import annotations

import numpy as np


def estimate_timing_offset(samples: np.ndarray, sample_rate_hz: float, symbol_rate_hz: float, samples_per_symbol: int) -> float:
    """Choose the fractional timing offset that minimizes symbol constellation distance."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if x.size < 2:
        return 0.0
    if symbol_rate_hz <= 0 or sample_rate_hz <= 0:
        raise ValueError("Symbol rate and sample rate must be positive.")
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be at least two.")

    qpsk = np.exp(1j * (np.pi / 4.0 + np.pi / 2.0 * np.arange(4)))
    best_offset = 0.0
    best_score = np.inf
    for offset in np.linspace(0.0, samples_per_symbol, 33, endpoint=False):
        positions = np.round(np.arange(0, x.size, samples_per_symbol, dtype=float) + offset).astype(int)
        positions = positions[(positions >= 0) & (positions < x.size)]
        if positions.size < 8:
            continue
        symbols = x[positions]
        normalized = symbols / max(np.sqrt(np.mean(np.abs(symbols) ** 2)), 1e-12)
        distances = np.min(np.abs(normalized[:, None] - qpsk[None, :]), axis=1)
        score = float(np.mean(distances))
        if score < best_score:
            best_score = score
            best_offset = float(offset)
    return best_offset
