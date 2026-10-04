from __future__ import annotations

import numpy as np


def rrc_taps(samples_per_symbol: int, rolloff: float = 0.35, span: int = 10) -> np.ndarray:
    """Return a root-raised cosine filter for matched filtering and timing recovery."""
    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be at least two.")
    if not 0.0 <= rolloff < 1.0:
        raise ValueError("Rolloff must lie in [0, 1).")
    if span < 1:
        raise ValueError("Filter span must be positive.")

    indices = np.arange(-span * samples_per_symbol, span * samples_per_symbol + 1, dtype=np.float64)
    time = indices / samples_per_symbol
    denominator = 1.0 - (2.0 * rolloff * time) ** 2
    numerator = (
        np.sin(np.pi * (1.0 - rolloff) * time)
        + 4.0 * rolloff * time * np.cos(np.pi * (1.0 + rolloff) * time)
    )
    response = np.zeros_like(time)
    nonzero = np.abs(denominator) > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        response[nonzero] = numerator[nonzero] / (np.pi * denominator[nonzero] * time[nonzero])
    response[~nonzero] = np.sinc(time[~nonzero])
    response[abs(time) < 1e-12] = 1.0
    return response / np.linalg.norm(response)
