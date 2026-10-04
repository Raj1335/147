from __future__ import annotations

import numpy as np

from sih26147.estimators import _psk_frequency


def estimate_carrier_offset(samples: np.ndarray, sample_rate_hz: float, *, scheme: str = "QPSK") -> float:
    """Estimate a coarse carrier offset using the project's PSK spectral estimator."""
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if x.size < 32:
        return 0.0
    if sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    order = 2 if scheme == "BPSK" else 4
    return float(_psk_frequency(x, sample_rate_hz, order))
