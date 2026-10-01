from __future__ import annotations

import numpy as np

from sih26147.analysis import analyze_capture
from sih26147.ingest import Capture


def test_analysis_finds_single_tone_offset_and_absolute_peak() -> None:
    sample_rate = 256_000
    offset = 16_000
    time = np.arange(8192) / sample_rate
    samples = np.exp(2j * np.pi * offset * time).astype(np.complex64)
    capture = Capture(
        samples=samples,
        sample_rate_hz=sample_rate,
        center_frequency_hz=100_000_000,
        source_name="tone.iq",
        source_format="cf32",
        total_samples=samples.size,
        first_sample=0,
        truncated=False,
    )
    result = analyze_capture(capture)
    assert abs(result.peak_offset_hz - offset) <= sample_rate / 4096
    assert result.peak_frequency_hz == 100_000_000 + result.peak_offset_hz
    assert result.occupied_bandwidth_hz >= 0
    assert result.summary()["sample_count"] == samples.size
