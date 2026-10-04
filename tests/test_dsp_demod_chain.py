from __future__ import annotations

import numpy as np

from sih26147.demo import generate_qpsk_demo
from sih26147.dsp import demodulate_chain


def test_dsp_demod_chain_recovers_rrc_qpsk_with_offsets_and_reference_bits() -> None:
    demo = generate_qpsk_demo()
    result = demodulate_chain(
        demo.capture.samples,
        demo.capture.sample_rate_hz,
        demo.demod_config,
    )
    decoded = np.asarray(result.bits, dtype=np.uint8).reshape(-1)
    assert decoded.size == demo.reference_bits.size
    assert np.mean(decoded != demo.reference_bits) < 0.02
    assert abs(result.freq_trace[0] - 150.0) < 1.0
    assert abs(result.timing_trace[0] - 3.0) < 0.5
    assert result.llrs.size == 0
    assert result.warnings == ()


def test_synthetic_demo_is_reproducible_and_identifies_its_provenance() -> None:
    first = generate_qpsk_demo(seed=17)
    second = generate_qpsk_demo(seed=17)

    assert first.capture.source_format == "synthetic-cf32-iq"
    assert first.capture.source_name == "synthetic-qpsk-demo.iq"
    assert first.capture.samples.size < 2_000_000
    assert np.array_equal(first.reference_bits, second.reference_bits)
    assert np.array_equal(first.capture.samples, second.capture.samples)
