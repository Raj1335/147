from __future__ import annotations

import pytest

from sih26147.validation import run_reference_validation


def test_reference_pipeline_recovers_known_payload_exactly() -> None:
    result = run_reference_validation(seed=26147, payload_bits=512, samples_per_symbol=8)
    assert result.passed
    assert result.coded_bits == (result.payload_bits + 6) * 2
    assert result.demodulated_ber == 0.0
    assert result.decoded_ber == 0.0


def test_reference_pipeline_rejects_too_short_payload() -> None:
    with pytest.raises(ValueError, match="at least 32 bits"):
        run_reference_validation(payload_bits=16)
