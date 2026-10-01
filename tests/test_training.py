from __future__ import annotations

import csv

import pytest

from sih26147.training import REQUIRED_COLUMNS, train_real_capture_model


def test_training_refuses_simulated_data_before_loading_capture(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    row = {column: "value" for column in REQUIRED_COLUMNS}
    row.update(
        {
            "capture_path": "missing.iq",
            "label": "BPSK",
            "sample_rate_hz": "1000000",
            "raw_format": "cf32",
            "first_sample": "0",
            "sample_count": "256",
            "capture_group": "capture-1",
            "data_origin": "synthetic",
        }
    )
    with manifest.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        for _ in range(4):
            writer.writerow(row)
    with pytest.raises(ValueError, match="real measurement"):
        train_real_capture_model(manifest, model_path=tmp_path / "model.joblib")
