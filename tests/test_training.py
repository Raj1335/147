from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import sih26147.training as training
from sih26147.training import (
    REQUIRED_COLUMNS,
    TrainingResult,
    _grouped_split,
    _optional_float,
    _read_manifest,
    predict_with_trained_model,
    result_json,
    train_real_capture_model,
)


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


def test_read_manifest_rejects_missing_columns_and_too_few_rows(tmp_path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("unexpected\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        _read_manifest(manifest)

    with manifest.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows({column: "fixture" for column in REQUIRED_COLUMNS} for _ in range(3))
    with pytest.raises(ValueError, match="at least four"):
        _read_manifest(manifest)


def test_grouped_split_keeps_recording_groups_separate_and_labels_present() -> None:
    features = np.arange(32, dtype=float).reshape(16, 2)
    labels = np.asarray(["BPSK"] * 8 + ["QPSK"] * 8)
    groups = np.asarray(
        [f"bpsk-{index // 2}" for index in range(8)]
        + [f"qpsk-{index // 2}" for index in range(8)]
    )

    train, test = _grouped_split(
        features, labels, groups, test_size=0.25, random_state=9
    )

    assert set(labels[train]) == set(labels)
    assert set(labels[test]) == set(labels)
    assert set(groups[train]).isdisjoint(groups[test])


def test_grouped_split_rejects_unseparable_labels() -> None:
    features = np.arange(8, dtype=float).reshape(4, 2)
    labels = np.asarray(["BPSK", "BPSK", "QPSK", "QPSK"])
    groups = np.asarray(["bpsk-only", "bpsk-only", "qpsk-only", "qpsk-only"])

    with pytest.raises(ValueError, match="Could not make a recording-group split"):
        _grouped_split(features, labels, groups, test_size=0.25, random_state=1)


def test_optional_float_accepts_blank_or_numeric_values() -> None:
    assert _optional_float("") is None
    assert _optional_float("  ") is None
    assert _optional_float(" 145.5 ") == pytest.approx(145.5)


@pytest.mark.parametrize(
    ("bundle", "message"),
    [
        ([], "unsupported or invalid feature version"),
        ({"feature_version": 2}, "unsupported or invalid feature version"),
        ({"feature_version": 1, "model": object()}, "probability-capable classifier"),
    ],
)
def test_model_prediction_rejects_invalid_bundles(
    monkeypatch: pytest.MonkeyPatch, bundle: object, message: str
) -> None:
    monkeypatch.setattr(training.joblib, "load", lambda _: bundle)

    with pytest.raises(ValueError, match=message):
        predict_with_trained_model("unused.joblib", np.ones(128), 48_000.0)


def test_training_result_serializes_to_json() -> None:
    result = TrainingResult(
        model_path="model.joblib",
        sample_count=4,
        labels=["BPSK"],
        accuracy=1.0,
        classification={},
        confusion_matrix=[[1]],
        provenance=[],
    )

    assert json.loads(result_json(result))["sample_count"] == 4


def _valid_manifest_row(**overrides: str) -> dict[str, str]:
    row = {
        "capture_path": "capture.iq",
        "label": "BPSK",
        "sample_rate_hz": "48000",
        "raw_format": "cf32",
        "first_sample": "0",
        "sample_count": "128",
        "capture_group": "capture-a",
        "data_origin": "measured",
        "source_url": "https://example.invalid/capture",
        "license": "test fixture only",
        "annotation_reference": "metadata validation fixture",
    }
    row.update(overrides)
    return row


def _write_manifest(path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source_url": ""}, "source URL, license, and annotation"),
        ({"label": ""}, "label and capture_group"),
        ({"capture_group": ""}, "label and capture_group"),
        ({"source_url": "http://example.invalid/capture"}, "HTTPS URL"),
    ],
)
def test_training_rejects_incomplete_provenance_before_reading_capture(
    tmp_path, overrides: dict[str, str], message: str
) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [_valid_manifest_row(**overrides) for _ in range(4)]
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=message):
        train_real_capture_model(manifest, model_path=tmp_path / "model.joblib")


@pytest.mark.parametrize(
    ("sample_count", "message"),
    [("127", "between 128 and 1,000,000"), ("1000001", "between 128 and 1,000,000")],
)
def test_training_rejects_out_of_range_sample_count(
    tmp_path, sample_count: str, message: str
) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [_valid_manifest_row(sample_count=sample_count) for _ in range(4)]
    _write_manifest(manifest, rows)
    (tmp_path / "capture.iq").write_bytes(b"")

    with pytest.raises(ValueError, match=message):
        train_real_capture_model(manifest, model_path=tmp_path / "model.joblib")


def test_training_group_count_and_sample_window_guards(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.csv"
    (tmp_path / "capture.iq").write_bytes(b"")
    # Metadata-only test doubles exercise validation; they are not a training dataset.
    monkeypatch.setattr(
        training,
        "read_capture",
        lambda *_args, **kwargs: SimpleNamespace(
            samples=np.zeros(kwargs["sample_limit"]),
            sample_rate_hz=kwargs["sample_rate_hz"],
        ),
    )
    monkeypatch.setattr(training, "extract_features", lambda samples, _rate: samples[:2])
    one_group_per_label = [
        _valid_manifest_row(label="BPSK", capture_group="bpsk")
        for _ in range(2)
    ] + [
        _valid_manifest_row(label="QPSK", capture_group="qpsk")
        for _ in range(2)
    ]
    _write_manifest(manifest, one_group_per_label)
    with pytest.raises(ValueError, match="at least two distinct recording groups"):
        train_real_capture_model(manifest, model_path=tmp_path / "model.joblib")

    enough_groups = [
        _valid_manifest_row(label=label, capture_group=f"{label}-{index}")
        for label in ("BPSK", "QPSK")
        for index in range(2)
    ]
    _write_manifest(manifest, enough_groups)
    monkeypatch.setattr(
        training,
        "read_capture",
        lambda *_args, **kwargs: SimpleNamespace(
            samples=np.zeros(kwargs["sample_limit"] - 1),
            sample_rate_hz=kwargs["sample_rate_hz"],
        ),
    )
    with pytest.raises(ValueError, match="expected 128 samples, read 127"):
        train_real_capture_model(manifest, model_path=tmp_path / "model.joblib")
