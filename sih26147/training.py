from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from .analysis import extract_features
from .ingest import read_capture


REQUIRED_COLUMNS = {
    "capture_path",
    "label",
    "sample_rate_hz",
    "raw_format",
    "first_sample",
    "sample_count",
    "capture_group",
    "data_origin",
    "source_url",
    "license",
    "annotation_reference",
}


@dataclass(frozen=True)
class TrainingResult:
    model_path: str
    sample_count: int
    labels: list[str]
    accuracy: float
    classification: dict[str, Any]
    confusion_matrix: list[list[int]]
    provenance: list[dict[str, str]]


def train_real_capture_model(
    manifest_path: str | Path,
    *,
    model_path: str | Path,
    test_size: float = 0.25,
    random_state: int = 26147,
) -> TrainingResult:
    """Train from explicitly annotated measured captures with recording-level holdout."""
    manifest_path = Path(manifest_path).resolve()
    rows = _read_manifest(manifest_path)
    features: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    provenance: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(rows, start=2):
        origin = row["data_origin"].strip().lower()
        if origin not in {"measured", "over_the_air", "recorded"}:
            raise ValueError(f"Manifest row {row_number}: data_origin must identify a real measurement.")
        if not all(row[key].strip() for key in ("source_url", "license", "annotation_reference")):
            raise ValueError(f"Manifest row {row_number}: source URL, license, and annotation reference are required.")
        if not row["label"].strip() or not row["capture_group"].strip():
            raise ValueError(f"Manifest row {row_number}: label and capture_group cannot be empty.")
        source_url = urlparse(row["source_url"].strip())
        if source_url.scheme != "https" or not source_url.netloc:
            raise ValueError(f"Manifest row {row_number}: source_url must be an HTTPS URL.")
        capture_path = (manifest_path.parent / row["capture_path"]).resolve()
        if not capture_path.is_file():
            raise FileNotFoundError(f"Manifest row {row_number}: capture not found: {capture_path}")
        sample_count = int(row["sample_count"])
        if sample_count < 128 or sample_count > 1_000_000:
            raise ValueError(f"Manifest row {row_number}: sample_count must be between 128 and 1,000,000.")
        with capture_path.open("rb") as capture_file:
            capture = read_capture(
                capture_file,
                source_name=capture_path.name,
                raw_format=row["raw_format"].strip(),
                sample_rate_hz=float(row["sample_rate_hz"]),
                center_frequency_hz=_optional_float(row.get("center_frequency_hz", "")),
                first_sample=int(row["first_sample"]),
                sample_limit=sample_count,
            )
        if capture.samples.size != sample_count:
            raise ValueError(
                f"Manifest row {row_number}: expected {sample_count} samples, read {capture.samples.size}."
            )
        features.append(extract_features(capture.samples, capture.sample_rate_hz))
        label = row["label"].strip()
        labels.append(label)
        group = row["capture_group"].strip()
        groups.append(group)
        provenance[group] = {
            "source_url": row["source_url"].strip(),
            "license": row["license"].strip(),
            "annotation_reference": row["annotation_reference"].strip(),
        }

    x = np.vstack(features)
    y = np.asarray(labels)
    group_array = np.asarray(groups)
    classes = np.unique(y)
    group_count_by_class = {
        label: len(set(group_array[y == label]))
        for label in classes
    }
    if any(count < 2 for count in group_count_by_class.values()):
        raise ValueError(
            "Each class needs examples from at least two distinct recording groups "
            "to evaluate without same-recording train/test leakage."
        )
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5.")
    train_indices, test_indices = _grouped_split(
        x, y, group_array, test_size=test_size, random_state=random_state
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    classifier.fit(x[train_indices], y[train_indices])
    predictions = classifier.predict(x[test_indices])
    classification = classification_report(
        y[test_indices],
        predictions,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    model_path = Path(model_path).resolve()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "feature_version": 1,
            "labels": classes.tolist(),
            "training_manifest": str(manifest_path),
            "provenance": list(provenance.values()),
            "validation_accuracy": float(accuracy_score(y[test_indices], predictions)),
            "validation_classification": classification,
            "validation_confusion_matrix": confusion_matrix(
                y[test_indices], predictions, labels=classes
            ).tolist(),
        },
        model_path,
    )
    return TrainingResult(
        model_path=str(model_path),
        sample_count=len(labels),
        labels=classes.tolist(),
        accuracy=float(accuracy_score(y[test_indices], predictions)),
        classification=classification,
        confusion_matrix=confusion_matrix(y[test_indices], predictions, labels=classes).tolist(),
        provenance=list(provenance.values()),
    )


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {', '.join(sorted(missing))}.")
        rows = list(reader)
    if len(rows) < 4:
        raise ValueError("Manifest must contain at least four labeled signal windows.")
    return rows


def _grouped_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=100, test_size=test_size, random_state=random_state)
    for train, test in splitter.split(features, labels, groups):
        if set(labels[train]) == set(labels) and set(labels[test]) == set(labels):
            return train, test
    raise ValueError(
        "Could not make a recording-group split containing every label in both train and test. "
        "Add more independently recorded examples per modulation class."
    )


def _optional_float(value: str) -> float | None:
    return float(value) if value.strip() else None


def predict_with_trained_model(
    model_path: str | Path,
    samples: np.ndarray,
    sample_rate_hz: float,
) -> dict[str, object]:
    """Use a locally generated training artifact and return its saved holdout metadata."""
    bundle = joblib.load(Path(model_path))
    if not isinstance(bundle, dict) or bundle.get("feature_version") != 1:
        raise ValueError("Model artifact has an unsupported or invalid feature version.")
    classifier = bundle.get("model")
    if classifier is None or not hasattr(classifier, "predict_proba"):
        raise ValueError("Model artifact does not contain a probability-capable classifier.")
    feature = extract_features(samples, sample_rate_hz).reshape(1, -1)
    probabilities = classifier.predict_proba(feature)[0]
    index = int(np.argmax(probabilities))
    classes = classifier.classes_
    return {
        "label": str(classes[index]),
        "confidence": float(probabilities[index]),
        "probabilities": {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        },
        "validation_accuracy": bundle.get("validation_accuracy"),
        "validation_classification": bundle.get("validation_classification"),
        "validation_confusion_matrix": bundle.get("validation_confusion_matrix"),
        "labels": bundle.get("labels", []),
        "training_manifest": bundle.get("training_manifest"),
    }


def result_json(result: TrainingResult) -> str:
    return json.dumps(asdict(result), indent=2)
