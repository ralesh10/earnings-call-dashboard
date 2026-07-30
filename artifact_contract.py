"""Validated, model-agnostic artifact loading for the dashboard."""

from __future__ import annotations

import csv
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_FILES = ("model.joblib", "feature_schema.json", "feature_table.csv", "run_manifest.json")
OPTIONAL_FILES = ("predictions.csv", "metrics.csv")
# A comparison/report directory can contain a run manifest without being a
# selectable model bundle. Require at least one model-data marker before
# treating a directory as a bundle candidate.
BUNDLE_MARKER_FILES = ("model.joblib", "feature_schema.json", "feature_table.csv")
FORBIDDEN_FEATURE_MARKERS = (
    "label", "target", "abnormal_return", "future_return", "stock_return",
    "market_return", "return_5d", "actual_outcome", "realized_return",
)
PREDICTION_STATUSES = {
    "out_of_sample_holdout",
    "walk_forward_validated",
    "retrospective_inference",
    "unavailable",
}


class ArtifactValidationError(ValueError):
    """Raised when a model bundle is incomplete or unsafe to use."""


@dataclass(frozen=True)
class ArtifactBundle:
    root: Path
    model: Any
    schema: dict[str, Any]
    manifest: dict[str, Any]
    feature_table: pd.DataFrame
    predictions: pd.DataFrame | None
    metrics: pd.DataFrame | None

    @property
    def model_version(self) -> str:
        return str(self.schema.get("model_version") or self.manifest.get("model_version") or "unknown")

    @property
    def display_name(self) -> str:
        return str(self.schema.get("display_name") or self.manifest.get("display_name") or self.model_version)

    @property
    def status(self) -> str:
        """Lifecycle state shown in the UI (stable, experimental, or provisional)."""
        return str(self.schema.get("status") or self.manifest.get("status") or "stable").strip().lower()

    @property
    def is_experimental(self) -> bool:
        return self.status in {"experimental", "provisional", "candidate"}

    @property
    def display_label(self) -> str:
        """Human-facing selector label without changing the stored model name."""
        if self.is_experimental and "experimental" not in self.display_name.lower():
            return f"{self.display_name} * Experimental"
        return self.display_name

    @property
    def feature_columns(self) -> list[str]:
        return list(self.schema["feature_columns"])

    @property
    def prediction_source(self) -> str:
        """Describe where per-call probabilities come from, when declared."""
        return str(self.manifest.get("prediction_source") or self.schema.get("prediction_source") or "")

    @property
    def validation_summary(self) -> dict[str, Any]:
        """Return optional human-facing validation metadata without requiring it."""
        value = self.schema.get("validation_summary") or self.manifest.get("validation_summary") or {}
        return value if isinstance(value, dict) else {}

    @property
    def optional_metadata(self) -> dict[str, Any]:
        """Expose optional artifact metadata for richer UI layers."""
        merged: dict[str, Any] = {}
        for source in (self.manifest.get("metadata"), self.schema.get("metadata")):
            if isinstance(source, dict):
                merged.update(source)
        return merged

    def stored_prediction(self, symbol: str, call_datetime: Any) -> pd.Series | None:
        """Return the stored prediction row for a call, when one exists.

        A missing row is meaningful: the dashboard must distinguish stored
        out-of-sample predictions from ad hoc model inference on historical
        feature rows.
        """
        if self.predictions is None:
            return None
        predictions = self.predictions.copy()
        predictions["symbol"] = predictions["symbol"].astype(str)
        predictions["call_datetime"] = pd.to_datetime(predictions["call_datetime"], errors="coerce")
        timestamp = pd.to_datetime(call_datetime, errors="coerce")
        matched = predictions[
            (predictions["symbol"] == str(symbol))
            & (predictions["call_datetime"] == timestamp)
        ]
        return matched.iloc[-1] if not matched.empty else None


def discover_artifact_dirs(root: str | Path | None = None) -> list[Path]:
    """Find model bundles for the model selector.

    ``EARNINGS_ARTIFACT_DIR`` remains a deployment override for one bundle.
    For the local/demo app, sibling bundles under ``artifacts/`` are exposed
    as selectable models. A directory is considered a bundle candidate if it
    contains at least one model-data marker, so an incomplete bundle can
    produce a visible validation error instead of silently disappearing.
    Report directories containing only comparison manifests are ignored.
    """
    forced = os.environ.get("EARNINGS_ARTIFACT_DIR") if root is None else None
    if forced:
        return [resolve_artifact_dir(forced)]

    configured_root = root or os.environ.get("EARNINGS_ARTIFACT_ROOT")
    if configured_root:
        candidate_root = Path(configured_root)
        if not candidate_root.is_absolute():
            candidate_root = Path(__file__).resolve().parent / candidate_root
    else:
        candidate_root = Path(__file__).resolve().parent / "artifacts"

    if not candidate_root.is_dir():
        raise ArtifactValidationError(f"Artifact root does not exist: {candidate_root}")
    if any((candidate_root / name).exists() for name in BUNDLE_MARKER_FILES):
        return [candidate_root]
    candidates = [
        path for path in sorted(candidate_root.iterdir())
        if path.is_dir() and any((path / name).exists() for name in BUNDLE_MARKER_FILES)
    ]
    if not candidates:
        raise ArtifactValidationError(f"No artifact bundles found under: {candidate_root}")
    return candidates


def resolve_artifact_dir(root: str | Path | None = None) -> Path:
    """Resolve only the new artifact contract; do not silently use legacy files."""
    configured = root or os.environ.get("EARNINGS_ARTIFACT_DIR")
    if configured:
        configured_path = Path(configured)
        candidates = [configured_path]
        if not configured_path.is_absolute():
            candidates.append(Path(__file__).resolve().parent / configured_path)
    else:
        candidates = [Path(__file__).resolve().parent / "artifacts" / "original_baseline"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise ArtifactValidationError(
        "No validated artifact directory found. Set EARNINGS_ARTIFACT_DIR or create artifacts/original_baseline/."
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactValidationError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{name} must contain a JSON object.")
    return value


def _validate_schema(schema: dict[str, Any], table: pd.DataFrame, manifest: dict[str, Any]) -> None:
    required_schema = {"artifact_version", "model_version", "model_family", "feature_columns", "target_column", "prediction_threshold"}
    missing = required_schema.difference(schema)
    if missing:
        raise ArtifactValidationError(f"feature_schema.json is missing: {sorted(missing)}")
    features = schema["feature_columns"]
    if not isinstance(features, list) or not features or any(not isinstance(value, str) for value in features):
        raise ArtifactValidationError("feature_columns must be a non-empty list of strings.")
    if len(features) != len(set(features)):
        raise ArtifactValidationError("feature_columns contains duplicate names.")
    forbidden = [
        column for column in features
        if any(marker in column.lower() for marker in FORBIDDEN_FEATURE_MARKERS)
    ]
    if forbidden:
        raise ArtifactValidationError(f"Target-like columns cannot be model features: {forbidden}")
    missing_features = [column for column in features if column not in table.columns]
    if missing_features:
        raise ArtifactValidationError(f"feature_table.csv is missing model features: {missing_features}")
    if schema["target_column"] in features:
        raise ArtifactValidationError("target_column cannot also appear in feature_columns.")
    threshold = schema["prediction_threshold"]
    if not isinstance(threshold, (int, float)) or not 0 < float(threshold) < 1:
        raise ArtifactValidationError("prediction_threshold must be between 0 and 1.")
    if not manifest.get("artifact_version"):
        raise ArtifactValidationError("run_manifest.json must include artifact_version.")
    if str(manifest["artifact_version"]) != str(schema["artifact_version"]):
        raise ArtifactValidationError("feature_schema.json and run_manifest.json have different artifact versions.")
    for column in features:
        numeric = pd.to_numeric(table[column], errors="coerce")
        if numeric.notna().sum() == 0:
            raise ArtifactValidationError(f"Feature column has no numeric values: {column}")


def _validate_predictions(predictions: pd.DataFrame | None) -> None:
    if predictions is None:
        return
    required = {"symbol", "call_datetime", "probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ArtifactValidationError(f"predictions.csv is missing: {sorted(missing)}")
    probabilities = pd.to_numeric(predictions["probability"], errors="coerce")
    if probabilities.isna().any() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ArtifactValidationError("predictions.csv contains probabilities outside [0, 1].")
    keys = predictions[["symbol", "call_datetime"]].copy()
    keys["symbol"] = keys["symbol"].astype(str)
    keys["call_datetime"] = pd.to_datetime(keys["call_datetime"], errors="coerce")
    if keys["call_datetime"].isna().any():
        raise ArtifactValidationError("predictions.csv contains invalid call_datetime values.")
    if keys.duplicated().any():
        raise ArtifactValidationError("predictions.csv contains duplicate symbol + call_datetime keys.")
    if "prediction_status" in predictions:
        statuses = predictions["prediction_status"].dropna().astype(str).str.strip().str.lower()
        invalid = sorted(set(statuses) - PREDICTION_STATUSES)
        if invalid:
            raise ArtifactValidationError(
                f"predictions.csv contains unsupported prediction_status values: {invalid}"
            )


def _repair_serialized_model_compatibility(model: Any) -> Any:
    """Repair a known scikit-learn 1.6 -> 1.9 private-state rename.

    The saved rich pipeline contains a fitted ``SimpleImputer`` created by
    scikit-learn 1.6.1. Newer releases renamed its internal ``_fill_dtype``
    field to ``_fit_dtype``. The learned statistics are unchanged; restoring
    the expected alias lets the artifact validate in a newer local runtime as
    well as in the pinned deployment environment.
    """
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if hasattr(value, "_fit_dtype") and not hasattr(value, "_fill_dtype"):
            try:
                value._fill_dtype = value._fit_dtype
            except Exception:
                pass
        named_steps = getattr(value, "named_steps", None)
        if isinstance(named_steps, dict):
            for child in named_steps.values():
                visit(child)
        steps = getattr(value, "steps", None)
        if isinstance(steps, list):
            for _, child in steps:
                visit(child)

    visit(model)
    return model


def load_artifact_bundle(root: str | Path | None = None) -> ArtifactBundle:
    """Load and validate a model bundle before the UI can use it."""
    artifact_dir = resolve_artifact_dir(root)
    missing = [name for name in REQUIRED_FILES if not (artifact_dir / name).exists()]
    if missing:
        raise ArtifactValidationError(f"Artifact bundle is missing required files: {missing}")
    try:
        import joblib
        model = joblib.load(artifact_dir / "model.joblib")
        model = _repair_serialized_model_compatibility(model)
    except Exception as exc:
        raise ArtifactValidationError(f"Could not load model.joblib: {exc}") from exc
    schema = _read_json(artifact_dir / "feature_schema.json", "feature_schema.json")
    manifest = _read_json(artifact_dir / "run_manifest.json", "run_manifest.json")
    feature_table_path = artifact_dir / "feature_table.csv"
    with feature_table_path.open(newline="", encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle), [])
    if len(header) != len(set(header)):
        duplicates = sorted({column for column in header if header.count(column) > 1})
        raise ArtifactValidationError(f"feature_table.csv contains duplicate columns: {duplicates}")
    table = pd.read_csv(feature_table_path)
    identity_columns = {"symbol", "call_datetime"}
    missing_identity = identity_columns.difference(table.columns)
    if missing_identity:
        raise ArtifactValidationError(
            f"feature_table.csv must contain prediction key columns: {sorted(missing_identity)}"
        )
    if pd.to_datetime(table["call_datetime"], errors="coerce").isna().any():
        raise ArtifactValidationError("feature_table.csv contains invalid call_datetime values.")
    if not callable(getattr(model, "predict_proba", None)):
        raise ArtifactValidationError("model.joblib must expose predict_proba for dashboard inference.")
    predictions = pd.read_csv(artifact_dir / "predictions.csv") if (artifact_dir / "predictions.csv").exists() else None
    metrics = pd.read_csv(artifact_dir / "metrics.csv") if (artifact_dir / "metrics.csv").exists() else None
    _validate_schema(schema, table, manifest)
    _validate_predictions(predictions)
    try:
        sample = table[schema["feature_columns"]].apply(pd.to_numeric, errors="coerce").head(1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without feature names")
            output = np.asarray(model.predict_proba(sample))
    except Exception as exc:
        raise ArtifactValidationError(f"model.joblib could not score the feature schema: {exc}") from exc
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < 2:
        raise ArtifactValidationError("model.joblib predict_proba must return a two-column probability matrix.")
    if not np.isfinite(output[0, 1]) or not 0 <= float(output[0, 1]) <= 1:
        raise ArtifactValidationError("model.joblib returned an invalid probability for the feature schema.")
    return ArtifactBundle(artifact_dir, model, schema, manifest, table, predictions, metrics)
