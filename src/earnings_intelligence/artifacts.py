"""Versioned artifact export for reproducible demos and deployment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def save_artifact_bundle(
    output_dir: str | Path,
    model: Any,
    feature_columns: list[str],
    metadata: Mapping[str, Any],
    feature_frame: pd.DataFrame | None = None,
    predictions: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> Path:
    """Save a model, schema, data, and metrics bundle for the dashboard."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("Install joblib to save model artifacts.") from exc

    joblib.dump(model, output / "model.joblib")
    schema = {
        "model_version": metadata.get("model_version", "dev"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns": feature_columns,
        "metadata": dict(metadata),
    }
    (output / "feature_schema.json").write_text(json.dumps(schema, indent=2, default=str))
    if feature_frame is not None:
        feature_frame.to_csv(output / "feature_table.csv", index=False)
    if predictions is not None:
        predictions.to_csv(output / "predictions.csv", index=False)
    if metrics is not None:
        metrics.to_csv(output / "metrics.csv", index=False)
    return output


def load_artifact_bundle(output_dir: str | Path) -> dict[str, Any]:
    """Load the dashboard bundle and validate its required files."""
    output = Path(output_dir)
    required = [output / "model.joblib", output / "feature_schema.json", output / "feature_table.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Artifact bundle is missing: {missing}")
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install joblib to load model artifacts.") from exc
    return {
        "model": joblib.load(output / "model.joblib"),
        "schema": json.loads((output / "feature_schema.json").read_text()),
        "feature_table": pd.read_csv(output / "feature_table.csv"),
        "predictions": pd.read_csv(output / "predictions.csv") if (output / "predictions.csv").exists() else None,
        "metrics": pd.read_csv(output / "metrics.csv") if (output / "metrics.csv").exists() else None,
    }
