import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
joblib = pytest.importorskip("joblib")

from artifact_contract import (  # noqa: E402
    ArtifactValidationError,
    _validate_predictions,
    _validate_schema,
    load_artifact_bundle,
)


class FakeProbabilityModel:
    def __init__(self, probability=0.64):
        self.probability = probability

    def predict_proba(self, frame):
        return np.asarray([[1.0 - self.probability, self.probability] for _ in range(len(frame))])


FEATURES = ["tone_a", "tone_b"]


def write_bundle(root: Path, *, features=FEATURES, model_version="test-v1"):
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(FakeProbabilityModel(), root / "model.joblib")
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA"],
            "call_datetime": ["2024-01-01T08:00:00", "2024-04-01T08:00:00"],
            "tone_a": [0.2, 0.4],
            "tone_b": [0.8, 0.6],
            "abnormal_return_5d": [0.01, -0.02],
        }
    ).to_csv(root / "feature_table.csv", index=False)
    (root / "feature_schema.json").write_text(json.dumps({
        "artifact_version": "1.0",
        "model_version": model_version,
        "model_family": "logistic_regression",
        "feature_columns": features,
        "target_column": "abnormal_return_5d",
        "prediction_threshold": 0.5,
    }))
    (root / "run_manifest.json").write_text(json.dumps({
        "artifact_version": "1.0",
        "model_version": model_version,
    }))


def test_valid_bundle_loads_and_can_be_swapped(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_bundle(first, model_version="first-v1")
    write_bundle(second, features=["tone_b", "tone_a"], model_version="second-v1")

    first_bundle = load_artifact_bundle(first)
    second_bundle = load_artifact_bundle(second)

    assert first_bundle.model_version == "first-v1"
    assert second_bundle.model_version == "second-v1"
    assert first_bundle.feature_columns != second_bundle.feature_columns
    assert first_bundle.model.predict_proba(first_bundle.feature_table.iloc[:1])[:, 1][0] == pytest.approx(0.64)


def test_missing_required_file_is_rejected(tmp_path):
    bundle = tmp_path / "missing"
    write_bundle(bundle)
    (bundle / "feature_schema.json").unlink()
    with pytest.raises(ArtifactValidationError, match="missing required files"):
        load_artifact_bundle(bundle)


def test_target_like_feature_is_rejected():
    table = pd.DataFrame({"future_return_5d": [0.1]})
    schema = {
        "artifact_version": "1.0",
        "model_version": "x",
        "model_family": "x",
        "feature_columns": ["future_return_5d"],
        "target_column": "target",
        "prediction_threshold": 0.5,
    }
    with pytest.raises(ArtifactValidationError, match="Target-like"):
        _validate_schema(schema, table, {"artifact_version": "1.0"})


def test_duplicate_feature_names_are_rejected():
    table = pd.DataFrame([[1.0, 2.0]], columns=["tone_a", "tone_a"])
    schema = {
        "artifact_version": "1.0",
        "model_version": "x",
        "model_family": "x",
        "feature_columns": ["tone_a", "tone_a"],
        "target_column": "target",
        "prediction_threshold": 0.5,
    }
    with pytest.raises(ArtifactValidationError, match="duplicate"):
        _validate_schema(schema, table, {"artifact_version": "1.0"})


def test_invalid_prediction_probability_is_rejected():
    predictions = pd.DataFrame({
        "symbol": ["AAA"],
        "call_datetime": ["2024-01-01T08:00:00"],
        "probability": [1.2],
    })
    with pytest.raises(ArtifactValidationError, match=r"outside \[0, 1\]"):
        _validate_predictions(predictions)
