"""Build no-transcript baselines for the dashboard's fixed 886-event comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "artifacts" / "experimental_rich" / "feature_table.csv"
OUTPUT_DIR = ROOT / "artifacts" / "model_comparison"
MARKET_FEATURES = [
    "momentum_5d",
    "momentum_20d",
    "volatility_20d",
    "market_momentum_20d",
    "beta_120d",
]
WALK_YEARS = (2019, 2020, 2021, 2022)


def _metrics(y_true: pd.Series, probability: np.ndarray, split: str, model: str) -> dict[str, object]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    predicted = (p >= 0.5).astype(int)
    return {
        "n": len(y),
        "positive_rate": float(y.mean()),
        "accuracy": accuracy_score(y, predicted),
        "balanced_accuracy": balanced_accuracy_score(y, predicted),
        "mcc": matthews_corrcoef(y, predicted),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "model": model,
        "split": split,
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "tn": int(((y == 0) & (predicted == 0)).sum()),
        "fp": int(((y == 0) & (predicted == 1)).sum()),
        "fn": int(((y == 1) & (predicted == 0)).sum()),
        "tp": int(((y == 1) & (predicted == 1)).sum()),
    }


def main() -> None:
    frame = pd.read_csv(TABLE_PATH)
    frame["event_year"] = pd.to_numeric(frame["event_year"], errors="coerce")
    frame = frame.dropna(subset=["event_year", "abnormal_return_5d"]).copy()
    frame["y"] = (pd.to_numeric(frame["abnormal_return_5d"], errors="coerce") > 0).astype(int)
    missing = [column for column in MARKET_FEATURES if column not in frame]
    if missing:
        raise ValueError(f"Market-only features are missing: {missing}")

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for year in WALK_YEARS:
        train = frame[frame["event_year"] < year].copy()
        test = frame[frame["event_year"] == year].copy()
        if train.empty or test.empty or train["y"].nunique() < 2:
            continue
        train_rate = float(train["y"].mean())
        market_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]).fit(train[MARKET_FEATURES], train["y"])
        for model_name, probability in (
            ("train_rate_baseline", np.full(len(test), train_rate)),
            ("market_only_logistic", market_model.predict_proba(test[MARKET_FEATURES])[:, 1]),
        ):
            metric_rows.append(_metrics(test["y"], probability, f"walk_forward_{year}", model_name))
            for index, (_, row) in enumerate(test.iterrows()):
                prediction_rows.append({
                    "symbol": row["symbol"],
                    "call_datetime": row["call_datetime"],
                    "event_year": int(row["event_year"]),
                    "y": int(row["y"]),
                    "probability": float(probability[index]),
                    "model": model_name,
                    "split": f"walk_forward_{year}",
                })

    holdout_train = frame[frame["event_year"] < 2023].copy()
    holdout = frame[frame["event_year"] >= 2023].copy()
    if not holdout.empty and holdout_train["y"].nunique() >= 2:
        train_rate = float(holdout_train["y"].mean())
        market_model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]).fit(holdout_train[MARKET_FEATURES], holdout_train["y"])
        for model_name, probability in (
            ("train_rate_baseline", np.full(len(holdout), train_rate)),
            ("market_only_logistic", market_model.predict_proba(holdout[MARKET_FEATURES])[:, 1]),
        ):
            metric_rows.append(_metrics(holdout["y"], probability, "final_holdout", model_name))
            for index, (_, row) in enumerate(holdout.iterrows()):
                prediction_rows.append({
                    "symbol": row["symbol"],
                    "call_datetime": row["call_datetime"],
                    "event_year": int(row["event_year"]),
                    "y": int(row["y"]),
                    "probability": float(probability[index]),
                    "model": model_name,
                    "split": "final_holdout",
                })

    prediction_frame = pd.DataFrame(prediction_rows)
    for model_name in ("train_rate_baseline", "market_only_logistic"):
        walk = prediction_frame[
            prediction_frame["model"].eq(model_name)
            & prediction_frame["split"].str.startswith("walk_forward")
        ]
        metric_rows.append(_metrics(walk["y"], walk["probability"], "walk_forward_aggregate", model_name))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    prediction_frame.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)
    (OUTPUT_DIR / "baseline_manifest.json").write_text(json.dumps({
        "sample_rows": int(len(frame)),
        "companies": int(frame["symbol"].nunique()),
        "market_features": MARKET_FEATURES,
        "walk_forward_years": list(WALK_YEARS),
        "holdout_cutoff_year": 2023,
        "threshold": 0.5,
        "note": "No-transcript baselines evaluated on the exact complete-language 886-event Industrials sample.",
    }, indent=2))
    print(pd.DataFrame(metric_rows).query("split in ['walk_forward_aggregate', 'final_holdout']").to_string(index=False))


if __name__ == "__main__":
    main()
