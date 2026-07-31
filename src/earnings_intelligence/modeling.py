"""Fold-safe model fitting, feature ablations, and evaluation metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .config import EvaluationConfig, ModelConfig


def _sklearn_imports():
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            log_loss,
            matthews_corrcoef,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("Install scikit-learn to run model evaluation.") from exc
    return {
        "SimpleImputer": SimpleImputer,
        "LogisticRegression": LogisticRegression,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
        "accuracy_score": accuracy_score,
        "average_precision_score": average_precision_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "brier_score_loss": brier_score_loss,
        "log_loss": log_loss,
        "matthews_corrcoef": matthews_corrcoef,
        "roc_auc_score": roc_auc_score,
    }


def select_non_redundant_columns(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold: float = 0.95,
) -> list[str]:
    """Drop later members of highly correlated pairs using training data only."""
    # A rich experiment may start from a frame that already contains baseline
    # features.  Deduplicate requested names before selecting/correlating;
    # duplicate pandas columns return Series/DataFrames instead of scalars and
    # can make boolean correlation checks ambiguous.
    # Remove duplicate labels from the frame itself before constructing the
    # correlation matrix.  Label-based pandas lookup can otherwise return a
    # DataFrame rather than a scalar when a name occurs more than once.
    clean_frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    columns = list(dict.fromkeys(column for column in feature_columns if column in clean_frame.columns))
    if len(columns) < 2:
        return columns
    correlations = clean_frame[columns].apply(pd.to_numeric, errors="coerce").corr().abs().fillna(0.0).to_numpy()
    selected: list[str] = []
    selected_positions: list[int] = []
    for position, column in enumerate(columns):
        if not selected_positions or all(correlations[position, prior_position] < threshold for prior_position in selected_positions):
            selected.append(column)
            selected_positions.append(position)
    return selected


def fit_fold_preprocessor(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    correlation_threshold: float = 0.95,
):
    """Fit imputation/scaling state using one training fold only."""
    sklearn = _sklearn_imports()
    selected = select_non_redundant_columns(train_frame, feature_columns, correlation_threshold)
    preprocessor = sklearn["Pipeline"](
        [
            ("imputer", sklearn["SimpleImputer"](strategy="median")),
            ("scaler", sklearn["StandardScaler"]()),
        ]
    )
    preprocessor.fit(train_frame[selected])
    return selected, preprocessor


def fit_logistic_model(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "label",
    model_config: ModelConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
):
    """Fit the standardized primary baseline and return model metadata."""
    sklearn = _sklearn_imports()
    model_config = model_config or ModelConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    selected, preprocessor = fit_fold_preprocessor(train_frame, feature_columns, evaluation_config.correlation_threshold)
    estimator = sklearn["LogisticRegression"](
        C=model_config.logistic_c,
        max_iter=model_config.logistic_max_iter,
        random_state=evaluation_config.random_state,
    )
    pipeline = sklearn["Pipeline"]([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(train_frame[selected], train_frame[target_col].astype(int))
    pipeline.selected_features_ = selected
    return pipeline


def fit_elastic_net_model(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "label",
    model_config: ModelConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
    C: float | None = None,
    l1_ratio: float | None = None,
):
    """Fit a fold-safe standardized Elastic Net logistic classifier."""
    sklearn = _sklearn_imports()
    model_config = model_config or ModelConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    selected, preprocessor = fit_fold_preprocessor(train_frame, feature_columns, evaluation_config.correlation_threshold)
    estimator = sklearn["LogisticRegression"](
        C=model_config.elastic_net_c if C is None else C,
        penalty="elasticnet",
        solver="saga",
        l1_ratio=model_config.elastic_net_l1_ratio if l1_ratio is None else l1_ratio,
        max_iter=model_config.elastic_net_max_iter,
        random_state=evaluation_config.random_state,
    )
    pipeline = sklearn["Pipeline"]([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(train_frame[selected], train_frame[target_col].astype(int))
    pipeline.selected_features_ = selected
    pipeline.elastic_net_C_ = model_config.elastic_net_c if C is None else C
    pipeline.elastic_net_l1_ratio_ = model_config.elastic_net_l1_ratio if l1_ratio is None else l1_ratio
    return pipeline


def fit_xgb_model(
    train_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_col: str = "label",
    model_config: ModelConfig | None = None,
    evaluation_config: EvaluationConfig | None = None,
):
    """Fit a shallow XGBoost comparison model when xgboost is installed."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError("Install xgboost to run the gradient-boosting comparison.") from exc
    sklearn = _sklearn_imports()
    model_config = model_config or ModelConfig()
    evaluation_config = evaluation_config or EvaluationConfig()
    selected, preprocessor = fit_fold_preprocessor(train_frame, feature_columns, evaluation_config.correlation_threshold)
    estimator = XGBClassifier(
        n_estimators=model_config.xgb_estimators,
        max_depth=model_config.xgb_depth,
        learning_rate=model_config.xgb_learning_rate,
        min_child_weight=model_config.xgb_min_child_weight,
        subsample=model_config.xgb_subsample,
        colsample_bytree=model_config.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=model_config.xgb_random_state,
        **model_config.extra_params,
    )
    pipeline = sklearn["Pipeline"]([("preprocessor", preprocessor), ("model", estimator)])
    pipeline.fit(train_frame[selected], train_frame[target_col].astype(int))
    pipeline.selected_features_ = selected
    return pipeline


def classification_metrics(y_true, probabilities) -> dict[str, float]:
    """Calculate classification metrics without assuming balanced classes."""
    sklearn = _sklearn_imports()
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
        "accuracy": float(sklearn["accuracy_score"](y_true, predictions)),
        "balanced_accuracy": float(sklearn["balanced_accuracy_score"](y_true, predictions)),
        "mcc": float(sklearn["matthews_corrcoef"](y_true, predictions)),
        "log_loss": float(sklearn["log_loss"](y_true, probabilities, labels=[0, 1])),
        "brier": float(sklearn["brier_score_loss"](y_true, probabilities)),
        "auc": np.nan,
        "average_precision": np.nan,
    }
    if len(np.unique(y_true)) > 1:
        result["auc"] = float(sklearn["roc_auc_score"](y_true, probabilities))
        result["average_precision"] = float(sklearn["average_precision_score"](y_true, probabilities))
    return result


def top_bottom_spread(predictions: pd.DataFrame, cost_per_leg: float = 0.001, fraction: float = 0.20) -> dict[str, float]:
    """Evaluate equal-weight top-minus-bottom predicted return spread."""
    if predictions.empty:
        return {"top_bottom_spread": np.nan, "after_cost_spread": np.nan}
    ordered = predictions.sort_values("probability")
    size = max(1, int(len(ordered) * fraction))
    bottom = ordered.head(size)
    top = ordered.tail(size)
    spread = float(top["return"].mean() - bottom["return"].mean())
    return {
        "top_bottom_spread": spread,
        "after_cost_spread": spread - 2 * cost_per_leg,
        "top_mean_return": float(top["return"].mean()),
        "bottom_mean_return": float(bottom["return"].mean()),
        "top_hit_rate": float((top["return"] > 0).mean()),
        "top_n": int(len(top)),
        "bottom_n": int(len(bottom)),
    }


def cluster_bootstrap(
    predictions: pd.DataFrame,
    metric: str = "auc",
    group_col: str = "symbol",
    repetitions: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Return a company-clustered 95% interval for one prediction metric."""
    if predictions.empty or group_col not in predictions:
        return np.nan, np.nan
    groups = predictions[group_col].astype(str).unique()
    if len(groups) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([predictions[predictions[group_col].astype(str) == group] for group in sampled], ignore_index=True)
        try:
            if metric in {"auc", "average_precision"} and sample["y"].nunique() < 2:
                continue
            values.append(classification_metrics(sample["y"], sample["probability"])[metric])
        except (ValueError, ZeroDivisionError):
            continue
    return tuple(np.quantile(values, [0.025, 0.975])) if values else (np.nan, np.nan)


def _fit_model(name, train, features, target_col, model_config, evaluation_config):
    if name == "logistic":
        return fit_logistic_model(train, features, target_col, model_config, evaluation_config)
    if name == "xgboost":
        return fit_xgb_model(train, features, target_col, model_config, evaluation_config)
    raise ValueError(f"Unknown model {name!r}; choose logistic or xgboost.")


def evaluate_walk_forward(
    frame: pd.DataFrame,
    feature_blocks: dict[str, Sequence[str]],
    target_col: str = "label",
    model_names: Sequence[str] = ("logistic",),
    evaluation_config: EvaluationConfig | None = None,
    model_config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate predefined feature blocks using expanding walk-forward folds."""
    evaluation_config = evaluation_config or EvaluationConfig()
    model_config = model_config or ModelConfig()
    if "event_year" not in frame or target_col not in frame:
        raise KeyError("frame must contain event_year and the target column")
    summary_rows = []
    prediction_frames = []
    for block_name, block_features in feature_blocks.items():
        features = [column for column in block_features if column in frame.columns]
        for model_name in model_names:
            for test_year in evaluation_config.walk_forward_years:
                train = frame[frame["event_year"] < test_year].dropna(subset=[target_col]).copy()
                test = frame[frame["event_year"] == test_year].dropna(subset=[target_col]).copy()
                if train.empty or test.empty or train[target_col].nunique() < 2 or not features:
                    continue
                model = _fit_model(model_name, train, features, target_col, model_config, evaluation_config)
                probabilities = model.predict_proba(test[getattr(model, "selected_features_", features)])[:, 1]
                prediction_columns = [column for column in ["symbol", "call_datetime", "quarter", "event_year", target_col] if column in test]
                predictions = test[prediction_columns].copy()
                predictions = predictions.rename(columns={target_col: "y"})
                predictions["return"] = test.get("abnormal_return_5d", np.nan).to_numpy()
                predictions["probability"] = probabilities
                predictions["feature_block"] = block_name
                predictions["model"] = model_name
                predictions["split"] = f"walk_forward_{test_year}"
                prediction_frames.append(predictions)
                metrics = classification_metrics(predictions["y"], probabilities)
                metrics.update({"feature_block": block_name, "model": model_name, "split": f"walk_forward_{test_year}"})
                metrics.update(top_bottom_spread(predictions, fraction=evaluation_config.probability_quintile))
                summary_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions.empty:
        for (block_name, model_name), group in predictions.groupby(["feature_block", "model"]):
            metrics = classification_metrics(group["y"], group["probability"])
            metrics.update({"feature_block": block_name, "model": model_name, "split": "walk_forward_aggregate"})
            metrics.update(top_bottom_spread(group, fraction=evaluation_config.probability_quintile))
            lower, upper = cluster_bootstrap(group, "auc", repetitions=evaluation_config.bootstrap_repetitions, random_state=evaluation_config.random_state)
            metrics.update({"auc_lower_95": lower, "auc_upper_95": upper})
            summary_rows.append(metrics)
    return pd.DataFrame(summary_rows), predictions


def evaluate_holdout(
    frame: pd.DataFrame,
    feature_blocks: dict[str, Sequence[str]],
    target_col: str = "label",
    cutoff_year: int = 2023,
    model_names: Sequence[str] = ("logistic",),
    evaluation_config: EvaluationConfig | None = None,
    model_config: ModelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a frozen pre-cutoff model on the exploratory final holdout."""
    evaluation_config = evaluation_config or EvaluationConfig(final_cutoff_year=cutoff_year)
    model_config = model_config or ModelConfig()
    train = frame[frame["event_year"] < cutoff_year].dropna(subset=[target_col]).copy()
    test = frame[frame["event_year"] >= cutoff_year].dropna(subset=[target_col]).copy()
    summary_rows = []
    prediction_frames = []
    for block_name, block_features in feature_blocks.items():
        features = [column for column in block_features if column in frame.columns]
        for model_name in model_names:
            if train.empty or test.empty or train[target_col].nunique() < 2 or not features:
                continue
            model = _fit_model(model_name, train, features, target_col, model_config, evaluation_config)
            selected = getattr(model, "selected_features_", features)
            probabilities = model.predict_proba(test[selected])[:, 1]
            prediction_columns = [column for column in ["symbol", "call_datetime", "quarter", "event_year", target_col] if column in test]
            predictions = test[prediction_columns].copy().rename(columns={target_col: "y"})
            predictions["return"] = test.get("abnormal_return_5d", np.nan).to_numpy()
            predictions["probability"] = probabilities
            predictions["feature_block"] = block_name
            predictions["model"] = model_name
            predictions["split"] = "final_holdout"
            prediction_frames.append(predictions)
            metrics = classification_metrics(predictions["y"], probabilities)
            metrics.update({"feature_block": block_name, "model": model_name, "split": "final_holdout"})
            metrics.update(top_bottom_spread(predictions, fraction=evaluation_config.probability_quintile))
            lower, upper = cluster_bootstrap(predictions, "auc", repetitions=evaluation_config.bootstrap_repetitions, random_state=evaluation_config.random_state)
            metrics.update({"auc_lower_95": lower, "auc_upper_95": upper})
            summary_rows.append(metrics)
    return pd.DataFrame(summary_rows), pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
