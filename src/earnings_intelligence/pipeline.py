"""Convenience functions for running the reusable experiment pipeline."""

from __future__ import annotations

from typing import Mapping, Sequence

import pandas as pd

from .features import add_expanding_history_features
from .modeling import evaluate_holdout, evaluate_walk_forward


def fit_fold_preprocessor(*args, **kwargs):
    """Public compatibility wrapper for the model preprocessing interface."""
    from .modeling import fit_fold_preprocessor as _fit

    return _fit(*args, **kwargs)


def fit_models(*args, **kwargs):
    """Fit one or more named models on a training frame."""
    from .modeling import fit_logistic_model, fit_xgb_model

    train_frame, feature_columns = args[:2]
    target_col = kwargs.get("target_col", "label")
    model_names = kwargs.get("model_names", ("logistic",))
    models = {}
    for name in model_names:
        if name == "logistic":
            models[name] = fit_logistic_model(train_frame, feature_columns, target_col=target_col)
        elif name == "xgboost":
            models[name] = fit_xgb_model(train_frame, feature_columns, target_col=target_col)
        else:
            raise ValueError(f"Unknown model {name!r}")
    return models


def evaluate_models(
    frame: pd.DataFrame,
    feature_blocks: Mapping[str, Sequence[str]],
    target_col: str = "label",
    model_names: Sequence[str] = ("logistic",),
):
    """Run walk-forward and exploratory holdout evaluations."""
    blocks = dict(feature_blocks)
    walk_summary, walk_predictions = evaluate_walk_forward(frame, blocks, target_col, model_names)
    holdout_summary, holdout_predictions = evaluate_holdout(frame, blocks, target_col, model_names=model_names)
    return {
        "walk_forward_summary": walk_summary,
        "walk_forward_predictions": walk_predictions,
        "holdout_summary": holdout_summary,
        "holdout_predictions": holdout_predictions,
    }


def add_causal_history(frame: pd.DataFrame, feature_columns: Sequence[str], **kwargs) -> pd.DataFrame:
    """Apply prior-only company/sector normalization."""
    return add_expanding_history_features(frame, feature_columns, **kwargs)
