"""Controlled E9 tuning and multi-universe transfer experiments.

This module deliberately keeps experiment selection separate from the final
artifact pipeline.  It evaluates every declared model configuration on the
same temporal folds, selects configurations using primary-target walk-forward
performance only, and then applies the selected configurations unchanged to
other universes.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .config import EvaluationConfig, ModelConfig
from .artifacts import save_artifact_bundle
from .final_pipeline import build_final_feature_blocks
from .modeling import (
    classification_metrics,
    cluster_bootstrap,
    fit_elastic_net_model,
    fit_logistic_model,
    fit_xgb_model,
    top_bottom_spread,
)


E9_PRIMARY_BLOCKS = ("sentence_sentiment", "historical_surprise", "all_features")
E9_CONTROL_BLOCKS = ("market_only", "baseline_sentiment")
E9_ALL_BLOCKS = E9_CONTROL_BLOCKS + E9_PRIMARY_BLOCKS
E9_CACHE_VERSION = "language-features-v3-experiment"


def elastic_net_grid() -> list[dict]:
    """Return the predeclared 12-setting Elastic Net grid."""
    return [
        {
            "name": f"elastic_net_C{C:g}_l1{ratio:g}",
            "kind": "elastic_net",
            "C": C,
            "l1_ratio": ratio,
            "config": ModelConfig(),
        }
        for C in (0.01, 0.1, 1.0, 10.0)
        for ratio in (0.1, 0.5, 0.9)
    ]


def xgboost_grid() -> list[dict]:
    """Return the predeclared conservative six-setting XGBoost grid."""
    return [
        {
            "name": f"xgboost_depth{depth}_trees{trees}",
            "kind": "xgboost",
            "config": ModelConfig(xgb_depth=depth, xgb_estimators=trees),
        }
        for depth in (1, 2, 3)
        for trees in (100, 200)
    ]


def e9_model_grid(include_xgboost: bool = True) -> list[dict]:
    """Return fixed L2, Elastic Net, and optionally XGBoost candidates."""
    specs = [{"name": "logistic", "kind": "logistic", "config": ModelConfig()}]
    specs.extend(elastic_net_grid())
    if include_xgboost:
        specs.extend(xgboost_grid())
    return specs


def _fit_spec(spec: Mapping, train: pd.DataFrame, features: Sequence[str], target_col: str, evaluation: EvaluationConfig):
    kind = spec["kind"]
    config = spec.get("config", ModelConfig())
    if kind == "logistic":
        return fit_logistic_model(train, features, target_col, config, evaluation)
    if kind == "elastic_net":
        return fit_elastic_net_model(
            train,
            features,
            target_col,
            config,
            evaluation,
            C=float(spec["C"]),
            l1_ratio=float(spec["l1_ratio"]),
        )
    if kind == "xgboost":
        return fit_xgb_model(train, features, target_col, config, evaluation)
    raise ValueError(f"Unknown model specification: {kind!r}")


def _calibration_mae(y: pd.Series, probability: np.ndarray, bins: int = 5) -> float:
    values = pd.DataFrame({"y": np.asarray(y), "p": probability})
    if len(values) < 2 or values["p"].nunique() < 2:
        return np.nan
    values["bin"] = pd.qcut(values["p"], q=min(bins, len(values)), duplicates="drop")
    grouped = values.groupby("bin", observed=False)
    return float((grouped["y"].mean() - grouped["p"].mean()).abs().mean())


def _evaluate_spec(
    frame: pd.DataFrame,
    block_name: str,
    features: Sequence[str],
    spec: Mapping,
    target_col: str,
    evaluation: EvaluationConfig,
    holdout: bool = False,
    bootstrap_repetitions: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one block/spec and return fold/aggregate metrics and predictions."""
    usable_features = list(dict.fromkeys(column for column in features if column in frame.columns))
    if not usable_features:
        return pd.DataFrame(), pd.DataFrame()
    summary_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    if holdout:
        split_years = ("final_holdout",)
    else:
        split_years = evaluation.walk_forward_years

    for year in split_years:
        if holdout:
            train = frame[frame["event_year"] < evaluation.final_cutoff_year].dropna(subset=[target_col]).copy()
            test = frame[frame["event_year"] >= evaluation.final_cutoff_year].dropna(subset=[target_col]).copy()
            split = "final_holdout"
        else:
            train = frame[frame["event_year"] < year].dropna(subset=[target_col]).copy()
            test = frame[frame["event_year"] == year].dropna(subset=[target_col]).copy()
            split = f"walk_forward_{year}"
        if train.empty or test.empty or train[target_col].nunique() < 2:
            continue
        model = _fit_spec(spec, train, usable_features, target_col, evaluation)
        selected = getattr(model, "selected_features_", usable_features)
        probability = model.predict_proba(test[selected])[:, 1]
        columns = [column for column in ["symbol", "call_datetime", "quarter", "event_year", target_col] if column in test]
        predictions = test[columns].copy().rename(columns={target_col: "y"})
        predictions["return"] = test.get("abnormal_return_5d", np.nan).to_numpy()
        predictions["probability"] = probability
        predictions["feature_block"] = block_name
        predictions["model"] = spec["name"]
        predictions["split"] = split
        prediction_frames.append(predictions)
        metrics = classification_metrics(predictions["y"], probability)
        metrics.update({"feature_block": block_name, "model": spec["name"], "split": split})
        metrics.update(top_bottom_spread(predictions, fraction=evaluation.probability_quintile))
        metrics["calibration_mae"] = _calibration_mae(predictions["y"], probability)
        summary_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if predictions.empty:
        return pd.DataFrame(summary_rows), predictions

    aggregate = classification_metrics(predictions["y"], predictions["probability"])
    aggregate.update({
        "feature_block": block_name,
        "model": spec["name"],
        "split": "final_holdout" if holdout else "walk_forward_aggregate",
    })
    aggregate.update(top_bottom_spread(predictions, fraction=evaluation.probability_quintile))
    aggregate["calibration_mae"] = _calibration_mae(predictions["y"], predictions["probability"])
    fold_metrics = pd.DataFrame(summary_rows)
    if not fold_metrics.empty:
        aggregate["mean_fold_auc"] = float(fold_metrics["auc"].mean())
        aggregate["mean_fold_log_loss"] = float(fold_metrics["log_loss"].mean())
        aggregate["fold_count"] = int(len(fold_metrics))
    else:
        aggregate["mean_fold_auc"] = np.nan
        aggregate["mean_fold_log_loss"] = np.nan
        aggregate["fold_count"] = 0
    if bootstrap_repetitions and not holdout:
        lower, upper = cluster_bootstrap(
            predictions,
            "auc",
            repetitions=bootstrap_repetitions,
            random_state=evaluation.random_state,
        )
        aggregate["auc_lower_95"] = lower
        aggregate["auc_upper_95"] = upper
    return pd.concat([fold_metrics, pd.DataFrame([aggregate])], ignore_index=True), predictions


def _select_best(rows: pd.DataFrame) -> pd.Series:
    aggregate = rows[rows["split"] == "walk_forward_aggregate"].copy()
    if aggregate.empty:
        raise ValueError("No walk-forward aggregate rows were produced for selection.")
    return aggregate.sort_values(
        ["mean_fold_auc", "mean_fold_log_loss"], ascending=[False, True]
    ).iloc[0]


def validate_language_cache(cache: pd.DataFrame) -> dict[str, object]:
    """Validate that a language cache contains no trusted targets or derived fields."""
    required = {"symbol", "call_datetime", "pres_sent_mean", "qa_sent_mean"}
    missing = required.difference(cache.columns)
    if missing:
        raise ValueError(f"Language cache is missing required columns: {sorted(missing)}")
    keys = cache[["symbol", "call_datetime"]].astype({"symbol": str}).copy()
    keys["call_datetime"] = pd.to_datetime(keys["call_datetime"], errors="coerce")
    if keys["call_datetime"].isna().any():
        raise ValueError("Language cache contains invalid call_datetime values.")
    if keys.duplicated().any():
        raise ValueError("Language cache contains duplicate symbol/call_datetime keys.")
    forbidden = [
        column for column in cache.columns
        if column in {"label", "beta_label", "market_subtracted", "beta_market_subtracted"}
        or column.startswith(("abnormal_return_", "beta_abnormal_return_", "future_return_", "target_"))
        or column in {"momentum_5d", "momentum_20d", "volatility_20d", "market_momentum_20d", "beta_120d"}
        or column in {"historical_score_source", "target_baseline_date", "target_end_date_5d", "target_price_basis"}
        or column.endswith("_z")
        or column.endswith("_history_count")
    ]
    if forbidden:
        raise ValueError(f"Language cache contains forbidden target/derived columns: {sorted(forbidden)}")
    return {
        "rows": int(len(cache)),
        "companies": int(cache["symbol"].nunique()),
        "complete_language_rows": int((cache["pres_sent_mean"].notna() & cache["qa_sent_mean"].notna()).sum()),
        "cache_version": cache.attrs.get("cache_version", "legacy-unversioned"),
    }


def run_e9_experiment(
    frame: pd.DataFrame,
    output_dir: str | Path = "e9_artifacts",
    include_xgboost: bool = True,
    complete_case: bool = True,
    evaluation: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Tune every declared model configuration on each E9 feature block."""
    evaluation = evaluation or EvaluationConfig(random_state=42, bootstrap_repetitions=250)
    working = frame.copy().loc[:, ~frame.columns.duplicated()]
    required = {"event_year", "label", "symbol", "abnormal_return_5d"}
    missing = required.difference(working.columns)
    if missing:
        raise KeyError(f"E9 frame is missing required columns: {sorted(missing)}")
    if complete_case:
        required_language = {"pres_sent_mean", "qa_sent_mean"}
        if not required_language.issubset(working.columns):
            raise KeyError("Complete-case E9 requires pres_sent_mean and qa_sent_mean.")
        working = working[working["pres_sent_mean"].notna() & working["qa_sent_mean"].notna()].copy()
    blocks = build_final_feature_blocks(working)
    selected_blocks = {name: blocks[name] for name in E9_ALL_BLOCKS if name in blocks}
    specs = e9_model_grid(include_xgboost=include_xgboost)
    tuning_rows: list[pd.DataFrame] = []
    tuning_predictions: list[pd.DataFrame] = []
    for block_name, features in selected_blocks.items():
        for spec in specs:
            metrics, predictions = _evaluate_spec(
                working,
                block_name,
                features,
                spec,
                "label",
                evaluation,
                bootstrap_repetitions=0,
            )
            if not metrics.empty:
                tuning_rows.append(metrics.assign(tuning_stage="e9_walk_forward"))
                tuning_predictions.append(predictions)
    tuning = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()
    if tuning.empty:
        raise RuntimeError("E9 produced no valid tuning results.")

    selected_specs: dict[str, dict] = {}
    selected_rows = []
    for block_name in selected_blocks:
        best = _select_best(tuning[tuning["feature_block"] == block_name])
        selected_rows.append(best)
        spec = next(spec for spec in specs if spec["name"] == best["model"])
        selected_specs[block_name] = spec

    primary_selected = pd.DataFrame(selected_rows)
    primary_candidates = primary_selected[primary_selected["feature_block"].isin(E9_PRIMARY_BLOCKS)]
    overall = _select_best(primary_candidates)
    winner_block = str(overall["feature_block"])
    winner_model_name = str(overall["model"])
    winner_spec = selected_specs[winner_block]
    winner_mean_walk_auc = float(overall["mean_fold_auc"])
    winner_pooled_walk_auc = float(overall["auc"])

    selected_walk_rows = []
    selected_walk_predictions = []
    selected_holdout_rows = []
    selected_holdout_predictions = []
    for block_name, spec in selected_specs.items():
        features = selected_blocks[block_name]
        walk_metrics, walk_predictions = _evaluate_spec(
            working, block_name, features, spec, "label", evaluation,
            bootstrap_repetitions=0,
        )
        holdout_metrics, holdout_predictions = _evaluate_spec(
            working, block_name, features, spec, "label", evaluation,
            holdout=True, bootstrap_repetitions=0,
        )
        if not walk_metrics.empty:
            selected_walk_rows.append(walk_metrics.assign(evaluation_stage="selected_config"))
            selected_walk_predictions.append(walk_predictions)
        if not holdout_metrics.empty:
            selected_holdout_rows.append(holdout_metrics.assign(evaluation_stage="selected_config"))
            selected_holdout_predictions.append(holdout_predictions)

    selected_walk = pd.concat(selected_walk_rows, ignore_index=True) if selected_walk_rows else pd.DataFrame()
    selected_holdout = pd.concat(selected_holdout_rows, ignore_index=True) if selected_holdout_rows else pd.DataFrame()
    selected_predictions = pd.concat(selected_walk_predictions, ignore_index=True) if selected_walk_predictions else pd.DataFrame()
    holdout_predictions = pd.concat(selected_holdout_predictions, ignore_index=True) if selected_holdout_predictions else pd.DataFrame()

    winner_predictions = selected_predictions[
        (selected_predictions["feature_block"] == winner_block)
        & (selected_predictions["model"] == winner_model_name)
    ]
    winner_bootstrap = cluster_bootstrap(
        winner_predictions,
        "auc",
        repetitions=1000,
        random_state=evaluation.random_state,
    )
    winner_mask = (
        (selected_walk["feature_block"] == winner_block)
        & (selected_walk["model"] == winner_model_name)
        & (selected_walk["split"] == "walk_forward_aggregate")
    )
    selected_walk.loc[winner_mask, "auc_lower_95"] = winner_bootstrap[0]
    selected_walk.loc[winner_mask, "auc_upper_95"] = winner_bootstrap[1]
    winner_holdout_aggregate = selected_holdout[
        (selected_holdout["feature_block"] == winner_block)
        & (selected_holdout["model"] == winner_model_name)
        & (selected_holdout["split"] == "final_holdout")
    ] if not selected_holdout.empty else pd.DataFrame()
    winner_holdout_auc = float(winner_holdout_aggregate["auc"].iloc[0]) if not winner_holdout_aggregate.empty else np.nan
    winner_fold_rows = selected_walk[
        (selected_walk["feature_block"] == winner_block)
        & (selected_walk["model"] == winner_model_name)
        & selected_walk["split"].astype(str).str.startswith("walk_forward_")
    ]

    development = working[working["event_year"] < evaluation.final_cutoff_year].dropna(subset=["label"]).copy()
    winner_model = _fit_spec(winner_spec, development, selected_blocks[winner_block], "label", evaluation)
    selected_features = getattr(winner_model, "selected_features_", selected_blocks[winner_block])

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    blocks_path = output / "feature_blocks.json"
    blocks_path.write_text(json.dumps(selected_blocks, indent=2))
    tuning.to_csv(output / "tuning_results.csv", index=False)
    selected_walk.to_csv(output / "metrics.csv", index=False)
    selected_holdout.to_csv(output / "holdout_metrics.csv", index=False)
    pd.concat(tuning_predictions, ignore_index=True).to_csv(output / "tuning_predictions.csv", index=False)
    selected_predictions.to_csv(output / "predictions.csv", index=False)
    holdout_predictions.to_csv(output / "holdout_predictions.csv", index=False)
    manifest = {
        "experiment": "E9 per-block hyperparameter comparison",
        "sample_rows": int(len(working)),
        "companies": int(working["symbol"].nunique()),
        "complete_case": bool(complete_case),
        "blocks": list(selected_blocks),
        "primary_blocks": list(E9_PRIMARY_BLOCKS),
        "winner_block": winner_block,
        "winner_model": winner_model_name,
        "winner_spec": {key: value for key, value in winner_spec.items() if key != "config"},
        "selected_configs": {
            block: {key: value for key, value in spec.items() if key != "config"}
            for block, spec in selected_specs.items()
        },
        "selection_rule": "highest mean 2019-2022 walk-forward AUC, then lowest mean walk-forward log loss; holdout excluded",
        "walk_forward_years": list(evaluation.walk_forward_years),
        "holdout_cutoff_year": evaluation.final_cutoff_year,
        "random_state": evaluation.random_state,
        "winner_auc_lower_95": winner_bootstrap[0],
        "winner_auc_upper_95": winner_bootstrap[1],
        "reference_mean_walk_auc": 0.611,
        "reference_pooled_walk_auc": 0.622,
        "reference_holdout_auc": 0.596,
        "material_improvement_threshold": 0.02,
        "winner_mean_walk_auc": winner_mean_walk_auc,
        "winner_pooled_walk_auc": winner_pooled_walk_auc,
        "winner_holdout_auc": winner_holdout_auc,
        "mean_walk_auc_delta_vs_reference": winner_mean_walk_auc - 0.611,
        "pooled_walk_auc_delta_vs_reference": winner_pooled_walk_auc - 0.622,
        "holdout_auc_delta_vs_reference": winner_holdout_auc - 0.596 if np.isfinite(winner_holdout_auc) else np.nan,
        "folds_at_or_above_reference_mean": int((winner_fold_rows["auc"] >= 0.611).sum()),
        "cache_version": E9_CACHE_VERSION,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (output / "feature_schema.json").write_text(json.dumps({
        "feature_block": winner_block,
        "feature_columns": list(selected_features),
        "blocks": selected_blocks,
        "model": winner_model_name,
    }, indent=2, default=str))
    pd.DataFrame([
        {"check": "rows", "value": len(working)},
        {"check": "companies", "value": working["symbol"].nunique()},
        {"check": "positive_rate", "value": working["label"].mean()},
        {"check": "complete_case", "value": complete_case},
        {"check": "winner_block", "value": winner_block},
        {"check": "winner_model", "value": winner_model_name},
        {"check": "winner_mean_walk_auc", "value": winner_mean_walk_auc},
        {"check": "winner_pooled_walk_auc", "value": winner_pooled_walk_auc},
        {"check": "winner_holdout_auc", "value": winner_holdout_auc},
    ]).to_csv(output / "target_audit.csv", index=False)
    winner_holdout = holdout_predictions[
        (holdout_predictions["feature_block"] == winner_block)
        & (holdout_predictions["model"] == winner_model_name)
    ].copy()
    save_artifact_bundle(
        output / "artifacts",
        winner_model,
        list(selected_features),
        manifest,
        feature_frame=working,
        predictions=winner_holdout,
        metrics=selected_walk,
    )
    return {
        "frame": working,
        "blocks": selected_blocks,
        "tuning": tuning,
        "metrics": selected_walk,
        "holdout_metrics": selected_holdout,
        "predictions": selected_predictions,
        "holdout_predictions": holdout_predictions,
        "winner": manifest,
        "selected_configs": selected_specs,
        "model": winner_model,
        "selected_features": list(selected_features),
    }


def run_focused_feature_experiment(
    frame: pd.DataFrame,
    output_dir: str | Path = "focused_feature_artifacts",
    include_xgboost: bool = True,
    complete_case: bool = True,
    evaluation: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Test targeted unions of the strongest language feature families.

    This is intentionally narrower than ``all_features``.  It tests whether
    sentence-level language, company-relative language surprise, and
    earnings/guidance proxies add incremental information when combined.
    Selection uses walk-forward results only; the holdout is reported after
    selection and is never used to choose a block or configuration.
    """
    evaluation = evaluation or EvaluationConfig(random_state=42, bootstrap_repetitions=250)
    working = frame.copy().loc[:, ~frame.columns.duplicated()]
    required = {"event_year", "label", "symbol", "abnormal_return_5d"}
    missing = required.difference(working.columns)
    if missing:
        raise KeyError(f"Focused experiment frame is missing required columns: {sorted(missing)}")
    if complete_case:
        required_language = {"pres_sent_mean", "qa_sent_mean"}
        if not required_language.issubset(working.columns):
            raise KeyError("Complete-case focused experiment requires sentence sentiment columns.")
        working = working[working["pres_sent_mean"].notna() & working["qa_sent_mean"].notna()].copy()

    base_blocks = build_final_feature_blocks(working)
    for name in ("sentence_sentiment", "historical_surprise", "earnings_language_proxy"):
        if not base_blocks.get(name):
            raise ValueError(f"Focused experiment requires a non-empty {name} block.")

    def union(*names: str) -> list[str]:
        return list(dict.fromkeys(column for name in names for column in base_blocks[name]))

    # Single blocks are references.  The three unions are the actual focused
    # candidates; none includes the noisy dictionary block or all market
    # controls, so incremental language value is isolated.
    blocks = {
        "sentence_sentiment": base_blocks["sentence_sentiment"],
        "historical_surprise": base_blocks["historical_surprise"],
        "earnings_language_proxy": base_blocks["earnings_language_proxy"],
        "sentence_plus_historical": union("sentence_sentiment", "historical_surprise"),
        "historical_plus_earnings": union("historical_surprise", "earnings_language_proxy"),
        "sentence_plus_historical_plus_earnings": union(
            "sentence_sentiment", "historical_surprise", "earnings_language_proxy"
        ),
    }
    combination_blocks = {
        "sentence_plus_historical",
        "historical_plus_earnings",
        "sentence_plus_historical_plus_earnings",
    }
    specs = e9_model_grid(include_xgboost=include_xgboost)
    tuning_rows: list[pd.DataFrame] = []
    tuning_predictions: list[pd.DataFrame] = []
    for block_name, features in blocks.items():
        for spec in specs:
            metrics, predictions = _evaluate_spec(
                working, block_name, features, spec, "label", evaluation,
                bootstrap_repetitions=0,
            )
            if not metrics.empty:
                tuning_rows.append(metrics.assign(tuning_stage="focused_walk_forward"))
                tuning_predictions.append(predictions)
    tuning = pd.concat(tuning_rows, ignore_index=True) if tuning_rows else pd.DataFrame()
    if tuning.empty:
        raise RuntimeError("Focused feature experiment produced no tuning results.")

    selected_specs: dict[str, dict] = {}
    selected_rows: list[pd.Series] = []
    for block_name in blocks:
        best = _select_best(tuning[tuning["feature_block"] == block_name])
        selected_rows.append(best)
        selected_specs[block_name] = next(spec for spec in specs if spec["name"] == best["model"])

    selected = pd.DataFrame(selected_rows)
    focused_selected = selected[selected["feature_block"].isin(combination_blocks)]
    overall = _select_best(focused_selected)
    winner_block = str(overall["feature_block"])
    winner_model_name = str(overall["model"])
    winner_spec = selected_specs[winner_block]

    walk_rows: list[pd.DataFrame] = []
    walk_predictions: list[pd.DataFrame] = []
    holdout_rows: list[pd.DataFrame] = []
    holdout_predictions: list[pd.DataFrame] = []
    for block_name, spec in selected_specs.items():
        walk_metrics, walk_pred = _evaluate_spec(
            working, block_name, blocks[block_name], spec, "label", evaluation,
            bootstrap_repetitions=0,
        )
        holdout_metrics, holdout_pred = _evaluate_spec(
            working, block_name, blocks[block_name], spec, "label", evaluation,
            holdout=True, bootstrap_repetitions=0,
        )
        if not walk_metrics.empty:
            walk_rows.append(walk_metrics.assign(evaluation_stage="selected_config"))
            walk_predictions.append(walk_pred)
        if not holdout_metrics.empty:
            holdout_rows.append(holdout_metrics.assign(evaluation_stage="selected_config"))
            holdout_predictions.append(holdout_pred)

    metrics = pd.concat(walk_rows, ignore_index=True) if walk_rows else pd.DataFrame()
    holdout_metrics = pd.concat(holdout_rows, ignore_index=True) if holdout_rows else pd.DataFrame()
    predictions = pd.concat(walk_predictions, ignore_index=True) if walk_predictions else pd.DataFrame()
    holdout_prediction_frame = pd.concat(holdout_predictions, ignore_index=True) if holdout_predictions else pd.DataFrame()

    winner_predictions = predictions[
        (predictions["feature_block"] == winner_block)
        & (predictions["model"] == winner_model_name)
    ]
    lower, upper = cluster_bootstrap(
        winner_predictions, "auc", repetitions=1000, random_state=evaluation.random_state
    )
    winner_mask = (
        (metrics["feature_block"] == winner_block)
        & (metrics["model"] == winner_model_name)
        & (metrics["split"] == "walk_forward_aggregate")
    )
    metrics.loc[winner_mask, "auc_lower_95"] = lower
    metrics.loc[winner_mask, "auc_upper_95"] = upper
    winner_holdout = holdout_metrics[
        (holdout_metrics["feature_block"] == winner_block)
        & (holdout_metrics["model"] == winner_model_name)
        & (holdout_metrics["split"] == "final_holdout")
    ]
    manifest = {
        "experiment": "focused language feature union comparison",
        "sample_rows": int(len(working)),
        "companies": int(working["symbol"].nunique()),
        "complete_case": bool(complete_case),
        "blocks": list(blocks),
        "combination_blocks": sorted(combination_blocks),
        "winner_block": winner_block,
        "winner_model": winner_model_name,
        "winner_spec": {key: value for key, value in winner_spec.items() if key != "config"},
        "selected_configs": {
            block: {key: value for key, value in spec.items() if key != "config"}
            for block, spec in selected_specs.items()
        },
        "selection_rule": "highest mean 2019-2022 walk-forward AUC, then lowest mean walk-forward log loss; holdout excluded",
        "walk_forward_years": list(evaluation.walk_forward_years),
        "holdout_cutoff_year": evaluation.final_cutoff_year,
        "random_state": evaluation.random_state,
        "winner_auc_lower_95": lower,
        "winner_auc_upper_95": upper,
        "winner_mean_walk_auc": float(overall["mean_fold_auc"]),
        "winner_pooled_walk_auc": float(overall["auc"]),
        "winner_holdout_auc": float(winner_holdout["auc"].iloc[0]) if not winner_holdout.empty else np.nan,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tuning.to_csv(output / "tuning_results.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    holdout_metrics.to_csv(output / "holdout_metrics.csv", index=False)
    pd.concat(tuning_predictions, ignore_index=True).to_csv(output / "tuning_predictions.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    holdout_prediction_frame.to_csv(output / "holdout_predictions.csv", index=False)
    (output / "feature_blocks.json").write_text(json.dumps(blocks, indent=2))
    (output / "feature_schema.json").write_text(json.dumps({
        "feature_block": winner_block,
        "feature_columns": blocks[winner_block],
        "blocks": blocks,
        "model": winner_model_name,
    }, indent=2))
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    pd.DataFrame([{
        "rows": len(working),
        "companies": working["symbol"].nunique(),
        "positive_rate": working["label"].mean(),
        "winner_block": winner_block,
        "winner_model": winner_model_name,
        "winner_mean_walk_auc": manifest["winner_mean_walk_auc"],
        "winner_pooled_walk_auc": manifest["winner_pooled_walk_auc"],
        "winner_holdout_auc": manifest["winner_holdout_auc"],
    }]).to_csv(output / "target_audit.csv", index=False)

    development = working[working["event_year"] < evaluation.final_cutoff_year].dropna(subset=["label"]).copy()
    winner_model = _fit_spec(winner_spec, development, blocks[winner_block], "label", evaluation)
    selected_features = getattr(winner_model, "selected_features_", blocks[winner_block])
    winner_holdout_predictions = holdout_prediction_frame[
        (holdout_prediction_frame["feature_block"] == winner_block)
        & (holdout_prediction_frame["model"] == winner_model_name)
    ]
    save_artifact_bundle(
        output / "artifacts", winner_model, list(selected_features), manifest,
        feature_frame=working, predictions=winner_holdout_predictions, metrics=metrics,
    )
    return {
        "frame": working,
        "blocks": blocks,
        "tuning": tuning,
        "metrics": metrics,
        "holdout_metrics": holdout_metrics,
        "predictions": predictions,
        "holdout_predictions": holdout_prediction_frame,
        "winner": manifest,
        "selected_configs": selected_specs,
        "model": winner_model,
        "selected_features": list(selected_features),
    }


def apply_stratified_event_cap(
    frame: pd.DataFrame,
    cap: int = 3000,
    seed: int = 42,
    strata: Sequence[str] = ("gics_sector", "event_year"),
) -> pd.DataFrame:
    """Deterministically cap events without using labels or future returns."""
    if len(frame) <= cap:
        return frame.copy()
    available = [column for column in strata if column in frame.columns]
    if not available:
        return frame.sample(n=cap, random_state=seed).sort_index().copy()
    result = frame.copy()
    result["_sampling_key"] = result[available].astype(str).agg("|".join, axis=1)
    groups = list(result.groupby("_sampling_key", sort=True, dropna=False))
    total = len(result)
    allocations = {key: int(cap * len(group) // total) for key, group in groups}
    if len(groups) <= cap:
        allocations = {key: max(1, value) for key, value in allocations.items()}
    remainder = cap - sum(allocations.values())
    ranked_groups = sorted(
        groups,
        key=lambda item: (cap * len(item[1]) / total - allocations[item[0]]),
        reverse=True,
    )
    for key, group in ranked_groups:
        if remainder <= 0:
            break
        if allocations[key] < len(group):
            allocations[key] += 1
            remainder -= 1
    if sum(allocations.values()) > cap:
        for key, group in sorted(groups, key=lambda item: allocations[item[0]], reverse=True):
            while allocations[key] > 0 and sum(allocations.values()) > cap:
                allocations[key] -= 1
    sampled = []
    for key, group in groups:
        n = min(allocations[key], len(group))
        digest = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16)
        sampled.append(group.sample(n=n, random_state=seed + digest % 100000))
    return pd.concat(sampled).drop(columns=["_sampling_key"]).sort_values(
        [column for column in ["event_year", "symbol", "call_datetime"] if column in result]
    ).reset_index(drop=True)


def compare_frozen_configs(
    frames: Mapping[str, pd.DataFrame],
    e9_results: Mapping[str, object],
    output_dir: str | Path = "sector_artifacts",
    complete_case: bool = False,
    evaluation: EvaluationConfig | None = None,
) -> dict[str, object]:
    """Apply Industrials-selected configs unchanged to sector/universe frames."""
    evaluation = evaluation or EvaluationConfig(random_state=42, bootstrap_repetitions=250)
    blocks = e9_results["blocks"]
    selected_configs = e9_results["selected_configs"]
    rows = []
    predictions = []
    audits = []
    for universe, source in frames.items():
        frame = source.copy().loc[:, ~source.columns.duplicated()]
        if complete_case and {"pres_sent_mean", "qa_sent_mean"}.issubset(frame.columns):
            frame = frame[frame["pres_sent_mean"].notna() & frame["qa_sent_mean"].notna()].copy()
        if "label" not in frame and "abnormal_return_5d" in frame:
            frame["label"] = (frame["abnormal_return_5d"] > 0).astype(int)
        audits.append({
            "universe": universe,
            "rows": len(frame),
            "companies": frame["symbol"].nunique() if "symbol" in frame else np.nan,
        "complete_language_rows": int((frame["pres_sent_mean"].notna() & frame["qa_sent_mean"].notna()).sum()) if {"pres_sent_mean", "qa_sent_mean"}.issubset(frame.columns) else np.nan,
        })
        for block_name, spec in selected_configs.items():
            metrics, prediction = _evaluate_spec(
                frame,
                block_name,
                blocks[block_name],
                spec,
                "label",
                evaluation,
                bootstrap_repetitions=250,
            )
            if not metrics.empty:
                rows.append(metrics.assign(universe=universe, transfer_mode="frozen_industrials_config"))
            if not prediction.empty:
                predictions.append(prediction.assign(universe=universe, transfer_mode="frozen_industrials_config"))
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    audit = pd.DataFrame(audits)
    metrics.to_csv(output / "metrics.csv", index=False)
    prediction_frame.to_csv(output / "predictions.csv", index=False)
    audit.to_csv(output / "target_audit.csv", index=False)
    (output / "feature_blocks.json").write_text(json.dumps(blocks, indent=2))
    (output / "feature_schema.json").write_text(json.dumps({
        "blocks": blocks,
        "selected_configs": {
            block: {key: value for key, value in spec.items() if key != "config"}
            for block, spec in selected_configs.items()
        },
        "transfer_mode": "frozen_industrials_config",
    }, indent=2, default=str))
    manifest = {
        "experiment": "frozen cross-universe transfer",
        "universes": list(frames),
        "complete_case": complete_case,
        "transfer_mode": "frozen_industrials_config",
        "selection_source": e9_results["winner"],
        "random_state": evaluation.random_state,
        "walk_forward_years": list(evaluation.walk_forward_years),
        "holdout_cutoff_year": evaluation.final_cutoff_year,
        "cache_version": E9_CACHE_VERSION,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return {"metrics": metrics, "predictions": prediction_frame, "audit": audit, "manifest": manifest}
