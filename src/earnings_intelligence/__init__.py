"""Reusable components for the earnings-call intelligence project."""

from .config import EventConfig, EvaluationConfig, ModelConfig
from .events import add_beta_adjusted_targets, add_precall_market_features, build_event_dataset
from .features import add_expanding_history_features
from .final_pipeline import (
    build_final_feature_blocks,
    build_universe_audit,
    build_universe_frame,
    download_market_data,
    load_industrials_streaming,
    load_sector_streaming,
    run_final_experiment,
)
from .experiments import (
    apply_stratified_event_cap,
    compare_frozen_configs,
    run_e9_experiment,
    run_focused_feature_experiment,
    validate_language_cache,
)
from .modeling import evaluate_holdout, evaluate_walk_forward

__all__ = [
    "EventConfig",
    "EvaluationConfig",
    "ModelConfig",
    "add_beta_adjusted_targets",
    "add_precall_market_features",
    "add_expanding_history_features",
    "build_event_dataset",
    "build_final_feature_blocks",
    "build_universe_audit",
    "build_universe_frame",
    "download_market_data",
    "load_industrials_streaming",
    "load_sector_streaming",
    "evaluate_holdout",
    "evaluate_walk_forward",
    "run_final_experiment",
    "run_e9_experiment",
    "run_focused_feature_experiment",
    "compare_frozen_configs",
    "apply_stratified_event_cap",
    "validate_language_cache",
]
