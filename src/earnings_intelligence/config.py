"""Configuration objects shared by the event, feature, and model pipelines."""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class EventConfig:
    """Rules for constructing point-in-time event labels."""

    date_col: str = "date"
    symbol_col: str = "symbol"
    call_datetime_col: str = "call_datetime"
    phase_col: str = "call_phase"
    event_year_col: str = "event_year"
    price_col: str = "Close"
    horizons: Tuple[int, ...] = (1, 3, 5, 30)
    market_open_minute: int = 9 * 60 + 30
    market_close_minute: int = 16 * 60
    timezone: str = "America/New_York"
    estimation_window: int = 120
    min_beta_observations: int = 60


@dataclass(frozen=True)
class EvaluationConfig:
    """Reproducible time-series evaluation settings."""

    final_cutoff_year: int = 2023
    walk_forward_years: Tuple[int, ...] = (2019, 2020, 2021, 2022)
    probability_quintile: float = 0.20
    bootstrap_repetitions: int = 1000
    permutation_repetitions: int = 1000
    random_state: int = 42
    correlation_threshold: float = 0.95


@dataclass(frozen=True)
class ModelConfig:
    """Conservative model settings used after feature construction."""

    logistic_c: float = 1.0
    logistic_max_iter: int = 2000
    elastic_net_c: float = 0.1
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 3000
    xgb_estimators: int = 150
    xgb_depth: int = 2
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: int = 10
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_random_state: int = 42
    extra_params: dict = field(default_factory=dict)
