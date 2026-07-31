"""Causal historical normalization and feature-block utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_expanding_history_features(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    group_col: str = "symbol",
    timestamp_col: str = "call_datetime",
    sector_col: str = "gics_sector",
    min_history: int = 4,
    clip: float = 5.0,
) -> pd.DataFrame:
    """Add prior-only company and sector z-scores.

    For each row, the current value is compared with statistics from earlier
    calls only. Company history is preferred; sector history is used only when
    company history has fewer than ``min_history`` observations. The returned
    frame includes one history count per base feature and a source indicator.
    """
    result = frame.copy()
    if group_col not in result or timestamp_col not in result:
        raise KeyError(f"frame must contain {group_col!r} and {timestamp_col!r}")
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    result["_original_order"] = np.arange(len(result))
    result = result.sort_values([timestamp_col, group_col, "_original_order"], kind="mergesort").copy()
    source_values = pd.Series("none", index=result.index, dtype="object")

    for column in feature_columns:
        if column not in result:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        company_stats = _prior_stats(result, values, result[group_col])
        company_z, company_count = _z_score(values, company_stats["mean"], company_stats["std"], company_stats["count"], min_history, clip)

        if sector_col in result:
            sector_stats = _prior_stats(result, values, result[sector_col])
            sector_z, sector_count = _z_score(values, sector_stats["mean"], sector_stats["std"], sector_stats["count"], min_history, clip)
        else:
            sector_z = pd.Series(np.nan, index=result.index)
            sector_count = pd.Series(0.0, index=result.index)

        use_company = company_z.notna()
        use_sector = ~use_company & sector_z.notna()
        result[f"{column}_z"] = company_z.where(use_company, sector_z)
        result[f"{column}_history_count"] = company_count.where(use_company, sector_count)
        source_values.loc[use_company] = "company"
        source_values.loc[use_sector] = "sector"

    result["historical_score_source"] = source_values
    result = result.sort_values("_original_order", kind="mergesort").drop(columns=["_original_order"])
    return result


def _prior_stats(frame: pd.DataFrame, values: pd.Series, groups: pd.Series) -> dict[str, pd.Series]:
    grouped = values.groupby(groups, sort=False)
    prior_count = grouped.cumcount().astype(float)
    prior_sum = grouped.cumsum() - values
    prior_sum_sq = (values ** 2).groupby(groups, sort=False).cumsum() - values ** 2
    mean = prior_sum / prior_count.replace(0, np.nan)
    variance = (prior_sum_sq - prior_count * mean ** 2) / (prior_count - 1).replace(0, np.nan)
    return {"count": prior_count, "mean": mean, "std": np.sqrt(variance.clip(lower=0))}


def _z_score(values, mean, std, count, min_history, clip):
    z = (values - mean) / std.replace(0, np.nan)
    z = z.where((count >= min_history) & np.isfinite(z)).clip(-clip, clip)
    return z, count


def feature_block_columns(frame: pd.DataFrame, blocks: dict[str, Sequence[str]]) -> dict[str, list[str]]:
    """Keep only columns present in each declared feature block."""
    return {name: [column for column in columns if column in frame.columns] for name, columns in blocks.items()}
