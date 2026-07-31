"""Point-in-time event labels and market-model robustness targets."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .config import EventConfig


def _normalise_index(index: pd.Index) -> pd.DatetimeIndex:
    dates = pd.to_datetime(index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    return dates.normalize()


def _normalise_prices(prices: pd.DataFrame) -> pd.DataFrame:
    result = prices.copy()
    result.index = _normalise_index(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _normalise_series(values: pd.Series) -> pd.Series:
    result = values.copy()
    result.index = _normalise_index(result.index)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _phase(call_datetime: pd.Timestamp, config: EventConfig) -> str:
    minutes = call_datetime.hour * 60 + call_datetime.minute
    if minutes < config.market_open_minute:
        return "pre_open"
    if minutes >= config.market_close_minute:
        return "after_close"
    return "intraday"


def build_event_dataset(
    transcripts: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    benchmark: pd.Series,
    config: EventConfig | None = None,
) -> pd.DataFrame:
    """Build causally aligned event labels from daily prices.

    Intraday calls are excluded because daily data cannot distinguish price
    movement before and after the transcript. The returned frame stores an
    ``event_audit`` dictionary in ``DataFrame.attrs`` with exclusion counts.
    Stock and benchmark returns use the same configured price basis.
    """
    config = config or EventConfig()
    if config.date_col not in transcripts or config.symbol_col not in transcripts:
        raise KeyError(f"transcripts must contain {config.date_col!r} and {config.symbol_col!r}")

    source = transcripts.copy()
    source[config.call_datetime_col] = pd.to_datetime(source[config.date_col], errors="coerce")
    source = source.dropna(subset=[config.call_datetime_col]).copy()
    source[config.event_year_col] = source[config.call_datetime_col].dt.year.astype(int)
    source[config.phase_col] = source[config.call_datetime_col].map(lambda x: _phase(x, config))

    benchmark = _normalise_series(benchmark).dropna()
    records = []
    audit = {
        "input_rows": int(len(source)),
        "intraday_excluded": 0,
        "missing_price_data": 0,
        "missing_baseline": 0,
        "missing_future_target": 0,
        "invalid_ticker": 0,
        "retained_rows": 0,
    }

    for _, row in source.iterrows():
        phase = row[config.phase_col]
        if phase == "intraday":
            audit["intraday_excluded"] += 1
            continue

        ticker = str(row[config.symbol_col])
        if ticker not in price_data:
            audit["invalid_ticker"] += 1
            continue
        prices = _normalise_prices(price_data[ticker])
        if config.price_col not in prices:
            audit["missing_price_data"] += 1
            continue
        prices = prices.dropna(subset=[config.price_col])
        if prices.empty:
            audit["missing_price_data"] += 1
            continue

        call_day = pd.Timestamp(row[config.call_datetime_col]).normalize()
        start_day = call_day + pd.Timedelta(days=1) if phase == "after_close" else call_day
        candidates = prices.index[prices.index >= start_day]
        if len(candidates) == 0:
            audit["missing_future_target"] += 1
            continue
        t0_date = candidates[0]
        t0_idx = prices.index.get_loc(t0_date)
        if t0_idx == 0:
            audit["missing_baseline"] += 1
            continue

        baseline_date = prices.index[t0_idx - 1]
        baseline_price = float(prices.loc[baseline_date, config.price_col])
        benchmark_before = benchmark.index[benchmark.index < t0_date]
        if len(benchmark_before) == 0:
            audit["missing_baseline"] += 1
            continue
        benchmark_baseline_date = benchmark_before[-1]
        benchmark_baseline = float(benchmark.loc[benchmark_baseline_date])
        if not np.isfinite(baseline_price) or baseline_price <= 0 or not np.isfinite(benchmark_baseline) or benchmark_baseline <= 0:
            audit["missing_baseline"] += 1
            continue

        target_values = {}
        complete = True
        for horizon in config.horizons:
            target_idx = t0_idx + horizon - 1
            if target_idx >= len(prices.index):
                complete = False
                break
            future_date = prices.index[target_idx]
            benchmark_future = benchmark.index[benchmark.index >= future_date]
            if len(benchmark_future) == 0:
                complete = False
                break
            future_price = float(prices.loc[future_date, config.price_col])
            future_benchmark = float(benchmark.loc[benchmark_future[0]])
            if future_price <= 0 or future_benchmark <= 0:
                complete = False
                break
            stock_return = future_price / baseline_price - 1.0
            market_return = future_benchmark / benchmark_baseline - 1.0
            target_values[f"abnormal_return_{horizon}d"] = stock_return - market_return

        if not complete:
            audit["missing_future_target"] += 1
            continue

        record = row.to_dict()
        record.update(
            {
                "target_baseline_date": baseline_date,
                "target_end_date_5d": prices.index[t0_idx + 4] if t0_idx + 4 < len(prices.index) else pd.NaT,
                "target_price_basis": config.price_col,
                **target_values,
            }
        )
        records.append(record)

    result = pd.DataFrame(records)
    audit["retained_rows"] = int(len(result))
    result.attrs["event_audit"] = audit
    return result


def add_beta_adjusted_targets(
    frame: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    benchmark: pd.Series,
    config: EventConfig | None = None,
    horizon: int = 5,
) -> pd.DataFrame:
    """Add a pre-event market-model cumulative abnormal return.

    Alpha and beta are estimated only from returns strictly before the call
    date. The output column is ``beta_abnormal_return_{horizon}d``.
    """
    config = config or EventConfig()
    result = frame.copy()
    benchmark = _normalise_series(benchmark).dropna()
    values = []

    for _, row in result.iterrows():
        ticker = str(row[config.symbol_col])
        call_time = pd.Timestamp(row[config.call_datetime_col])
        phase = row.get(config.phase_col, _phase(call_time, config))
        if phase == "intraday" or ticker not in price_data:
            values.append(np.nan)
            continue
        prices = _normalise_prices(price_data[ticker])
        if config.price_col not in prices:
            values.append(np.nan)
            continue
        prices = prices[config.price_col].dropna()
        call_day = call_time.normalize()
        pre_event = prices[prices.index < call_day]
        market_pre_event = benchmark[benchmark.index < call_day]
        stock_returns = np.log(pre_event).diff().dropna()
        market_returns = np.log(market_pre_event).diff().dropna()
        joined = pd.concat([stock_returns.rename("stock"), market_returns.rename("market")], axis=1).dropna().tail(config.estimation_window)
        if len(joined) < config.min_beta_observations or joined["market"].var() <= 0:
            values.append(np.nan)
            continue
        design = np.column_stack([np.ones(len(joined)), joined["market"].to_numpy()])
        alpha, beta = np.linalg.lstsq(design, joined["stock"].to_numpy(), rcond=None)[0]

        start_day = call_day + pd.Timedelta(days=1) if phase == "after_close" else call_day
        post_dates = prices.index[prices.index >= start_day]
        if len(post_dates) < horizon:
            values.append(np.nan)
            continue
        end_date = post_dates[horizon - 1]
        stock_baseline_candidates = prices.index[prices.index < post_dates[0]]
        market_baseline_candidates = benchmark.index[benchmark.index < post_dates[0]]
        if len(stock_baseline_candidates) == 0 or len(market_baseline_candidates) == 0:
            values.append(np.nan)
            continue
        stock_window = prices[(prices.index >= stock_baseline_candidates[-1]) & (prices.index <= end_date)]
        market_window = benchmark[(benchmark.index >= market_baseline_candidates[-1]) & (benchmark.index <= end_date)]
        stock_log = np.log(stock_window).diff().dropna()
        market_log = np.log(market_window).diff().dropna()
        post = pd.concat([stock_log.rename("stock"), market_log.rename("market")], axis=1).dropna()
        if len(post) < horizon:
            values.append(np.nan)
            continue
        values.append(float(post["stock"].sum() - (alpha * len(post) + beta * post["market"].sum())))

    result[f"beta_abnormal_return_{horizon}d"] = values
    return result


def add_precall_market_features(
    frame: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    benchmark: pd.Series,
    config: EventConfig | None = None,
) -> pd.DataFrame:
    """Add momentum, volatility, market momentum, and pre-call beta features."""
    config = config or EventConfig()
    benchmark = _normalise_series(benchmark).dropna()
    result = frame.copy()
    feature_rows = []
    for _, row in result.iterrows():
        ticker = str(row[config.symbol_col])
        call_time = pd.Timestamp(row[config.call_datetime_col])
        phase = row.get(config.phase_col, _phase(call_time, config))
        missing = {"momentum_5d": np.nan, "momentum_20d": np.nan, "volatility_20d": np.nan, "market_momentum_20d": np.nan, "beta_120d": np.nan}
        if phase == "intraday" or ticker not in price_data:
            feature_rows.append(missing)
            continue
        prices = _normalise_prices(price_data[ticker])
        price_column = "Adj Close" if "Adj Close" in prices else config.price_col
        if price_column not in prices:
            feature_rows.append(missing)
            continue
        call_day = call_time.normalize()
        cutoff = call_day + pd.Timedelta(days=1) if phase == "after_close" else call_day
        available = prices.loc[prices.index < cutoff, price_column].dropna()
        market_available = benchmark[benchmark.index < cutoff]
        if len(available) >= 21:
            missing["momentum_5d"] = float(available.iloc[-1] / available.iloc[-6] - 1.0)
            missing["momentum_20d"] = float(available.iloc[-1] / available.iloc[-21] - 1.0)
            missing["volatility_20d"] = float(available.iloc[-21:].pct_change().dropna().std())
        if len(market_available) >= 21:
            missing["market_momentum_20d"] = float(market_available.iloc[-1] / market_available.iloc[-21] - 1.0)
        stock_returns = np.log(available).diff().dropna()
        market_returns = np.log(market_available).diff().dropna()
        joined = pd.concat([stock_returns.rename("stock"), market_returns.rename("market")], axis=1).dropna().tail(config.estimation_window)
        if len(joined) >= config.min_beta_observations and joined["market"].var() > 0:
            design = np.column_stack([np.ones(len(joined)), joined["market"].to_numpy()])
            missing["beta_120d"] = float(np.linalg.lstsq(design, joined["stock"].to_numpy(), rcond=None)[0][1])
        feature_rows.append(missing)
    return pd.concat([result.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)
