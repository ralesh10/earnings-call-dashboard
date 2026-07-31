"""One-pass, reproducible Colab experiment for the earnings-call project.

This module deliberately separates reusable transcript features from event
labels. A language-feature cache may be reused, but abnormal-return targets
are rebuilt from the current adjusted-price event rules on every run.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .artifacts import save_artifact_bundle
from .config import EvaluationConfig, EventConfig, ModelConfig
from .events import add_beta_adjusted_targets, add_precall_market_features, build_event_dataset
from .features import add_expanding_history_features
from .modeling import (
    classification_metrics,
    cluster_bootstrap,
    fit_elastic_net_model,
    fit_logistic_model,
    fit_xgb_model,
    top_bottom_spread,
)
from .text_features import (
    add_dictionary_features,
    add_earnings_language_features,
    build_sentence_feature_frame,
    extract_structural_text,
    load_lm_lexicons,
    make_finbert_sentence_scorer,
)


CACHE_VERSION = "language-features-v4-compact-text-free"
TARGET_VERSION = "adjusted-close-abnormal-return-v2"
PRIMARY_TARGET = "market_subtracted"


def load_sector_streaming(
    dataset_name: str = "Bose345/sp500_earnings_transcripts",
    split: str = "train",
    min_event_year: int = 2015,
    sectors: Sequence[str] | None = ("Industrials",),
    metadata_only: bool = False,
    selected_keys: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Load selected sector rows from Hugging Face without materializing all transcripts.

    The ordinary ``load_dataset`` path can consume substantial RAM because it
    creates a DataFrame for every transcript before sector filtering.  This
    two-pass streaming loader first keeps only candidate symbols, then retains
    transcript rows whose current sector is in ``sectors``. ``None`` retains
    every mapped sector. ``metadata_only`` retains only compact identifying
    fields, which is suitable for universe audits. ``selected_keys`` can be
    used to load a deterministic subset without retaining unrelated text.
    """
    from datasets import load_dataset

    def stream_rows():
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        return dataset

    candidate_symbols: set[str] = set()
    scanned = 0
    for row in stream_rows():
        scanned += 1
        symbol = str(row.get("symbol", "")).replace("-", ".").replace("/", ".").strip().upper()
        content = str(row.get("content", ""))
        timestamp = pd.to_datetime(row.get("date"), errors="coerce")
        word_count = pd.to_numeric(row.get("word_count"), errors="coerce")
        if pd.isna(word_count):
            word_count = len(content.split())
        if pd.notna(timestamp) and int(timestamp.year) >= min_event_year and 500 < int(word_count) < 20000:
            candidate_symbols.add(symbol)
        if scanned % 5000 == 0:
            print(f"Streaming transcript scan: {scanned} rows...")

    sector_lookup = _sector_mapping(sorted(candidate_symbols))
    selected_sectors = None if sectors is None else set(sectors)
    selected_symbols = {
        symbol for symbol, sector in sector_lookup.items()
        if selected_sectors is None or sector in selected_sectors
    }
    if not selected_symbols:
        raise RuntimeError(f"Streaming dataset scan found no symbols for sectors={sectors!r}.")

    retained = []
    normalized_keys = None
    if selected_keys is not None:
        normalized_keys = {
            (str(symbol), pd.Timestamp(pd.to_datetime(timestamp)).isoformat())
            for symbol, timestamp in selected_keys
        }
    scanned = 0
    for row in stream_rows():
        scanned += 1
        symbol = str(row.get("symbol", "")).replace("-", ".").replace("/", ".").strip().upper()
        if symbol not in selected_symbols:
            continue
        timestamp = pd.to_datetime(row.get("date"), errors="coerce")
        word_count = pd.to_numeric(row.get("word_count"), errors="coerce")
        content = str(row.get("content", ""))
        if pd.isna(word_count):
            word_count = len(content.split())
        if pd.isna(timestamp) or int(timestamp.year) < min_event_year or not (500 < int(word_count) < 20000):
            continue
        row_key = (symbol, pd.Timestamp(timestamp).isoformat())
        if normalized_keys is not None and row_key not in normalized_keys:
            continue
        record = dict(row)
        record["symbol"] = symbol
        record["gics_sector"] = sector_lookup[symbol]
        record["word_count"] = int(word_count)
        record["call_datetime"] = timestamp
        if metadata_only:
            keep = [
                "symbol", "company_name", "company_id", "quarter", "year",
                "date", "call_datetime", "word_count", "gics_sector",
            ]
            record = {key: record[key] for key in keep if key in record}
        retained.append(record)
        if len(retained) % 1000 == 0:
            print(f"Streaming sector rows retained: {len(retained)}")
    result = pd.DataFrame(retained)
    print(f"Loaded streaming sector frame ({sectors!r}): {result.shape}")
    return result


def load_industrials_streaming(
    dataset_name: str = "Bose345/sp500_earnings_transcripts",
    split: str = "train",
    min_event_year: int = 2015,
) -> pd.DataFrame:
    """Backward-compatible wrapper for the original Industrials loader."""
    return load_sector_streaming(dataset_name, split, min_event_year, sectors=("Industrials",))


def _normalise_tickers(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"[-/]", ".", regex=True).str.strip().str.upper()


def _sector_mapping(symbols: Sequence[str]) -> dict[str, str]:
    """Fetch current GICS labels plus historical ticker fallbacks."""
    import requests

    from io import StringIO

    mapping: dict[str, str] = {}
    try:
        response = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "earnings-intelligence-research/1.0"},
            timeout=30,
        )
        table = pd.read_html(StringIO(response.text))[0][["Symbol", "GICS Sector"]]
        table.columns = ["symbol", "gics_sector"]
        table["symbol"] = _normalise_tickers(table["symbol"])
        mapping.update(dict(zip(table["symbol"], table["gics_sector"])))
    except Exception as exc:
        print(f"Warning: current sector table unavailable: {exc}")

    historical = {
        "PARA": "Communication Services", "WBA": "Consumer Staples", "LUMN": "Communication Services",
        "IPG": "Communication Services", "BK": "Financials", "K": "Consumer Staples",
        "BBWI": "Consumer Discretionary", "WHR": "Consumer Discretionary", "RHI": "Industrials",
        "MMC": "Financials", "AAL": "Industrials", "CMA": "Financials", "CAG": "Consumer Staples",
        "ZION": "Financials", "JNPR": "Information Technology", "ANSS": "Information Technology",
        "CTRA": "Energy", "DAY": "Information Technology", "FI": "Financials", "HES": "Energy",
        "SEE": "Materials", "HBI": "Consumer Discretionary",
    }
    mapping.update(historical)
    return {str(symbol): mapping[str(symbol)] for symbol in symbols if str(symbol) in mapping}


def _prepare_sector_frame(
    raw: pd.DataFrame,
    sectors: Sequence[str] | None = ("Industrials",),
) -> pd.DataFrame:
    frame = raw.copy()
    if "content" not in frame or "symbol" not in frame or "date" not in frame:
        raise KeyError("Transcript data must contain content, symbol, and date columns.")
    if "word_count" not in frame:
        frame["word_count"] = frame["content"].astype(str).str.split().str.len()
    frame = frame[frame["word_count"] > 500].copy()
    frame["symbol"] = _normalise_tickers(frame["symbol"])
    frame["call_datetime"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["call_datetime"])
    if "gics_sector" not in frame:
        lookup = _sector_mapping(frame["symbol"].unique())
        frame["gics_sector"] = frame["symbol"].map(lookup)
    frame = frame.dropna(subset=["gics_sector"]).copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce") if "year" in frame else frame["call_datetime"].dt.year
    frame = frame[(frame["year"] >= 2008) & (frame["word_count"] < 20000)]
    frame["event_year"] = frame["call_datetime"].dt.year.astype(int)
    frame = frame[frame["event_year"] >= 2015].copy()
    if sectors is not None:
        frame = frame[frame["gics_sector"].isin(set(sectors))].copy()
    return frame.reset_index(drop=True)


def _ticker_frame(downloaded: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if isinstance(downloaded.columns, pd.MultiIndex):
        levels = [set(downloaded.columns.get_level_values(i)) for i in range(downloaded.columns.nlevels)]
        if ticker in levels[0]:
            return downloaded[ticker].copy()
        if ticker in levels[1]:
            return downloaded.xs(ticker, axis=1, level=1).copy()
        return None
    return downloaded.copy() if ticker == "^GSPC" else None


def _download_market_data(tickers: Sequence[str], start: str = "2007-01-01", end: str = "2026-06-01"):
    import yfinance as yf

    tickers = sorted(set(map(str, tickers)))
    downloaded = yf.download(
        tickers + ["^GSPC"],
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )
    price_data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame = _ticker_frame(downloaded, ticker)
        if frame is not None and "Adj Close" in frame:
            price_data[ticker] = frame
    benchmark_frame = _ticker_frame(downloaded, "^GSPC")
    if benchmark_frame is None:
        raise RuntimeError("Benchmark ^GSPC data was not downloaded.")
    benchmark_column = "Adj Close" if "Adj Close" in benchmark_frame else "Close"
    benchmark = benchmark_frame[benchmark_column].dropna()
    return price_data, benchmark


def download_market_data(tickers: Sequence[str], start: str = "2007-01-01", end: str = "2026-06-01"):
    """Public wrapper used by the multi-universe experiment runner."""
    return _download_market_data(tickers, start=start, end=end)


def _add_probability_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for section in ("pres", "qa"):
        for label in ("pos", "neg", "neu"):
            target = f"{section}_{label}"
            source = f"{section}_{label}_mean"
            if target not in result and source in result:
                result[target] = result[source]
    if "pres_net_sentiment" not in result and {"pres_pos", "pres_neg"}.issubset(result.columns):
        result["pres_net_sentiment"] = result["pres_pos"] - result["pres_neg"]
    if "qa_net_sentiment" not in result and {"qa_pos", "qa_neg"}.issubset(result.columns):
        result["qa_net_sentiment"] = result["qa_pos"] - result["qa_neg"]
    if "sentiment_mismatch_pos" not in result and {"pres_pos", "qa_pos"}.issubset(result.columns):
        result["sentiment_mismatch_pos"] = result["pres_pos"] - result["qa_pos"]
    if "sentiment_mismatch_neg" not in result and {"pres_neg", "qa_neg"}.issubset(result.columns):
        result["sentiment_mismatch_neg"] = result["pres_neg"] - result["qa_neg"]
    if "evasion_index" not in result and {"pres_pos", "qa_neu"}.issubset(result.columns):
        result["evasion_index"] = result["pres_pos"] * result["qa_neu"]
    return result


def _drop_untrusted_targets(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove labels and any realized/future-return fields from a cache.

    Older rich-experiment caches contain several target aliases in addition
    to ``abnormal_return_5d``.  Treat the cache as untrusted input and remove
    all known aliases before the fresh event builder runs.
    """
    exact_targets = {
        "market_subtracted", "beta_market_subtracted", "label", "beta_label",
        "target", "y", "stock_return_5d", "market_return_5d", "return_5d",
    }
    target_like = [
        column for column in frame.columns
        if column in exact_targets
        or column.startswith(("abnormal_return_", "beta_abnormal_return_", "future_return_", "target_"))
        or column.endswith("_target")
    ]
    return frame.drop(columns=target_like, errors="ignore")


def _drop_stale_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove market/history columns from older caches before recomputation."""
    market_columns = {
        "momentum_5d", "momentum_20d", "volatility_20d", "market_momentum_20d", "beta_120d",
        "historical_score_source", "target_baseline_date", "target_end_date_5d", "target_price_basis",
    }
    history_columns = {
        column for column in frame.columns
        if column.endswith("_z") or column.endswith("_history_count")
    }
    return frame.drop(columns=market_columns | history_columns, errors="ignore")


def _build_language_frame(
    event_frame: pd.DataFrame,
    namespace: Mapping[str, object],
    lexicon_path: str | None,
    cache_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_used = False
    cache_manifest = {"cache_version": CACHE_VERSION, "cache_used": False}
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if not isinstance(cached, pd.DataFrame):
            raise ValueError(f"Cache {cache_path} must contain a pandas DataFrame.")
        cached = cached.loc[:, ~cached.columns.duplicated()].copy()
        cache_manifest["source_cache_version"] = cached.attrs.get("cache_version", "legacy-unversioned")
        # The cache is intentionally numeric/compact after sentence extraction.
        # Older caches may still contain pres_clean/qa_clean, but those large
        # text columns are not required to rebuild targets or model features.
        required = {"symbol", "call_datetime"}
        if required.issubset(cached.columns) and "pres_sent_mean" in cached.columns:
            language = _drop_stale_derived_features(_drop_untrusted_targets(cached))
            cache_used = True
            cache_manifest["cache_used"] = True
            print(f"Loaded language feature cache: {cache_path}")
        else:
            raise ValueError(f"Cache {cache_path} is missing required language feature columns.")
    else:
        language = event_frame.copy()
        extracted = language["structured_content"].map(extract_structural_text)
        language["pres_clean"] = extracted.map(lambda pair: pair[0])
        language["qa_clean"] = extracted.map(lambda pair: pair[1])
        language = language[(language["pres_clean"].str.len() > 100) & (language["qa_clean"].str.len() > 100)].copy()
        tokenizer = namespace.get("tokenizer")
        finbert = namespace.get("finbert")
        requested_device = namespace.get("device")
        if requested_device is None:
            try:
                import torch
                requested_device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                requested_device = "cpu"
        device = str(requested_device)
        if tokenizer is None or finbert is None:
            from transformers import BertForSequenceClassification, BertTokenizer
            tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
            finbert = BertForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
        print("Extracting sentence-level FinBERT features...")
        language = build_sentence_feature_frame(
            language,
            make_finbert_sentence_scorer(tokenizer, finbert, device=device),
            row_batch_size=16,
        )

    language = _add_probability_aliases(language)
    has_text = {"pres_clean", "qa_clean"}.issubset(language.columns)
    if lexicon_path and not any("_lm_" in column for column in language.columns) and has_text:
        print(f"Loading financial lexicon from {lexicon_path}...")
        language = add_dictionary_features(language, load_lm_lexicons(lexicon_path))
    # Compact caches already contain these transcript-derived proxies.  Do
    # not recompute them from missing text (which would silently create zeros).
    if "eps_mention_count" not in language.columns and has_text:
        language = add_earnings_language_features(language)
    language = _add_probability_aliases(language)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Do not keep raw or cleaned transcript text in the reusable cache or in
    # the returned frame.  Once FinBERT/dictionary features are computed, the
    # text serves no purpose for target reconstruction and can consume more
    # RAM than the numeric feature matrix by a large margin.
    text_columns = ["content", "structured_content", "pres_clean", "qa_clean"]
    cache_to_save = _drop_untrusted_targets(language).drop(columns=text_columns, errors="ignore")
    cache_to_save.attrs["cache_version"] = CACHE_VERSION
    cache_to_save.attrs["target_construction_version"] = TARGET_VERSION
    cache_to_save.to_pickle(cache_path)
    cache_manifest["language_rows"] = int(len(cache_to_save))
    cache_manifest["cache_version"] = CACHE_VERSION
    if cache_used:
        print("Revalidated and refreshed language feature cache.")
    else:
        print(f"Saved language feature cache: {cache_path}")
    language = language.drop(columns=text_columns, errors="ignore")
    return language, cache_manifest


def build_universe_frame(
    namespace: Mapping[str, object],
    sectors: Sequence[str] | None = ("Industrials",),
    lm_dictionary_path: str | None = None,
    cache_path: str | Path = "rich_experiment/feature_frame_ready.pkl",
    price_data: Mapping[str, pd.DataFrame] | None = None,
    benchmark: pd.Series | None = None,
) -> dict[str, object]:
    """Build corrected targets and model-ready features for a sector universe.

    Language features may be reused from a cache, but targets, market controls,
    and historical z-scores are always reconstructed for the requested
    universe. The returned frame is suitable for ``run_e9_experiment``.
    """
    cache_path = Path(cache_path)
    raw = namespace.get("df_bose")
    if not isinstance(raw, pd.DataFrame):
        raw = namespace.get("df")
    if cache_path.exists():
        sector_frame = pd.read_pickle(cache_path)
        if not isinstance(sector_frame, pd.DataFrame):
            raise ValueError(f"Cache {cache_path} must contain a pandas DataFrame.")
        sector_frame = sector_frame.copy()
        print(f"Using cached universe rows: {cache_path}")
    elif isinstance(raw, pd.DataFrame):
        sector_frame = _prepare_sector_frame(raw, sectors=sectors)
        print(f"Prepared universe rows: {len(sector_frame)}")
    else:
        raise RuntimeError("Provide df_bose/df or a language-feature cache.")

    sector_frame = _drop_stale_derived_features(_drop_untrusted_targets(sector_frame))
    sector_frame = sector_frame.loc[:, ~sector_frame.columns.duplicated()].copy()
    if "symbol" not in sector_frame or "call_datetime" not in sector_frame:
        if "date" in sector_frame:
            sector_frame["call_datetime"] = pd.to_datetime(sector_frame["date"], errors="coerce")
        else:
            raise KeyError("Universe frame must contain symbol and call_datetime/date columns.")
    if "gics_sector" not in sector_frame:
        sector_frame["gics_sector"] = sector_frame["symbol"].map(
            _sector_mapping(sector_frame["symbol"].dropna().unique())
        )
    sector_frame["call_datetime"] = pd.to_datetime(sector_frame["call_datetime"], errors="coerce")
    sector_frame = sector_frame.dropna(subset=["call_datetime", "symbol"]).copy()
    sector_frame["event_year"] = sector_frame["call_datetime"].dt.year.astype(int)
    sector_frame = sector_frame[sector_frame["event_year"] >= 2015].copy()
    if sectors is not None:
        sector_frame = sector_frame[sector_frame["gics_sector"].isin(set(sectors))].copy()
    if sector_frame.empty:
        raise RuntimeError(f"No rows remain for sectors={sectors!r}.")

    if price_data is None or benchmark is None:
        price_data, benchmark = _download_market_data(sector_frame["symbol"].unique())
    event_config = EventConfig(
        date_col="call_datetime",
        symbol_col="symbol",
        call_datetime_col="call_datetime",
        phase_col="call_phase",
        event_year_col="event_year",
        price_col="Adj Close",
        horizons=(1, 3, 5, 7, 30),
    )
    event_frame = build_event_dataset(sector_frame, price_data, benchmark, event_config)
    event_audit = dict(event_frame.attrs.get("event_audit", {}))
    if sectors is not None:
        event_frame = event_frame[event_frame["gics_sector"].isin(set(sectors))].copy()
    if event_frame.empty:
        raise RuntimeError(f"No valid target events remained for sectors={sectors!r}.")

    language_frame, cache_manifest = _build_language_frame(
        event_frame,
        namespace,
        lm_dictionary_path,
        cache_path,
    )
    key_columns = ["symbol", "call_datetime"]
    language_frame = language_frame.drop_duplicates(key_columns, keep="last")
    event_frame = event_frame.drop_duplicates(key_columns, keep="last")
    language_columns = [
        column for column in language_frame.columns
        if column not in {"abnormal_return_5d", "beta_abnormal_return_5d", "label", "beta_label"}
    ]
    event_columns = [column for column in event_frame.columns if column not in {"content", "structured_content"}]
    frame = event_frame[event_columns].merge(
        language_frame[key_columns + [column for column in language_columns if column not in event_frame.columns]],
        on=key_columns,
        how="left",
    )
    frame = add_precall_market_features(frame, price_data, benchmark, event_config)
    frame = add_beta_adjusted_targets(frame, price_data, benchmark, event_config)
    frame = _add_probability_aliases(frame)
    history_columns = [
        column for column in [
            "pres_sent_mean", "qa_sent_mean", "pres_entropy", "qa_entropy",
            "pres_neg_frac", "qa_neg_frac", "pres_slope", "qa_slope",
        ] if column in frame
    ]
    frame = frame.drop(
        columns=[column for column in frame.columns if column.endswith("_z") or column.endswith("_history_count")],
        errors="ignore",
    )
    if history_columns:
        frame = add_expanding_history_features(frame, history_columns)
    frame["label"] = (frame["abnormal_return_5d"] > 0).astype(int)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    blocks = build_final_feature_blocks(frame)
    non_union = [set(values) for name, values in blocks.items() if name != "all_features"]
    overlaps = set().union(*(
        non_union[i] & non_union[j]
        for i in range(len(non_union))
        for j in range(i + 1, len(non_union))
    )) if non_union else set()
    if overlaps:
        raise RuntimeError(f"Feature blocks are not disjoint: {sorted(overlaps)}")
    target_like = {
        column for column in frame.columns
        if column.startswith(("abnormal_return_", "beta_abnormal_return_", "future_return_", "target_"))
        or column in {"label", "beta_label", "market_subtracted", "beta_market_subtracted"}
    }
    leaked = target_like.intersection(set(blocks["all_features"]))
    if leaked:
        raise RuntimeError(f"Target-like columns entered model features: {sorted(leaked)}")
    cache_manifest.update({
        "target_rows": int(len(frame)),
        "language_feature_rows": int((frame["pres_sent_mean"].notna() & frame["qa_sent_mean"].notna()).sum()) if {"pres_sent_mean", "qa_sent_mean"}.issubset(frame.columns) else 0,
        "language_missing_rows": int(len(frame) - (frame["pres_sent_mean"].notna() & frame["qa_sent_mean"].notna()).sum()) if {"pres_sent_mean", "qa_sent_mean"}.issubset(frame.columns) else int(len(frame)),
        "sector_scope": list(sectors) if sectors is not None else "all_mapped_sectors",
    })
    target_audit = {
        "target_version": TARGET_VERSION,
        "price_basis": "Adj Close for stock and benchmark",
        "event_year_source": "call_datetime",
        "sector_scope": list(sectors) if sectors is not None else "all_mapped_sectors",
        "rows": int(len(frame)),
        "companies": int(frame["symbol"].nunique()),
        "positive_rate": float(frame["label"].mean()),
        "call_phases": frame["call_phase"].value_counts().to_dict(),
        "event_audit": event_audit,
    }
    return {
        "frame": frame,
        "blocks": blocks,
        "target_audit": target_audit,
        "cache_manifest": cache_manifest,
        "price_data": price_data,
        "benchmark": benchmark,
    }


def build_universe_audit(
    transcripts: pd.DataFrame,
    price_data: Mapping[str, pd.DataFrame],
    benchmark: pd.Series,
    event_config: EventConfig | None = None,
) -> pd.DataFrame:
    """Audit mapped sectors and corrected event availability before FinBERT."""
    config = event_config or EventConfig(
        date_col="call_datetime",
        call_datetime_col="call_datetime",
        price_col="Adj Close",
        horizons=(1, 3, 5, 7, 30),
    )
    if {"symbol", "gics_sector"}.issubset(transcripts.columns) and ({"call_datetime", "date"} & set(transcripts.columns)):
        prepared = transcripts.copy()
        date_column = "call_datetime" if "call_datetime" in prepared else "date"
        prepared["call_datetime"] = pd.to_datetime(prepared[date_column], errors="coerce")
        prepared["event_year"] = prepared["call_datetime"].dt.year
        prepared = prepared.dropna(subset=["call_datetime", "gics_sector", "symbol"])
        prepared = prepared[prepared["event_year"] >= 2015].copy()
    else:
        prepared = _prepare_sector_frame(transcripts, sectors=None)
    rows = []
    for sector, group in prepared.groupby("gics_sector", dropna=False, sort=True):
        events = build_event_dataset(group, price_data, benchmark, config)
        audit = dict(events.attrs.get("event_audit", {}))
        complete = np.nan
        if {"pres_sent_mean", "qa_sent_mean"}.issubset(group.columns):
            complete = int((group["pres_sent_mean"].notna() & group["qa_sent_mean"].notna()).sum())
        complete_events = events
        if {"pres_sent_mean", "qa_sent_mean"}.issubset(events.columns):
            complete_events = events[events["pres_sent_mean"].notna() & events["qa_sent_mean"].notna()]
        else:
            complete_events = events.iloc[0:0]
        rows.append({
            "sector": sector,
            "raw_transcript_count": int(len(group)),
            "valid_timestamp_count": int(group[config.call_datetime_col].notna().sum()),
            "companies": int(group[config.symbol_col].nunique()),
            "year_min": int(group["event_year"].min()) if not group.empty else np.nan,
            "year_max": int(group["event_year"].max()) if not group.empty else np.nan,
            "candidate_event_count": int(len(group)),
            "intraday_excluded": int(audit.get("intraday_excluded", 0)),
            "missing_price_exclusions": int(audit.get("missing_price_data", 0) + audit.get("invalid_ticker", 0) + audit.get("missing_baseline", 0) + audit.get("missing_future_target", 0)),
            "valid_target_events": int(len(events)),
            "complete_language_events": complete,
            "complete_case_positive_rate": float(complete_events["abnormal_return_5d"].gt(0).mean()) if not complete_events.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("valid_target_events", ascending=False).reset_index(drop=True)


def _numeric(columns: Sequence[str], frame: pd.DataFrame) -> list[str]:
    return [column for column in dict.fromkeys(columns) if column in frame and pd.api.types.is_numeric_dtype(frame[column])]


def build_final_feature_blocks(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return disjoint, named feature blocks with no target-like columns."""
    baseline = _numeric([
        "pres_pos", "pres_neg", "pres_neu", "qa_pos", "qa_neg", "qa_neu",
        "pres_net_sentiment", "qa_net_sentiment", "sentiment_mismatch_pos",
        "sentiment_mismatch_neg", "evasion_index",
    ], frame)
    sentence_names = [
        "sent_mean", "sent_std", "sent_p10", "sent_p90", "pos_mean", "neg_mean",
        "neutral_mean", "pos_frac", "neg_frac", "entropy", "begin_mean",
        "middle_mean", "end_mean", "slope", "n_sentences",
    ]
    sentence = _numeric(
        [f"{section}_{name}" for section in ("pres", "qa") for name in sentence_names]
        + [f"qa_minus_pres_{name}" for name in ("sent_mean", "sent_std", "entropy", "neg_frac", "pos_frac", "slope")],
        frame,
    )
    dictionary = _numeric([column for column in frame.columns if "_lm_" in column], frame)
    historical = _numeric([column for column in frame.columns if column.endswith("_z") or column.endswith("_history_count")], frame)
    earnings = _numeric([
        "eps_mention_count", "eps_mention_rate", "beat_language_count", "miss_language_count",
        "above_expectations_count", "below_expectations_count", "guidance_up_count",
        "guidance_down_count", "guidance_maintained_count", "forward_language_count",
    ], frame)
    market = _numeric(["momentum_5d", "momentum_20d", "volatility_20d", "market_momentum_20d", "beta_120d"], frame)
    blocks = {
        "market_only": market,
        "baseline_sentiment": baseline,
        "sentence_sentiment": sentence,
        "financial_dictionary": dictionary,
        "historical_surprise": historical,
        "earnings_language_proxy": earnings,
    }
    blocks["all_features"] = list(dict.fromkeys(sum(blocks.values(), [])))
    return blocks


def _calibration_mae(y: pd.Series, probability: np.ndarray, bins: int = 5) -> float:
    values = pd.DataFrame({"y": np.asarray(y), "p": probability})
    values["bin"] = pd.qcut(values["p"], q=min(bins, len(values)), duplicates="drop")
    grouped = values.groupby("bin", observed=False)
    return float((grouped["y"].mean() - grouped["p"].mean()).abs().mean())


def _fit_spec(name: str, train: pd.DataFrame, features: Sequence[str], spec: dict, evaluation: EvaluationConfig):
    config = spec.get("config", ModelConfig())
    if spec["kind"] == "logistic":
        return fit_logistic_model(train, features, model_config=config, evaluation_config=evaluation)
    if spec["kind"] == "elastic_net":
        return fit_elastic_net_model(
            train, features, model_config=config, evaluation_config=evaluation,
            C=spec["C"], l1_ratio=spec["l1_ratio"],
        )
    if spec["kind"] == "xgboost":
        return fit_xgb_model(train, features, model_config=config, evaluation_config=evaluation)
    raise ValueError(f"Unknown model specification: {spec}")


def _spec_name(spec: dict) -> str:
    if spec["kind"] != "elastic_net":
        return spec["name"]
    return f"elastic_net_C{spec['C']}_l1{spec['l1_ratio']}"


def _evaluate(
    frame: pd.DataFrame,
    blocks: Mapping[str, Sequence[str]],
    specs: Sequence[dict],
    target_col: str,
    evaluation: EvaluationConfig,
    cutoff_year: int = 2023,
    bootstrap_repetitions: int = 250,
    holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = ("final_holdout",) if holdout else evaluation.walk_forward_years
    summary_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    for block_name, block_features in blocks.items():
        features = [column for column in block_features if column in frame.columns]
        for spec in specs:
            model_name = _spec_name(spec)
            print(f"Evaluating {target_col} | {block_name} | {model_name} | {'holdout' if holdout else 'walk-forward'}")
            for year in years:
                if holdout:
                    train = frame[frame["event_year"] < cutoff_year].dropna(subset=[target_col]).copy()
                    test = frame[frame["event_year"] >= cutoff_year].dropna(subset=[target_col]).copy()
                    split = "final_holdout"
                else:
                    train = frame[frame["event_year"] < year].dropna(subset=[target_col]).copy()
                    test = frame[frame["event_year"] == year].dropna(subset=[target_col]).copy()
                    split = f"walk_forward_{year}"
                if train.empty or test.empty or train[target_col].nunique() < 2 or not features:
                    continue
                model = _fit_spec(model_name, train, features, spec, evaluation)
                selected = getattr(model, "selected_features_", features)
                probability = model.predict_proba(test[selected])[:, 1]
                prediction_columns = [column for column in ["symbol", "call_datetime", "quarter", "event_year", target_col] if column in test]
                prediction = test[prediction_columns].copy().rename(columns={target_col: "y"})
                prediction["return"] = test.get("abnormal_return_5d", np.nan).to_numpy()
                prediction["probability"] = probability
                prediction["feature_block"] = block_name
                prediction["model"] = model_name
                prediction["split"] = split
                prediction_frames.append(prediction)
                metrics = classification_metrics(prediction["y"], probability)
                metrics.update({"feature_block": block_name, "model": model_name, "split": split, "target": target_col})
                metrics.update(top_bottom_spread(prediction))
                metrics["calibration_mae"] = _calibration_mae(prediction["y"], probability)
                summary_rows.append(metrics)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    if not predictions.empty:
        for (block_name, model_name), group in predictions.groupby(["feature_block", "model"]):
            metrics = classification_metrics(group["y"], group["probability"])
            fold_metrics = pd.DataFrame(summary_rows)
            fold_metrics = fold_metrics[(fold_metrics["feature_block"] == block_name) & (fold_metrics["model"] == model_name) & (fold_metrics["split"] != "final_holdout")]
            metrics.update({"feature_block": block_name, "model": model_name, "split": "final_holdout" if holdout else "walk_forward_aggregate", "target": target_col})
            metrics.update(top_bottom_spread(group))
            metrics["calibration_mae"] = _calibration_mae(group["y"], group["probability"])
            metrics["mean_fold_auc"] = float(fold_metrics["auc"].mean()) if not fold_metrics.empty else np.nan
            metrics["mean_fold_log_loss"] = float(fold_metrics["log_loss"].mean()) if not fold_metrics.empty else np.nan
            metrics["fold_count"] = int(len(fold_metrics))
            if bootstrap_repetitions and not holdout:
                lower, upper = cluster_bootstrap(group, "auc", repetitions=bootstrap_repetitions, random_state=evaluation.random_state)
                metrics.update({"auc_lower_95": lower, "auc_upper_95": upper})
            summary_rows.append(metrics)
    return pd.DataFrame(summary_rows), predictions


def _elastic_grid() -> list[dict]:
    return [
        {"name": "elastic_net", "kind": "elastic_net", "C": C, "l1_ratio": ratio, "config": ModelConfig()}
        for C in (0.01, 0.1, 1.0, 10.0)
        for ratio in (0.1, 0.5, 0.9)
    ]


def _baseline_summary(
    frame: pd.DataFrame,
    target_col: str,
    evaluation: EvaluationConfig,
    cutoff_year: int = 2023,
    holdout: bool = False,
) -> pd.DataFrame:
    """Report train-rate baselines without fitting a predictive model."""
    rows: list[dict] = []
    years = ("final_holdout",) if holdout else evaluation.walk_forward_years
    for year in years:
        if holdout:
            train = frame[frame["event_year"] < cutoff_year].dropna(subset=[target_col])
            test = frame[frame["event_year"] >= cutoff_year].dropna(subset=[target_col])
            split = "final_holdout"
        else:
            train = frame[frame["event_year"] < year].dropna(subset=[target_col])
            test = frame[frame["event_year"] == year].dropna(subset=[target_col])
            split = f"walk_forward_{year}"
        if train.empty or test.empty:
            continue
        train_rate = float(train[target_col].mean())
        for model_name, probability_value in (
            ("train_rate_baseline", train_rate),
            ("majority_baseline", float(train_rate >= 0.5)),
        ):
            probability = np.full(len(test), probability_value)
            metrics = classification_metrics(test[target_col], probability)
            metrics.update({"feature_block": "baseline", "model": model_name, "split": split, "target": target_col})
            rows.append(metrics)
    return pd.DataFrame(rows)


def run_final_experiment(
    namespace: Mapping[str, object],
    lm_dictionary_path: str | None = None,
    cache_path: str | Path = "rich_experiment/feature_frame_ready.pkl",
    output_dir: str | Path = "final_artifacts",
    include_xgboost: bool = True,
) -> dict:
    """Run the final target/feature/model/artifact pipeline."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_path)
    raw = namespace.get("df_bose")
    if not isinstance(raw, pd.DataFrame):
        raw = namespace.get("df")
    cache_exists = cache_path.exists()
    if cache_exists:
        cached_preview = pd.read_pickle(cache_path)
        sector_frame = cached_preview.copy()
        print(f"Using cached transcript rows: {cache_path}")
    elif isinstance(raw, pd.DataFrame):
        sector_frame = _prepare_sector_frame(raw)
        print(f"Prepared sector frame: {len(sector_frame)} rows")
    else:
        raise RuntimeError("Provide df_bose/df in the notebook or upload a valid language feature cache.")

    # Cached labels and returns are never trusted. They are removed before the
    # event builder reconstructs targets from fresh adjusted-price data.
    sector_frame = _drop_stale_derived_features(_drop_untrusted_targets(sector_frame))
    if "gics_sector" not in sector_frame:
        sector_frame["gics_sector"] = sector_frame["symbol"].map(
            _sector_mapping(sector_frame["symbol"].dropna().unique())
        )
    sector_frame["call_datetime"] = pd.to_datetime(sector_frame.get("call_datetime", sector_frame.get("date")), errors="coerce")
    sector_frame = sector_frame.dropna(subset=["call_datetime", "symbol"]).copy()
    price_data, benchmark = _download_market_data(sector_frame["symbol"].unique())
    date_column = "call_datetime" if "call_datetime" in sector_frame else "date"
    event_config = EventConfig(
        date_col=date_column,
        symbol_col="symbol",
        call_datetime_col="call_datetime",
        phase_col="call_phase",
        event_year_col="event_year",
        price_col="Adj Close",
        horizons=(1, 3, 5, 7, 30),
    )
    event_frame = build_event_dataset(sector_frame, price_data, benchmark, event_config)
    event_audit = dict(event_frame.attrs.get("event_audit", {}))
    event_frame = event_frame[event_frame["gics_sector"].eq("Industrials") & event_frame["event_year"].ge(2015)].copy()
    if event_frame.empty:
        raise RuntimeError("No valid Industrials events remained after corrected target construction.")

    language_frame, cache_manifest = _build_language_frame(event_frame, namespace, lm_dictionary_path, cache_path)
    key_columns = ["symbol", "call_datetime"]
    feature_columns = [column for column in language_frame.columns if column not in {"abnormal_return_5d", "beta_abnormal_return_5d", "label", "beta_label"}]
    language_frame = language_frame.drop_duplicates(key_columns, keep="last")
    event_frame = event_frame.drop_duplicates(key_columns, keep="last")
    event_columns = [
        column for column in event_frame.columns
        if column not in {"content", "structured_content"}
    ]
    frame = event_frame[event_columns].merge(
        language_frame[key_columns + [column for column in feature_columns if column not in event_frame.columns]],
        on=key_columns,
        how="left",
    )
    frame = add_precall_market_features(frame, price_data, benchmark, event_config)
    frame = add_beta_adjusted_targets(frame, price_data, benchmark, event_config)
    frame = _add_probability_aliases(frame)
    history_columns = [column for column in ["pres_sent_mean", "qa_sent_mean", "pres_entropy", "qa_entropy", "pres_neg_frac", "qa_neg_frac", "pres_slope", "qa_slope"] if column in frame]
    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_z") or column.endswith("_history_count")], errors="ignore")
    if history_columns:
        frame = add_expanding_history_features(frame, history_columns)
    blocks = build_final_feature_blocks(frame)
    non_union_blocks = [set(values) for name, values in blocks.items() if name != "all_features"]
    overlaps = set().union(*(non_union_blocks[i] & non_union_blocks[j] for i in range(len(non_union_blocks)) for j in range(i + 1, len(non_union_blocks)))) if non_union_blocks else set()
    if overlaps:
        raise RuntimeError(f"Feature blocks are not disjoint: {sorted(overlaps)}")
    target_names = {
        column for column in frame.columns
        if column.startswith(("abnormal_return_", "beta_abnormal_return_", "future_return_", "target_"))
        or column in {"label", "beta_label", "market_subtracted", "beta_market_subtracted"}
    }
    leaked = target_names.intersection(set(blocks["all_features"]))
    if leaked:
        raise RuntimeError(f"Target-like columns entered model features: {sorted(leaked)}")
    frame["label"] = (frame["abnormal_return_5d"] > 0).astype(int)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    cache_manifest["target_rows"] = int(len(frame))
    cache_manifest["language_feature_rows"] = int(frame["pres_sent_mean"].notna().sum()) if "pres_sent_mean" in frame else 0
    cache_manifest["language_missing_rows"] = int(frame["pres_sent_mean"].isna().sum()) if "pres_sent_mean" in frame else int(len(frame))

    evaluation = EvaluationConfig(bootstrap_repetitions=250, random_state=42)
    fixed_specs = [
        {"name": "logistic", "kind": "logistic", "config": ModelConfig()},
    ]
    tuning_frame = frame.copy()
    tuning_summary, _ = _evaluate(tuning_frame, {"all_features": blocks["all_features"]}, _elastic_grid(), "label", evaluation, bootstrap_repetitions=0)
    tuning_aggregate = tuning_summary[tuning_summary["split"] == "walk_forward_aggregate"].copy()
    if tuning_aggregate.empty:
        raise RuntimeError("Elastic Net tuning produced no valid walk-forward folds.")
    best_tuning = tuning_aggregate.sort_values(["mean_fold_auc", "mean_fold_log_loss"], ascending=[False, True]).iloc[0]
    best_C = float(best_tuning["model"].split("_C", 1)[1].split("_l1", 1)[0])
    best_l1 = float(best_tuning["model"].rsplit("l1", 1)[1])
    fixed_specs.append({"name": "elastic_net", "kind": "elastic_net", "C": best_C, "l1_ratio": best_l1, "config": ModelConfig()})
    if include_xgboost:
        fixed_specs.append({"name": "xgboost", "kind": "xgboost", "config": ModelConfig(xgb_depth=2, xgb_estimators=150)})

    # Small intervals keep the model-search pass tractable.  The selected
    # primary model receives the full clustered bootstrap below.
    search_bootstrap_repetitions = 50
    walk_summary, walk_predictions = _evaluate(
        frame, blocks, fixed_specs, "label", evaluation,
        bootstrap_repetitions=search_bootstrap_repetitions,
    )
    holdout_summary, holdout_predictions = _evaluate(frame, blocks, fixed_specs, "label", evaluation, holdout=True, bootstrap_repetitions=0)
    beta_frame = frame.dropna(subset=["beta_abnormal_return_5d"]).copy()
    beta_frame["beta_label"] = (beta_frame["beta_abnormal_return_5d"] > 0).astype(int)
    beta_summary, beta_predictions = _evaluate(
        beta_frame, blocks, fixed_specs, "beta_label", evaluation,
        bootstrap_repetitions=search_bootstrap_repetitions,
    )
    beta_holdout_summary, beta_holdout_predictions = _evaluate(
        beta_frame, blocks, fixed_specs, "beta_label", evaluation,
        holdout=True, bootstrap_repetitions=0,
    )
    baseline_walk = _baseline_summary(frame, "label", evaluation)
    baseline_holdout = _baseline_summary(frame, "label", evaluation, holdout=True)
    beta_baseline_walk = _baseline_summary(beta_frame, "beta_label", evaluation)
    beta_baseline_holdout = _baseline_summary(beta_frame, "beta_label", evaluation, holdout=True)
    summary = pd.concat([
        tuning_summary.assign(target="primary_tuning"), baseline_walk, baseline_holdout,
        walk_summary, holdout_summary, beta_baseline_walk, beta_baseline_holdout,
        beta_summary, beta_holdout_summary,
    ], ignore_index=True)
    predictions = pd.concat([walk_predictions, holdout_predictions, beta_predictions, beta_holdout_predictions], ignore_index=True)

    candidates = walk_summary[walk_summary["split"] == "walk_forward_aggregate"].copy()
    winner = candidates.sort_values(["mean_fold_auc", "mean_fold_log_loss"], ascending=[False, True]).iloc[0]
    winner_block, winner_name = winner["feature_block"], winner["model"]
    winner_predictions = walk_predictions[
        (walk_predictions["feature_block"] == winner_block)
        & (walk_predictions["model"] == winner_name)
    ]
    if not winner_predictions.empty:
        full_lower, full_upper = cluster_bootstrap(
            winner_predictions, "auc", repetitions=1000, random_state=evaluation.random_state
        )
        winner_mask = (
            (walk_summary["feature_block"] == winner_block)
            & (walk_summary["model"] == winner_name)
            & (walk_summary["split"] == "walk_forward_aggregate")
        )
        walk_summary.loc[winner_mask, "auc_lower_95"] = full_lower
        walk_summary.loc[winner_mask, "auc_upper_95"] = full_upper
        summary_mask = (
            (summary["feature_block"] == winner_block)
            & (summary["model"] == winner_name)
            & (summary["split"] == "walk_forward_aggregate")
            & (summary["target"] == "label")
        )
        summary.loc[summary_mask, "auc_lower_95"] = full_lower
        summary.loc[summary_mask, "auc_upper_95"] = full_upper
    winner_spec = next(spec for spec in fixed_specs if _spec_name(spec) == winner_name)
    development = frame[frame["event_year"] < 2023].dropna(subset=["label"])
    winner_model = _fit_spec(winner_name, development, blocks[winner_block], winner_spec, evaluation)
    selected_features = getattr(winner_model, "selected_features_", blocks[winner_block])
    winner_holdout_predictions = holdout_predictions[
        (holdout_predictions["feature_block"] == winner_block)
        & (holdout_predictions["model"] == winner_name)
    ].copy()

    target_audit = {
        "target_version": TARGET_VERSION,
        "price_basis": "Adj Close for stock and benchmark",
        "event_year_source": "call_datetime",
        "timestamp_assumption": "timezone-naive source timestamps are interpreted using the source wall-clock time; event phase thresholds are America/New_York market hours",
        "sector_scope": "Industrials",
        "event_year_min": int(frame["event_year"].min()),
        "event_year_max": int(frame["event_year"].max()),
        "rows": int(len(frame)),
        "language_feature_rows": int(frame["pres_sent_mean"].notna().sum()) if "pres_sent_mean" in frame else 0,
        "language_missing_rows": int(frame["pres_sent_mean"].isna().sum()) if "pres_sent_mean" in frame else int(len(frame)),
        "companies": int(frame["symbol"].nunique()),
        "call_phases": frame["call_phase"].value_counts().to_dict(),
        "positive_rate": float(frame["label"].mean()),
        "event_audit": event_audit,
    }
    metadata = {
        "model_version": "earnings-intelligence-final-v1",
        "feature_block": winner_block,
        "model": winner_name,
        "selection_rule": "highest mean walk-forward AUC, then lowest mean walk-forward log loss; primary target only",
        "target": TARGET_VERSION,
        "target_construction_version": TARGET_VERSION,
        "cache_version": CACHE_VERSION,
        "event_rows": int(len(frame)),
        "language_feature_rows": int(frame["pres_sent_mean"].notna().sum()) if "pres_sent_mean" in frame else 0,
        "language_missing_rows": int(frame["pres_sent_mean"].isna().sum()) if "pres_sent_mean" in frame else int(len(frame)),
        "elastic_net_C": best_C,
        "elastic_net_l1_ratio": best_l1,
    }
    artifact_columns = list(dict.fromkeys([
        "symbol", "company_name", "quarter", "year", "event_year", "call_datetime", "call_phase", "gics_sector",
        "abnormal_return_5d", "beta_abnormal_return_5d",
    ] + sum(blocks.values(), [])))
    artifact_frame = frame[[column for column in artifact_columns if column in frame.columns]].copy()
    summary.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    artifact_frame.to_csv(output / "feature_table.csv", index=False)
    (output / "feature_blocks.json").write_text(json.dumps(blocks, indent=2))
    (output / "target_audit.json").write_text(json.dumps(target_audit, indent=2, default=str))
    (output / "run_manifest.json").write_text(json.dumps({**metadata, **cache_manifest}, indent=2, default=str))
    audit_rows = [{"check": key, "value": value} for key, value in event_audit.items()]
    audit_rows.extend([
        {"check": "target_version", "value": TARGET_VERSION},
        {"check": "price_basis", "value": "Adj Close for stock and benchmark"},
        {"check": "event_year_source", "value": "call_datetime"},
        {"check": "timestamp_assumption", "value": "timezone-naive source timestamps use source wall-clock time; phase thresholds use America/New_York market hours"},
        {"check": "final_rows", "value": len(frame)},
        {"check": "language_feature_rows", "value": int(frame["pres_sent_mean"].notna().sum()) if "pres_sent_mean" in frame else 0},
        {"check": "language_missing_rows", "value": int(frame["pres_sent_mean"].isna().sum()) if "pres_sent_mean" in frame else int(len(frame))},
        {"check": "final_companies", "value": frame["symbol"].nunique()},
        {"check": "final_positive_rate", "value": frame["label"].mean()},
        {"check": "call_phases", "value": frame["call_phase"].value_counts().to_dict()},
    ])
    pd.DataFrame(audit_rows).to_csv(output / "target_audit.csv", index=False)
    # ``output/artifacts`` is the standalone dashboard bundle.  Keep the
    # audit and manifest beside the dashboard-required files so the bundle is
    # complete when downloaded independently of the experiment directory.
    bundle_dir = save_artifact_bundle(
        output / "artifacts", winner_model, selected_features, metadata,
        feature_frame=artifact_frame,
        predictions=winner_holdout_predictions,
        metrics=summary,
    )
    for filename in ("feature_blocks.json", "target_audit.json", "target_audit.csv", "run_manifest.json"):
        shutil.copy2(output / filename, bundle_dir / filename)
    print(f"Winner: {winner_name} / {winner_block}")
    print(f"Saved final outputs to {output.resolve()}")
    return {"frame": frame, "blocks": blocks, "summary": summary, "predictions": predictions, "winner": metadata}


def run_complete_case_experiment(
    namespace: Mapping[str, object],
    output_dir: str | Path = "complete_case_artifacts",
    include_xgboost: bool = True,
) -> dict:
    """Re-evaluate language models only where language features are present.

    This diagnostic keeps the corrected targets fixed and removes rows with
    unavailable presentation or Q&A sentiment. Every block is evaluated on
    the same complete-case subset, making it directly comparable with the
    earlier roughly 883-event experiment.
    """
    source = namespace.get("frame")
    if not isinstance(source, pd.DataFrame):
        final_results = namespace.get("final_results")
        if isinstance(final_results, Mapping) and isinstance(final_results.get("frame"), pd.DataFrame):
            source = final_results["frame"]
    if not isinstance(source, pd.DataFrame):
        raise RuntimeError("Provide final_results['frame'] or a DataFrame under namespace['frame'].")

    original = source.copy().loc[:, ~source.columns.duplicated()]
    required = {"pres_sent_mean", "qa_sent_mean", "event_year", "abnormal_return_5d"}
    missing = required.difference(original.columns)
    if missing:
        raise KeyError(f"Complete-case experiment is missing columns: {sorted(missing)}")
    complete_mask = original["pres_sent_mean"].notna() & original["qa_sent_mean"].notna()
    frame = original.loc[complete_mask].copy()
    if frame.empty:
        raise RuntimeError("No complete language-feature rows are available.")
    frame["label"] = (frame["abnormal_return_5d"] > 0).astype(int)

    history_columns = [
        column for column in
        ["pres_sent_mean", "qa_sent_mean", "pres_entropy", "qa_entropy", "pres_neg_frac", "qa_neg_frac", "pres_slope", "qa_slope"]
        if column in frame
    ]
    frame = frame.drop(
        columns=[column for column in frame.columns if column.endswith("_z") or column.endswith("_history_count")],
        errors="ignore",
    )
    if history_columns:
        frame = add_expanding_history_features(frame, history_columns)
    blocks = build_final_feature_blocks(frame)
    non_union_blocks = [set(values) for name, values in blocks.items() if name != "all_features"]
    overlaps = set().union(
        *(non_union_blocks[i] & non_union_blocks[j]
          for i in range(len(non_union_blocks))
          for j in range(i + 1, len(non_union_blocks)))
    ) if non_union_blocks else set()
    if overlaps:
        raise RuntimeError(f"Complete-case feature blocks overlap: {sorted(overlaps)}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation = EvaluationConfig(bootstrap_repetitions=250, random_state=42)
    tuning_summary, _ = _evaluate(
        frame, {"all_features": blocks["all_features"]}, _elastic_grid(), "label", evaluation,
        bootstrap_repetitions=0,
    )
    tuning_aggregate = tuning_summary[tuning_summary["split"] == "walk_forward_aggregate"]
    if tuning_aggregate.empty:
        raise RuntimeError("Complete-case Elastic Net tuning produced no walk-forward folds.")
    best_tuning = tuning_aggregate.sort_values(
        ["mean_fold_auc", "mean_fold_log_loss"], ascending=[False, True]
    ).iloc[0]
    best_C = float(best_tuning["model"].split("_C", 1)[1].split("_l1", 1)[0])
    best_l1 = float(best_tuning["model"].rsplit("l1", 1)[1])
    fixed_specs = [
        {"name": "logistic", "kind": "logistic", "config": ModelConfig()},
        {"name": "elastic_net", "kind": "elastic_net", "C": best_C, "l1_ratio": best_l1, "config": ModelConfig()},
    ]
    if include_xgboost:
        fixed_specs.append({"name": "xgboost", "kind": "xgboost", "config": ModelConfig(xgb_depth=2, xgb_estimators=150)})

    walk_summary, walk_predictions = _evaluate(
        frame, blocks, fixed_specs, "label", evaluation, bootstrap_repetitions=50,
    )
    holdout_summary, holdout_predictions = _evaluate(
        frame, blocks, fixed_specs, "label", evaluation, holdout=True, bootstrap_repetitions=0,
    )
    beta_frame = frame.dropna(subset=["beta_abnormal_return_5d"]).copy()
    beta_frame["beta_label"] = (beta_frame["beta_abnormal_return_5d"] > 0).astype(int)
    beta_summary, beta_predictions = _evaluate(
        beta_frame, blocks, fixed_specs, "beta_label", evaluation, bootstrap_repetitions=50,
    )
    beta_holdout_summary, beta_holdout_predictions = _evaluate(
        beta_frame, blocks, fixed_specs, "beta_label", evaluation, holdout=True, bootstrap_repetitions=0,
    )
    summary = pd.concat([
        tuning_summary.assign(target="primary_tuning"),
        _baseline_summary(frame, "label", evaluation),
        _baseline_summary(frame, "label", evaluation, holdout=True),
        walk_summary, holdout_summary,
        _baseline_summary(beta_frame, "beta_label", evaluation),
        _baseline_summary(beta_frame, "beta_label", evaluation, holdout=True),
        beta_summary, beta_holdout_summary,
    ], ignore_index=True)
    predictions = pd.concat([
        walk_predictions, holdout_predictions, beta_predictions, beta_holdout_predictions,
    ], ignore_index=True)

    candidates = walk_summary[walk_summary["split"] == "walk_forward_aggregate"]
    winner = candidates.sort_values(
        ["mean_fold_auc", "mean_fold_log_loss"], ascending=[False, True]
    ).iloc[0]
    winner_block, winner_name = winner["feature_block"], winner["model"]
    winner_predictions = walk_predictions[
        (walk_predictions["feature_block"] == winner_block)
        & (walk_predictions["model"] == winner_name)
    ]
    if not winner_predictions.empty:
        lower, upper = cluster_bootstrap(
            winner_predictions, "auc", repetitions=1000, random_state=evaluation.random_state
        )
        summary_mask = (
            (summary["feature_block"] == winner_block)
            & (summary["model"] == winner_name)
            & (summary["split"] == "walk_forward_aggregate")
            & (summary["target"] == "label")
        )
        summary.loc[summary_mask, "auc_lower_95"] = lower
        summary.loc[summary_mask, "auc_upper_95"] = upper

    winner_spec = next(spec for spec in fixed_specs if _spec_name(spec) == winner_name)
    development = frame[frame["event_year"] < 2023].dropna(subset=["label"])
    winner_model = _fit_spec(winner_name, development, blocks[winner_block], winner_spec, evaluation)
    selected_features = getattr(winner_model, "selected_features_", blocks[winner_block])
    winner_holdout_predictions = holdout_predictions[
        (holdout_predictions["feature_block"] == winner_block)
        & (holdout_predictions["model"] == winner_name)
    ].copy()

    metadata = {
        "model_version": "earnings-intelligence-complete-case-v1",
        "feature_block": winner_block,
        "model": winner_name,
        "selection_rule": "highest mean walk-forward AUC, then lowest mean walk-forward log loss; primary target only",
        "target": TARGET_VERSION,
        "target_construction_version": TARGET_VERSION,
        "source_rows": int(len(original)),
        "complete_language_rows": int(len(frame)),
        "complete_language_rule": "pres_sent_mean and qa_sent_mean both non-missing",
        "elastic_net_C": best_C,
        "elastic_net_l1_ratio": best_l1,
    }
    artifact_columns = list(dict.fromkeys([
        "symbol", "company_name", "quarter", "year", "event_year", "call_datetime", "call_phase", "gics_sector",
        "abnormal_return_5d", "beta_abnormal_return_5d",
    ] + sum(blocks.values(), [])))
    artifact_frame = frame[[column for column in artifact_columns if column in frame.columns]].copy()
    target_audit = {
        **metadata,
        "event_year_min": int(frame["event_year"].min()),
        "event_year_max": int(frame["event_year"].max()),
        "companies": int(frame["symbol"].nunique()),
        "positive_rate": float(frame["label"].mean()),
        "call_phases": frame["call_phase"].value_counts().to_dict(),
    }
    summary.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    artifact_frame.to_csv(output / "feature_table.csv", index=False)
    (output / "feature_blocks.json").write_text(json.dumps(blocks, indent=2))
    (output / "target_audit.json").write_text(json.dumps(target_audit, indent=2, default=str))
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2, default=str))
    pd.DataFrame([
        {"check": "source_rows", "value": len(original)},
        {"check": "complete_language_rows", "value": len(frame)},
        {"check": "language_rows_removed", "value": int(len(original) - len(frame))},
        {"check": "complete_language_rule", "value": metadata["complete_language_rule"]},
        {"check": "target_version", "value": TARGET_VERSION},
    ]).to_csv(output / "target_audit.csv", index=False)
    bundle_dir = save_artifact_bundle(
        output / "artifacts", winner_model, selected_features, metadata,
        feature_frame=artifact_frame,
        predictions=winner_holdout_predictions,
        metrics=summary,
    )
    for filename in ("feature_blocks.json", "target_audit.json", "target_audit.csv", "run_manifest.json"):
        shutil.copy2(output / filename, bundle_dir / filename)
    print(f"Complete-case rows: {len(frame)} / {len(original)}")
    print(f"Complete-case winner: {winner_name} / {winner_block}")
    print(f"Saved complete-case outputs to {output.resolve()}")
    return {"frame": frame, "blocks": blocks, "summary": summary, "predictions": predictions, "winner": metadata}
