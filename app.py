"""Earnings Call Intelligence: a newcomer-first research workspace."""

from __future__ import annotations

import html
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from artifact_contract import (
    ArtifactBundle,
    ArtifactValidationError,
    discover_artifact_dirs,
    load_artifact_bundle,
)


st.set_page_config(page_title="Earnings Call Intelligence", page_icon="◈", layout="wide")

VALIDATED_STATUSES = {"Out-of-sample holdout", "Walk-forward validated"}
STATUS_FILTERS = (
    "Validated calls",
    "Exploratory archive",
    "Unavailable predictions",
    "All calls",
)
PAGE_SIZE = 20


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #111318;
            --surface: #181b22;
            --surface-raised: #20242d;
            --surface-soft: #15181e;
            --border: #303642;
            --border-strong: #424b5c;
            --text: #f5f6f8;
            --muted: #a5adba;
            --accent: #9aa9ff;
            --accent-strong: #b8c1ff;
            --accent-soft: rgba(154,169,255,.13);
            --amber: #f1c56d;
            --positive: #8ccba7;
            --negative: #e6a0a0;
        }

        .stApp { background: var(--bg); color: var(--text); }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
        .block-container { max-width: 1420px; padding-top: 1.05rem; padding-bottom: 3.5rem; }
        .brand-row { display: flex; justify-content: space-between; gap: 1rem; align-items: center; margin-bottom: .65rem; }
        .brand { color: var(--text); font-size: .93rem; font-weight: 800; letter-spacing: -.01em; }
        .brand-meta { color: var(--muted); font-size: .78rem; }
        .eyebrow { color: var(--accent-strong); letter-spacing: .14em; text-transform: uppercase; font-size: .7rem; font-weight: 850; }
        .app-title { margin: .12rem 0 .25rem; color: var(--text); font-size: 2.35rem; line-height: 1.05; font-weight: 780; letter-spacing: -.035em; }
        .app-subtitle { color: var(--muted); margin: 0 0 1.15rem; max-width: 820px; font-size: 1rem; line-height: 1.55; }
        .section-note { color: var(--muted); font-size: .92rem; line-height: 1.5; margin-top: -.35rem; max-width: 860px; }
        .intro-panel, .signal-panel, .probability-panel, .availability-panel, .narrative-panel {
            border: 1px solid var(--border);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(32,36,45,.98), rgba(24,27,34,.98));
            padding: 1.2rem 1.3rem;
        }
        .intro-panel { min-height: 190px; }
        .signal-panel { min-height: 208px; }
        .probability-panel { min-height: 208px; }
        .intro-panel h2, .signal-panel h2 { margin: .2rem 0 .45rem; font-size: 1.6rem; letter-spacing: -.02em; }
        .intro-panel p, .signal-panel p, .probability-panel p, .narrative-panel p { color: var(--muted); line-height: 1.55; margin: .2rem 0; }
        .card-kicker { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 800; }
        .card-title { color: var(--text); font-size: 1.02rem; font-weight: 750; margin: .2rem 0 .15rem; line-height: 1.3; }
        .card-meta { color: var(--muted); font-size: .84rem; line-height: 1.45; }
        .call-card, .model-card { border: 1px solid var(--border); border-radius: 14px; background: var(--surface); padding: 1rem 1.05rem; min-height: 175px; }
        .call-card { min-height: 192px; }
        .model-card { min-height: 205px; }
        .call-card:hover, .model-card:hover { border-color: var(--border-strong); }
        .call-card .card-value { color: var(--text); font-size: 1.35rem; font-weight: 800; margin: .55rem 0 .2rem; }
        .signal-positive { color: var(--accent-strong); font-weight: 850; }
        .signal-negative { color: var(--accent-strong); font-weight: 850; }
        .signal-neutral { color: var(--amber); font-weight: 850; }
        .outcome-positive { color: var(--positive); font-weight: 850; }
        .outcome-negative { color: var(--negative); font-weight: 850; }
        .status-pill { display: inline-block; padding: .23rem .52rem; border-radius: 999px; font-size: .7rem; font-weight: 800; letter-spacing: .01em; background: #2b313c; color: #d7dce6; }
        .status-validated { background: rgba(140,203,167,.14); color: #a9dfbd; }
        .status-retro { background: rgba(241,197,109,.14); color: #f4d58f; }
        .status-unavailable { background: rgba(230,160,160,.14); color: #f1b6b6; }
        .status-neutral { background: rgba(154,169,255,.14); color: #c5ccff; }
        .small-tag { color: var(--muted); font-size: .78rem; }
        .nav-caption { color: var(--muted); font-size: .76rem; margin: .1rem 0 -.4rem; }
        .process-step { border-top: 2px solid var(--border-strong); padding-top: .75rem; }
        .process-step strong { display: block; color: var(--text); font-size: .95rem; margin-bottom: .2rem; }
        .process-step span { color: var(--muted); font-size: .82rem; line-height: 1.45; }
        .probability-shell { margin: .8rem 0 .55rem; }
        .probability-track { position: relative; height: 18px; border-radius: 999px; background: #2b313c; overflow: visible; border: 1px solid #444c5b; }
        .probability-neutral-zone { position: absolute; top: 0; bottom: 0; border-radius: 999px; background: rgba(241,197,109,.2); }
        .probability-interval { position: absolute; top: 3px; bottom: 3px; border-radius: 999px; background: rgba(184,193,255,.45); }
        .probability-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: linear-gradient(90deg, #7588ec, #a9b4ff); }
        .probability-marker { position: absolute; top: -5px; width: 3px; height: 28px; border-radius: 3px; background: #fff; box-shadow: 0 0 0 2px rgba(17,19,24,.85); }
        .probability-base-marker { position: absolute; top: -2px; width: 2px; height: 22px; border-radius: 2px; background: var(--amber); }
        .probability-labels { display: flex; justify-content: space-between; gap: .5rem; color: var(--muted); font-size: .75rem; margin-top: .45rem; }
        .probability-legend { display: flex; gap: .85rem; flex-wrap: wrap; color: var(--muted); font-size: .77rem; margin-top: .6rem; }
        .legend-dot { display: inline-block; width: .58rem; height: .58rem; border-radius: 50%; margin-right: .25rem; }
        .legend-model { background: #fff; }
        .legend-base { background: var(--amber); }
        .legend-neutral { background: rgba(241,197,109,.55); }
        .availability-panel { border-color: rgba(241,197,109,.4); background: rgba(241,197,109,.08); }
        .availability-panel strong { color: #f6d991; }
        .availability-panel ul { margin: .55rem 0 0 1.15rem; color: var(--muted); }
        .availability-panel li { margin: .3rem 0; }
        .narrative-panel { border-left: 3px solid var(--accent); background: var(--accent-soft); }
        .narrative-panel strong { color: var(--accent-strong); }
        .metric-help { color: var(--muted); font-size: .78rem; line-height: 1.45; margin-top: -.6rem; }
        .table-note { color: var(--muted); font-size: .82rem; line-height: 1.45; }
        button:focus-visible, input:focus-visible, textarea:focus-visible, [role="button"]:focus-visible { outline: 3px solid var(--accent-strong) !important; outline-offset: 2px; }
        @media (max-width: 800px) {
            .block-container { padding: .75rem .75rem 2rem; }
            .brand-row { align-items: flex-start; flex-direction: column; gap: .15rem; }
            .app-title { font-size: 1.8rem; }
            .intro-panel, .signal-panel, .probability-panel { min-height: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_bundles(paths: tuple[str, ...]) -> tuple[tuple[ArtifactBundle, ...], tuple[str, ...]]:
    bundles: list[ArtifactBundle] = []
    errors: list[str] = []
    for path in paths:
        try:
            bundles.append(load_artifact_bundle(path))
        except ArtifactValidationError as exc:
            errors.append(f"{path}: {exc}")
    return tuple(bundles), tuple(errors)


@st.cache_data(show_spinner=False)
def _load_comparison_metrics() -> pd.DataFrame | None:
    comparison_dir = Path(__file__).resolve().parent / "artifacts" / "model_comparison"
    path = comparison_dir / "metrics.csv"
    baseline_path = comparison_dir / "baseline_metrics.csv"
    if not path.exists():
        return None
    metrics = pd.read_csv(path)
    if baseline_path.exists():
        metrics = pd.concat([metrics, pd.read_csv(baseline_path)], ignore_index=True, sort=False)
    return metrics


@st.cache_data(show_spinner=False)
def _load_comparison_manifest() -> dict[str, Any]:
    path = Path(__file__).resolve().parent / "artifacts" / "model_comparison" / "run_manifest.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _friendly(name: str) -> str:
    replacements = {
        "pres_": "Presentation ",
        "qa_": "Q&A ",
        "_z": " surprise",
        "_mean": " mean",
        "_frac": " fraction",
        "_entropy": " entropy",
        "_slope": " slope",
        "_history_count": " history count",
        "sentiment_mismatch_pos": "Presentation/Q&A positive mismatch",
        "sentiment_mismatch_neg": "Presentation/Q&A negative mismatch",
    }
    value = name
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value.replace("_", " ").strip().title()


def _as_float(value: Any) -> float | None:
    try:
        value = float(value)
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return None


def _first_feature(row: pd.Series, names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in row.index:
            value = _as_float(row.get(name))
            if value is not None:
                return value
    return None


def _format_percent(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _format_number(value: float | None) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _phase_label(row: pd.Series) -> str:
    raw = str(row.get("call_phase", "") or "").lower()
    if "pre" in raw or "open" in raw:
        return "Before market open"
    if "after" in raw or "close" in raw:
        return "After market close"
    timestamp = pd.to_datetime(row.get("call_datetime"), errors="coerce")
    if pd.notna(timestamp):
        return "Before market open" if timestamp.hour < 12 else "After market close"
    return "Timing unavailable"


def _call_key(symbol: str, call_datetime: Any) -> str:
    timestamp = pd.to_datetime(call_datetime, errors="coerce")
    return f"{symbol}|{timestamp.isoformat() if pd.notna(timestamp) else str(call_datetime)}"


def _call_label(row: pd.Series) -> str:
    timestamp = pd.to_datetime(row.get("call_datetime"), errors="coerce")
    date_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Undated call"
    return f"{row.get('company_name', row.get('symbol', 'Unknown'))} ({row.get('symbol', '—')}) · {date_text}"


def _status_description(status: str) -> str:
    return {
        "Out-of-sample holdout": "This call was kept outside the training data used for the stored prediction.",
        "Walk-forward validated": "This prediction was evaluated using only information available before the call’s evaluation period.",
        "Retrospective inference": "The model scored this historical call, but the call was not independently held out for validation.",
        "Unavailable": "No usable model score is available for this call.",
    }.get(status, "Validation provenance was not provided by the artifact.")


def _prediction_status(split: str | None, source: str = "", declared: str | None = None) -> str:
    declared_value = str(declared or "").strip().lower()
    declared_map = {
        "out_of_sample_holdout": "Out-of-sample holdout",
        "walk_forward_validated": "Walk-forward validated",
        "retrospective_inference": "Retrospective inference",
        "unavailable": "Unavailable",
    }
    if declared_value in declared_map:
        return declared_map[declared_value]
    value = str(split or "").lower()
    source_value = source.lower()
    if "final_holdout" in value or "holdout" in value:
        return "Out-of-sample holdout"
    if "walk_forward" in value or "walk-forward" in value:
        return "Walk-forward validated"
    if "out_of_sample" in source_value:
        return "Out-of-sample holdout"
    if "model_inference" in source_value or "inference" in source_value or not value:
        return "Retrospective inference"
    return "Retrospective inference"


def _status_class(status: str) -> str:
    if status in VALIDATED_STATUSES:
        return "status-validated"
    if status == "Retrospective inference":
        return "status-retro"
    if status == "Unavailable":
        return "status-unavailable"
    return "status-neutral"


def _status_pill(status: str) -> str:
    description = _escape(_status_description(status))
    return f'<span class="status-pill {_status_class(status)}" title="{description}" aria-label="{description}">{_escape(status)}</span>'


def _predict(bundle: ArtifactBundle, row: pd.DataFrame) -> tuple[float | None, str, str]:
    """Return probability, explicit provenance label, and a short source description."""
    key = row.iloc[0]
    matched_row = bundle.stored_prediction(str(key["symbol"]), key["call_datetime"])
    if matched_row is not None:
        probability = _as_float(matched_row.get("probability"))
        if probability is not None:
            status = _prediction_status(
                str(matched_row.get("split", "")),
                str(bundle.manifest.get("prediction_source", "")),
                str(matched_row.get("prediction_status", "")),
            )
            return probability, status, f"Stored {status.lower()} prediction"

    try:
        features = bundle.feature_columns
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X has feature names, but LogisticRegression was fitted without feature names")
            probabilities = bundle.model.predict_proba(row[features])[:, 1]
        probability = float(probabilities[0])
        if not 0 <= probability <= 1:
            raise ArtifactValidationError("The active model returned a probability outside [0, 1].")
        return probability, "Retrospective inference", "Model inference on a historical feature row"
    except Exception:
        return None, "Unavailable", "This model could not score the selected call"


def _base_rate(bundle: ArtifactBundle) -> float | None:
    if bundle.metrics is not None and not bundle.metrics.empty and "split" in bundle.metrics:
        for split in ("walk_forward_aggregate", "final_holdout"):
            rows = bundle.metrics[bundle.metrics["split"].eq(split)]
            if not rows.empty and "positive_rate" in rows:
                value = _as_float(rows.iloc[0]["positive_rate"])
                if value is not None:
                    return value
    target = str(bundle.schema.get("target_column", "abnormal_return_5d"))
    if target in bundle.feature_table:
        values = pd.to_numeric(bundle.feature_table[target], errors="coerce").dropna()
        if not values.empty:
            return float((values > 0).mean())
    return None


def _signal(probability: float | None, center: float) -> tuple[str, str, str]:
    if probability is None:
        return "Unavailable", "neutral", "This call does not have a usable model score."
    if probability >= min(.99, center + .05):
        return "Positive", "positive", "The model estimates a meaningfully higher chance of a positive five-session abnormal return than the base rate."
    if probability <= max(.01, center - .05):
        return "Negative", "negative", "The model estimates a meaningfully lower chance of a positive five-session abnormal return than the base rate."
    return "No clear signal", "neutral", "The model output is close to the typical positive-return rate, so the directional signal is limited."


def _conviction(probability: float | None, center: float) -> str:
    if probability is None:
        return "Unavailable"
    distance = abs(probability - center)
    if distance < .05:
        return "Low"
    if distance < .15:
        return "Medium"
    return "High"


def _collect_model_results(bundles: tuple[ArtifactBundle, ...], symbol: str, call_datetime: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    timestamp = pd.to_datetime(call_datetime, errors="coerce")
    for bundle in bundles:
        table = bundle.feature_table.copy()
        table["symbol"] = table["symbol"].astype(str)
        table["call_datetime"] = pd.to_datetime(table["call_datetime"], errors="coerce")
        matched = table[(table["symbol"] == str(symbol)) & (table["call_datetime"] == timestamp)]
        if matched.empty:
            continue
        probability, status, source = _predict(bundle, matched.iloc[[0]])
        center = _base_rate(bundle) or float(bundle.schema.get("prediction_threshold", .5))
        label, tone, _ = _signal(probability, center)
        results.append({
            "model": bundle.display_name,
            "bundle": bundle,
            "probability": probability,
            "status": status,
            "source": source,
            "signal": label,
            "tone": tone,
        })
    return results


def _model_agreement(results: list[dict[str, Any]]) -> str:
    scored = [result for result in results if result["probability"] is not None]
    if len(scored) < 2:
        return "Only one model is available for this call"
    directions = []
    for result in scored:
        center = _base_rate(result["bundle"]) or .5
        directions.append(result["probability"] >= center)
    return "Models agree on direction" if len(set(directions)) == 1 else "Models disagree on direction"


def _probability_interval(row: pd.Series) -> tuple[float, float] | None:
    for lower_name, upper_name in (
        ("probability_lower", "probability_upper"),
        ("uncertainty_lower", "uncertainty_upper"),
        ("interval_lower", "interval_upper"),
    ):
        if lower_name in row.index and upper_name in row.index:
            lower = _as_float(row.get(lower_name))
            upper = _as_float(row.get(upper_name))
            if lower is not None and upper is not None and 0 <= lower <= upper <= 1:
                return lower, upper
    value = row.get("probability_interval") if "probability_interval" in row.index else None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = None
    if isinstance(value, dict):
        lower = _as_float(value.get("lower"))
        upper = _as_float(value.get("upper"))
        if lower is not None and upper is not None and 0 <= lower <= upper <= 1:
            return lower, upper
    if isinstance(value, (list, tuple)) and len(value) == 2:
        lower = _as_float(value[0])
        upper = _as_float(value[1])
        if lower is not None and upper is not None and 0 <= lower <= upper <= 1:
            return lower, upper
    return None


def _render_probability(probability: float | None, center: float, base_rate: float | None, interval: tuple[float, float] | None = None) -> None:
    if probability is None:
        st.markdown('<div class="probability-panel"><div class="card-kicker">Probability context</div><div class="card-title">No probability available</div><p>This artifact cannot score the selected call.</p></div>', unsafe_allow_html=True)
        return
    pct = max(0.0, min(100.0, probability * 100))
    center_pct = max(0.0, min(100.0, center * 100))
    neutral_start = max(0.0, center_pct - 5)
    neutral_width = min(100.0 - neutral_start, 10.0)
    base_text = _format_percent(base_rate) if base_rate is not None else "unavailable"
    interval_markup = ""
    if interval is not None:
        lower, upper = interval
        interval_markup = f'<div class="probability-interval" style="left:{lower * 100:.2f}%;width:{(upper - lower) * 100:.2f}%"></div>'
    st.markdown(
        f'<div class="probability-panel"><div class="card-kicker">Probability context</div>'
        f'<div class="card-title">Chance of a positive five-session abnormal return</div>'
        f'<div class="probability-shell" aria-label="Model probability {_format_percent(probability)}; base rate {base_text}">'
        f'<div class="probability-track"><div class="probability-neutral-zone" style="left:{neutral_start:.2f}%;width:{neutral_width:.2f}%"></div>{interval_markup}'
        f'<div class="probability-fill" style="width:{pct:.2f}%"></div>'
        f'<div class="probability-base-marker" style="left:{center_pct:.2f}%"></div>'
        f'<div class="probability-marker" style="left:{pct:.2f}%"></div></div>'
        f'<div class="probability-labels"><span>0%</span><span>Model {probability:.0%}</span><span>100%</span></div></div>'
        f'<div class="probability-legend"><span><i class="legend-dot legend-model"></i>Model probability</span>'
        f'<span><i class="legend-dot legend-base"></i>Base rate {base_text}</span>'
        f'<span><i class="legend-dot legend-neutral"></i>Close to base rate</span>'
        f'{"<span>Interval " + _format_percent(interval[0]) + "–" + _format_percent(interval[1]) + "</span>" if interval is not None else ""}</div>'
        f'<p>Difference from base rate: <strong>{_format_percent(probability - center, signed=True)}</strong></p></div>',
        unsafe_allow_html=True,
    )


def _evidence_sections(row: pd.Series) -> list[tuple[str, str, str, list[str]]]:
    presentation = _first_feature(row, ("pres_net_sentiment", "pres_sent_mean"))
    presentation_surprise = _first_feature(row, ("pres_sent_mean_z",))
    qa = _first_feature(row, ("qa_net_sentiment", "qa_sent_mean"))
    qa_gap = _first_feature(row, ("qa_minus_pres_sent_mean", "sentiment_mismatch_pos"))
    qa_slope = _first_feature(row, ("qa_slope",))
    momentum = _first_feature(row, ("momentum_20d", "momentum_5d"))
    volatility = _first_feature(row, ("volatility_20d",))
    beta = _first_feature(row, ("beta_120d",))

    if presentation_surprise is not None and presentation_surprise > .75:
        tone_copy = "More positive than this company’s recent history."
    elif presentation_surprise is not None and presentation_surprise < -.75:
        tone_copy = "More negative than this company’s recent history."
    elif presentation is not None and presentation > .05:
        tone_copy = "Positive presentation tone, without a large historical surprise."
    elif presentation is not None and presentation < -.05:
        tone_copy = "Negative presentation tone, without a large historical surprise."
    else:
        tone_copy = "Presentation tone is close to neutral."

    if qa_gap is None:
        qa_copy = "Q&A comparison is not available in this artifact."
    elif qa_gap > .05:
        qa_copy = "Q&A tone is more positive than the prepared presentation."
    elif qa_gap < -.05:
        qa_copy = "Q&A tone is more negative than the prepared presentation."
    else:
        qa_copy = "Q&A tone is broadly aligned with the presentation."

    if momentum is None and volatility is None and beta is None:
        market_copy = "Market context is not available in this artifact."
    else:
        market_copy = "Recent market behavior is included as context, not as a price forecast."

    return [
        (
            "Management tone",
            tone_copy,
            "How the prepared presentation compares with the company’s recent language history.",
            [f"Presentation sentiment: {_format_number(presentation)}", f"Historical surprise: {_format_number(presentation_surprise)}"],
        ),
        (
            "Q&A behavior",
            qa_copy,
            "Whether the question-and-answer section shifts away from the prepared presentation.",
            [f"Q&A sentiment: {_format_number(qa)}", f"Presentation/Q&A gap: {_format_number(qa_gap)}", f"Sentiment slope: {_format_number(qa_slope)}"],
        ),
        (
            "Market context",
            market_copy,
            "Recent market conditions that help contextualize the language signal.",
            [f"20-day momentum: {_format_percent(momentum, signed=True)}", f"20-day volatility: {_format_percent(volatility)}", f"Beta: {_format_number(beta)}"],
        ),
    ]


def _render_evidence_cards(row: pd.Series) -> None:
    for title, summary, description, details in _evidence_sections(row):
        with st.expander(f"{title} — {summary}", expanded=False):
            st.caption(description)
            for detail in details:
                st.write(detail)
    st.caption("These are model inputs and associations, not proof that any single feature caused the outcome.")


def _extract_optional_series(row: pd.Series, prefixes: tuple[str, ...], json_names: tuple[str, ...]) -> list[float] | None:
    for name in json_names:
        if name not in row.index:
            continue
        value = row.get(name)
        if not isinstance(value, str):
            continue
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            decoded = list(decoded.values())
        if isinstance(decoded, list):
            values = [_as_float(item) for item in decoded]
            if values and all(item is not None for item in values):
                return [float(item) for item in values]

    columns: list[tuple[int, str]] = []
    for column in row.index:
        for prefix in prefixes:
            if str(column).startswith(prefix):
                suffix = str(column)[len(prefix):]
                match = re.search(r"-?\d+", suffix)
                if match:
                    columns.append((int(match.group()), str(column)))
    if not columns:
        return None
    values = [_as_float(row.get(column)) for _, column in sorted(columns)]
    if not values or any(item is None for item in values):
        return None
    return [float(item) for item in values]


def _render_event_chart(row: pd.Series) -> bool:
    stock = _extract_optional_series(row, ("price_", "close_", "stock_price_"), ("price_series", "stock_price_series"))
    benchmark = _extract_optional_series(row, ("benchmark_", "market_price_"), ("benchmark_series", "market_price_series"))
    if stock is None:
        return False
    frame = pd.DataFrame({"Stock": stock})
    if benchmark is not None and len(benchmark) == len(stock):
        frame["Market / sector benchmark"] = benchmark
    frame = frame.divide(frame.iloc[0]).multiply(100)
    frame.index = [f"Session {index - 1:+d}" for index in range(len(frame))]
    st.line_chart(frame, use_container_width=True)
    st.caption("Indexed to 100 at the first available session. The call marker and five-session window are represented by the session labels.")
    with st.expander("Event-window data table", expanded=False):
        accessible_frame = frame.reset_index().rename(columns={"index": "Session"})
        st.dataframe(accessible_frame, use_container_width=True, hide_index=True)
    return True


def _render_language_profile(row: pd.Series) -> bool:
    columns = ("pres_begin_mean", "pres_middle_mean", "pres_end_mean", "qa_begin_mean", "qa_middle_mean", "qa_end_mean")
    if not all(column in row.index for column in columns):
        return False
    timeline = pd.DataFrame(
        {
            "Presentation": [row.get("pres_begin_mean"), row.get("pres_middle_mean"), row.get("pres_end_mean")],
            "Q&A": [row.get("qa_begin_mean"), row.get("qa_middle_mean"), row.get("qa_end_mean")],
        },
        index=["Beginning", "Middle", "End"],
    )
    st.line_chart(timeline, use_container_width=True)
    st.caption("Descriptive language pattern; not causal evidence.")
    return True


def _render_availability(row: pd.Series, language_available: bool, event_available: bool) -> None:
    missing: list[str] = []
    if not event_available:
        missing.append("Price and benchmark series — the offline artifact does not contain an event-window chart yet.")
    evidence = row.get("transcript_evidence")
    if evidence in (None, "", "nan") or (isinstance(evidence, float) and pd.isna(evidence)):
        missing.append("Transcript excerpts — source-linked sentences are not stored, so no quotes are shown.")
    if not language_available:
        missing.append("Sentence-position language features — this model does not include the feature group needed for that view.")
    if not missing:
        return
    items = "".join(f"<li>{_escape(item)}</li>" for item in missing)
    st.markdown(
        f'<div class="availability-panel"><strong>Evidence availability</strong>'
        f'<div class="card-meta">The model score is still shown, but these supporting evidence layers are not present in the selected artifact:</div>'
        f'<ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def _model_explanation(bundle: ArtifactBundle, row: pd.DataFrame) -> pd.DataFrame:
    model = bundle.model
    is_pipeline = hasattr(model, "named_steps") and "model" in model.named_steps
    estimator = model.named_steps["model"] if is_pipeline else model
    features = bundle.feature_columns
    if hasattr(estimator, "coef_"):
        values = row[features].apply(pd.to_numeric, errors="coerce").fillna(0)
        if is_pipeline and "preprocessor" in model.named_steps:
            values = model.named_steps["preprocessor"].transform(values)
        else:
            values = values.iloc[0].to_numpy(dtype=float)
        if hasattr(values, "toarray"):
            values = values.toarray()
        values = np.asarray(values, dtype=float).reshape(-1)
        coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if len(values) != len(features) or len(coefficients) != len(features):
            return pd.DataFrame()
        contributions = values * coefficients
        return pd.DataFrame({"Feature": [_friendly(name) for name in features], "Contribution": contributions}).sort_values("Contribution", key=abs, ascending=False)
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({"Feature": [_friendly(name) for name in features], "Global importance": estimator.feature_importances_}).sort_values("Global importance", ascending=False)
    return pd.DataFrame()


def _render_call_comparison(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No comparable model scores are available for this call.")
        return
    rows = []
    for result in results:
        rows.append({
            "Model": result["model"],
            "Probability": _format_percent(result["probability"]),
            "Direction": result["signal"],
            "Validation": result["status"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(_model_agreement(results) + ". Agreement is useful context, not a guarantee of correctness.")


def _render_historical_outcome(bundle: ArtifactBundle, row: pd.Series) -> None:
    target_column = str(bundle.schema.get("target_column", "abnormal_return_5d"))
    abnormal = _as_float(row.get(target_column))
    raw_return = _as_float(row.get("return"))
    if abnormal is None and raw_return is None:
        st.info("No realized five-session outcome is included for this call.")
        return
    left, right = st.columns(2)
    with left:
        if abnormal is None:
            st.metric("Abnormal return outcome", "Unavailable")
        elif abnormal > 0:
            st.metric("Abnormal return outcome", "Positive", _format_percent(abnormal, signed=True))
        else:
            st.metric("Abnormal return outcome", "Negative", _format_percent(abnormal, signed=True))
    with right:
        if raw_return is None:
            st.metric("Raw stock return", "Unavailable")
        else:
            st.metric("Raw stock return", "Positive" if raw_return > 0 else "Negative", _format_percent(raw_return, signed=True))
    st.caption("Historical outcome · not a prediction. Backtest context only; realized outcomes were not provided to the model as inputs.")


def _prepare_table(bundle: ArtifactBundle) -> pd.DataFrame:
    table = bundle.feature_table.copy()
    table["symbol"] = table["symbol"].astype(str)
    table["call_datetime"] = pd.to_datetime(table["call_datetime"], errors="coerce")
    table = table.sort_values("call_datetime", ascending=False)
    center = _base_rate(bundle) or float(bundle.schema.get("prediction_threshold", .5))
    probabilities: list[float | None] = []
    statuses: list[str] = []
    signals: list[str] = []
    tones: list[str] = []
    convictions: list[str] = []
    for _, row in table.iterrows():
        probability, status, _ = _predict(bundle, row.to_frame().T)
        signal, tone, _ = _signal(probability, center)
        probabilities.append(probability)
        statuses.append(status)
        signals.append(signal)
        tones.append(tone)
        convictions.append(_conviction(probability, center))
    table["_probability"] = probabilities
    table["_status"] = statuses
    table["_signal"] = signals
    table["_tone"] = tones
    table["_conviction"] = convictions
    return table


def _open_call(row: pd.Series) -> None:
    st.session_state["selected_call_key"] = _call_key(str(row["symbol"]), row["call_datetime"])
    st.session_state["_next_view"] = "Detail"
    st.rerun()


def _render_call_card(row: pd.Series, base_rate: float, key_suffix: str, history: bool = False) -> None:
    timestamp = pd.to_datetime(row["call_datetime"], errors="coerce")
    date_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Undated call"
    signal_class = f"signal-{row['_tone']}"
    agreement = ""
    st.markdown(
        f'<div class="call-card"><div class="card-kicker">{_escape(_phase_label(row))} · {_status_pill(str(row["_status"]))}</div>'
        f'<div class="card-title">{_escape(row.get("company_name", row["symbol"]))} <span class="small-tag">{_escape(row["symbol"])}</span></div>'
        f'<div class="card-meta">{_escape(date_text)}</div>'
        f'<div class="card-value"><span class="{signal_class}">{_escape(row["_signal"])}</span> <span class="card-meta">· {_format_percent(row["_probability"])}</span></div>'
        f'<div class="card-meta">{_escape(row["_conviction"])} confidence · base rate {_format_percent(base_rate)}{agreement}</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("Open call" if not history else "Open", key=f"open-{key_suffix}", use_container_width=True):
        _open_call(row)


def _render_calls(bundle: ArtifactBundle, bundles: tuple[ArtifactBundle, ...]) -> None:
    st.markdown('<div class="eyebrow">Explore</div><div class="app-title">Calls</div>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Start with calls that have validation evidence. Open a company to compare its earnings-call history over time.</p>', unsafe_allow_html=True)

    table = _prepare_table(bundle)
    base_rate = _base_rate(bundle)
    control_a, control_b, control_c = st.columns([1.25, 1, 1])
    with control_a:
        query = st.text_input("Search company or ticker", placeholder="Search by company or ticker", key="calls_query")
    with control_b:
        status_filter = st.selectbox("Call coverage", STATUS_FILTERS, key="calls_status_filter")
    with control_c:
        timing_filter = st.selectbox("Event timing", ["All timings", "Before market open", "After market close"], key="calls_timing_filter")

    filter_a, filter_b, filter_c = st.columns([1, 1, 1.1])
    with filter_a:
        direction_filter = st.selectbox("Direction", ["All directions", "Positive", "Negative", "No clear signal"], key="calls_direction_filter")
    with filter_b:
        confidence_filter = st.selectbox("Confidence", ["All confidence", "Low", "Medium", "High"], key="calls_confidence_filter")
    with filter_c:
        st.caption(f"Active artifact: {bundle.display_name}")
        st.markdown(f'<span class="table-note">{len(table):,} calls in this artifact · base rate {_format_percent(base_rate)}</span>', unsafe_allow_html=True)

    filtered = table
    if query:
        query_upper = query.upper()
        company_series = filtered.get("company_name", pd.Series(index=filtered.index, dtype=str)).astype(str)
        filtered = filtered[
            filtered["symbol"].str.upper().str.contains(query_upper, na=False)
            | company_series.str.upper().str.contains(query_upper, na=False)
        ]
    if status_filter == "Validated calls":
        filtered = filtered[filtered["_status"].isin(VALIDATED_STATUSES)]
    elif status_filter == "Exploratory archive":
        filtered = filtered[filtered["_status"].eq("Retrospective inference")]
    elif status_filter == "Unavailable predictions":
        filtered = filtered[filtered["_status"].eq("Unavailable")]
    if timing_filter != "All timings":
        filtered = filtered[filtered.apply(_phase_label, axis=1).eq(timing_filter)]
    if direction_filter != "All directions":
        filtered = filtered[filtered["_signal"].eq(direction_filter)]
    if confidence_filter != "All confidence":
        filtered = filtered[filtered["_conviction"].eq(confidence_filter)]

    if filtered.empty:
        if status_filter == "Validated calls":
            st.warning("There are no validated calls for this model and filter combination.")
            st.info("Try the Exploratory archive to inspect retrospective scores, or choose a different model from a call’s detail page.")
        else:
            st.info("No calls match those filters.")
        return

    groups: list[tuple[str, str, pd.DataFrame]] = []
    group_columns = ["symbol", "company_name"] if "company_name" in filtered else ["symbol"]
    for group_key, group in filtered.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        symbol = str(group_key[0])
        company = str(group_key[1]) if len(group_key) > 1 else symbol
        groups.append((symbol, company, group.sort_values("call_datetime", ascending=False)))
    groups.sort(key=lambda item: item[2].iloc[0]["call_datetime"], reverse=True)

    page_count = max(1, int(np.ceil(len(groups) / PAGE_SIZE)))
    stored_page = int(st.session_state.get("calls_page", 1))
    if stored_page > page_count:
        st.session_state["calls_page"] = 1
        stored_page = 1
    page = int(st.number_input("Page", min_value=1, max_value=page_count, value=stored_page, step=1, key="calls_page"))
    start = (page - 1) * PAGE_SIZE
    visible_groups = groups[start:start + PAGE_SIZE]
    st.caption(f"Showing {len(filtered):,} matching calls across {len(groups):,} companies · page {page} of {page_count}")

    columns = st.columns(2)
    for index, (symbol, company, group) in enumerate(visible_groups):
        latest = group.iloc[0]
        with columns[index % 2]:
            _render_call_card(latest, base_rate or .5, f"latest-{_call_key(symbol, latest['call_datetime'])}")
            if len(group) > 1:
                with st.expander(f"View {len(group) - 1} earlier calls for {company}", expanded=False):
                    for history_index, (_, history_row) in enumerate(group.iloc[1:].iterrows()):
                        _render_call_card(history_row, base_rate or .5, f"history-{_call_key(symbol, history_row['call_datetime'])}", history=True)

    previous, page_label, next_page = st.columns([1, 2, 1])
    with previous:
        if st.button("Previous", disabled=page <= 1, key="calls_previous", use_container_width=True):
            st.session_state["calls_page"] = page - 1
            st.rerun()
    with page_label:
        st.markdown(f'<div style="text-align:center;color:var(--muted);padding:.45rem 0">Page {page} of {page_count}</div>', unsafe_allow_html=True)
    with next_page:
        if st.button("Next", disabled=page >= page_count, key="calls_next", use_container_width=True):
            st.session_state["calls_page"] = page + 1
            st.rerun()


def _comparison_row(metrics: pd.DataFrame | None, model: str, split: str) -> pd.Series | None:
    if metrics is None or metrics.empty or "model" not in metrics or "split" not in metrics:
        return None
    rows = metrics[(metrics["model"] == model) & (metrics["split"] == split)]
    return rows.iloc[0] if not rows.empty else None


def _metric_text(row: pd.Series | None, name: str, percent: bool = False) -> str:
    if row is None or name not in row:
        return "—"
    value = _as_float(row[name])
    if value is None:
        return "—"
    return f"{value:.1%}" if percent else f"{value:.3f}"


def _render_reliability() -> None:
    st.markdown('<div class="eyebrow">Reliability</div><div class="app-title">How much should you trust the models?</div>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Compare time-separated evaluation results without treating one metric or one holdout as a definitive winner.</p>', unsafe_allow_html=True)
    metrics = _load_comparison_metrics()
    if metrics is None:
        st.warning("The controlled comparison artifact is unavailable.")
        return

    model_config = [
        ("sentence_plus_historical_xgboost_depth1_trees100", "Rich XGBoost", "Language, sentence-position, and historical context features.", "Richer candidate"),
        ("original_logistic", "Original Logistic", "Smaller sentiment model used as the reference.", "Reference"),
        ("market_only_logistic", "Market-only baseline", "Recent market behavior without transcript language.", "Context baseline"),
    ]

    cards = st.columns(3)
    for column, (model, title, description, badge) in zip(cards, model_config):
        walk = _comparison_row(metrics, model, "walk_forward_aggregate")
        holdout = _comparison_row(metrics, model, "final_holdout")
        with column:
            st.markdown(f'<div class="model-card"><div class="card-kicker">{_escape(badge)}</div><div class="card-title">{_escape(title)}</div><div class="card-meta">{_escape(description)}</div></div>', unsafe_allow_html=True)
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Walk-forward AUC", _metric_text(walk, "auc"))
            metric_right.metric("Latest holdout", _metric_text(holdout, "auc"))
            st.markdown(f'<div class="metric-help">Brier score: walk-forward {_metric_text(walk, "brier")} · holdout {_metric_text(holdout, "brier")}</div>', unsafe_allow_html=True)
            if walk is not None and "auc_ci_lower_95" in walk and "auc_ci_upper_95" in walk:
                st.caption(f"Walk-forward 95% range: {_metric_text(walk, 'auc_ci_lower_95')}–{_metric_text(walk, 'auc_ci_upper_95')}")

    st.markdown('<div class="narrative-panel"><strong>What the current evidence says</strong><p>The richer model generalizes better across time-separated folds, while the original model performs slightly better on the latest holdout. The evidence does not establish a definitive winner.</p></div>', unsafe_allow_html=True)

    st.subheader("AUC across evaluation views")
    st.caption("AUC measures ranking quality. 0.50 is the no-information reference; read the direct labels because the differences are intentionally not exaggerated.")
    fig = go.Figure()
    all_values: list[float] = []
    for model, title, _, _ in model_config:
        walk = _comparison_row(metrics, model, "walk_forward_aggregate")
        holdout = _comparison_row(metrics, model, "final_holdout")
        walk_auc = _as_float(walk["auc"]) if walk is not None else None
        holdout_auc = _as_float(holdout["auc"]) if holdout is not None else None
        values = [value for value in (walk_auc, holdout_auc) if value is not None]
        all_values.extend(values)
        if not values:
            continue
        x_values = [walk_auc, holdout_auc]
        y_values = [title, title]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers+text",
            text=[_metric_text(walk, "auc"), _metric_text(holdout, "auc")],
            textposition=["top center", "bottom center"],
            textfont={"color": "#f5f6f8", "size": 12},
            marker={"size": 11, "color": ["#9aa9ff", "#f1c56d"]},
            line={"color": "#6f7890", "width": 2},
            name=title,
            hovertemplate=f"{title}<br>%{{x:.3f}}<extra></extra>",
        ))
        if walk is not None and _as_float(walk.get("auc_ci_lower_95")) is not None and _as_float(walk.get("auc_ci_upper_95")) is not None:
            lower = _as_float(walk["auc_ci_lower_95"])
            upper = _as_float(walk["auc_ci_upper_95"])
            fig.add_trace(go.Scatter(
                x=[walk_auc],
                y=[title],
                mode="markers",
                marker={"size": 11, "color": "#9aa9ff", "opacity": 0},
                error_x={"type": "data", "symmetric": False, "array": [upper - walk_auc], "arrayminus": [walk_auc - lower], "color": "#9aa9ff", "thickness": 2, "width": 5},
                showlegend=False,
                hoverinfo="skip",
            ))
    fig.add_vline(x=.5, line_dash="dash", line_color="#f1c56d", annotation_text="No information · 0.50", annotation_position="top")
    fig.update_layout(
        height=365,
        margin={"l": 8, "r": 18, "t": 28, "b": 28},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#a5adba"},
        showlegend=False,
        xaxis={"title": "AUC · higher is better", "range": [0, 1], "fixedrange": False, "gridcolor": "#303642", "zeroline": False},
        yaxis={"gridcolor": "#303642", "categoryorder": "array", "categoryarray": [item[1] for item in model_config]},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    manifest = _load_comparison_manifest()
    context_left, context_middle, context_right = st.columns(3)
    context_left.metric("Events compared", str(manifest.get("rows", "—")))
    context_middle.metric("Companies", str(manifest.get("companies", "—")))
    years = manifest.get("walk_forward_years", [])
    context_right.metric("Walk-forward years", f"{min(years)}–{max(years)}" if years else "—")

    with st.expander("Metric guide", expanded=False):
        st.markdown("**AUC** — how often the model ranks a positive outcome above a negative outcome. 0.50 is close to chance.\n\n**Brier score** — how closely probabilities match eventual outcomes; lower is better.\n\n**Confidence interval** — a plausible range around the measured AUC, not a guarantee about future performance.")

    with st.expander("Technical details", expanded=False):
        st.markdown("Walk-forward evaluation trains on prior years and tests on the next year. The latest holdout is separated from training and remains exploratory for this offline artifact.")
        selected_metrics = ["model", "split", "n", "positive_rate", "auc", "brier", "log_loss", "accuracy", "f1", "auc_ci_lower_95", "auc_ci_upper_95"]
        available = [column for column in selected_metrics if column in metrics]
        technical = metrics[metrics["model"].isin([item[0] for item in model_config]) & metrics["split"].isin(["walk_forward_aggregate", "final_holdout"])][available].copy()
        if not technical.empty:
            technical["model"] = technical["model"].map({item[0]: item[1] for item in model_config}).fillna(technical["model"])
            technical["split"] = technical["split"].replace({"walk_forward_aggregate": "Walk-forward", "final_holdout": "Latest holdout"})
            st.dataframe(technical, use_container_width=True, hide_index=True)


def _render_call_detail(bundles: tuple[ArtifactBundle, ...], label_to_bundle: dict[str, ArtifactBundle], labels: list[str]) -> None:
    back_column, model_column = st.columns([1, 2])
    with back_column:
        if st.button("← Back to calls", key="back-to-calls"):
            st.session_state["_next_view"] = "Calls"
            st.rerun()
    with model_column:
        current_label = st.session_state.get("active_model_label", labels[0])
        if current_label not in labels:
            current_label = labels[0]
        selected_model_label = st.selectbox("Model used for this view", labels, index=labels.index(current_label), key="detail_model_label", help="Changing the model changes the score, provenance, and available feature evidence for this call.")
        st.session_state["active_model_label"] = selected_model_label
    bundle = label_to_bundle[selected_model_label]

    table = _prepare_table(bundle)
    if table.empty:
        st.warning("This model artifact has no calls to display.")
        return
    selected_key = st.session_state.get("selected_call_key")
    selected_symbol = str(selected_key).split("|", 1)[0] if selected_key else str(table.iloc[0]["symbol"])
    symbols = sorted(table["symbol"].unique().tolist())
    if selected_symbol not in symbols:
        selected_symbol = symbols[0]
    company_names = table.groupby("symbol")["company_name"].first().to_dict() if "company_name" in table else {}
    company_label = lambda symbol: f"{company_names.get(symbol, symbol)} ({symbol})"
    selected_symbol = st.selectbox("Company", symbols, index=symbols.index(selected_symbol), format_func=company_label, key="detail_company_select")

    company_calls = table[table["symbol"] == selected_symbol].sort_values("call_datetime", ascending=False)
    call_options = list(company_calls.index)
    call_labels: list[str] = []
    for index in call_options:
        row = company_calls.loc[index]
        timestamp = pd.to_datetime(row["call_datetime"], errors="coerce")
        call_labels.append(f"{timestamp.strftime('%b %d, %Y · %H:%M') if pd.notna(timestamp) else 'Undated call'} · {row['_signal']} · {row['_status']}")
    default_index = 0
    if selected_key:
        for position, index in enumerate(call_options):
            row = company_calls.loc[index]
            if _call_key(selected_symbol, row["call_datetime"]) == selected_key:
                default_index = position
                break
    selected_call_label = st.selectbox("Call date", call_labels, index=default_index, key=f"detail_call_select_{selected_symbol}")
    selected_index = call_options[call_labels.index(selected_call_label)]
    selected = company_calls.loc[[selected_index]].copy()
    selected_row = selected.iloc[0]
    st.session_state["selected_call_key"] = _call_key(str(selected_row["symbol"]), selected_row["call_datetime"])

    probability = _as_float(selected_row["_probability"])
    status = str(selected_row["_status"])
    source = _predict(bundle, selected)[2]
    base_rate = _base_rate(bundle)
    center = base_rate or float(bundle.schema.get("prediction_threshold", .5))
    signal, tone, explanation = _signal(probability, center)
    company_name = selected_row.get("company_name", selected_row.get("symbol", "Unknown"))
    timestamp = pd.to_datetime(selected_row.get("call_datetime"), errors="coerce")
    timestamp_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Date unavailable"

    st.markdown(f'<div class="eyebrow">{_escape(selected_row.get("symbol", "—"))} · {_escape(_phase_label(selected_row))} · {_status_pill(status)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title">{_escape(company_name)}</div><p class="app-subtitle">{_escape(timestamp_text)} · Five-session abnormal-return research signal</p>', unsafe_allow_html=True)
    st.caption(_status_description(status) + " " + source + ". This is a directional research estimate, not a price target or trade recommendation.")

    panel_left, panel_right = st.columns([1.05, 1])
    with panel_left:
        st.markdown(f'<div class="signal-panel"><div class="card-kicker">What the model thinks</div><h2 class="signal-{tone}">{_escape(signal)}</h2><p>{_escape(explanation)}</p><div class="card-meta">Confidence: <strong>{_escape(selected_row["_conviction"])}</strong> · Model: {_escape(bundle.display_name)}</div></div>', unsafe_allow_html=True)
    with panel_right:
        _render_probability(probability, center, base_rate, _probability_interval(selected_row))

    summary = st.columns(4)
    summary[0].metric("Model probability", _format_percent(probability))
    summary[1].metric("Typical positive rate", _format_percent(base_rate))
    summary[2].metric("Confidence", str(selected_row["_conviction"]))
    summary[3].metric("Evaluation window", "5 sessions")

    st.divider()
    st.subheader("Why the model responded")
    _render_evidence_cards(selected_row)

    event_available = False
    language_available = False
    if _extract_optional_series(selected_row, ("price_", "close_", "stock_price_"), ("price_series", "stock_price_series")) is not None:
        st.subheader("Event window")
        event_available = _render_event_chart(selected_row)
    with st.expander("Supporting language pattern", expanded=False):
        language_available = _render_language_profile(selected_row)

    _render_availability(selected_row, language_available, event_available)

    with st.expander("Compare model outputs", expanded=False):
        _render_call_comparison(_collect_model_results(bundles, str(selected_row["symbol"]), selected_row["call_datetime"]))

    evidence = selected_row.get("transcript_evidence")
    if evidence not in (None, "", "nan") and not (isinstance(evidence, float) and pd.isna(evidence)):
        with st.expander("Source-linked transcript evidence", expanded=False):
            st.write(evidence)

    with st.expander("Historical outcome · not a prediction", expanded=False):
        _render_historical_outcome(bundle, selected_row)

    with st.expander("Company call history", expanded=False):
        history = table[table["symbol"] == str(selected_row["symbol"])].sort_values("call_datetime", ascending=False)
        history_columns = [column for column in ("symbol", "company_name", "call_datetime", "_signal", "_probability", "_status") if column in history]
        history_view = history[history_columns].copy()
        if "call_datetime" in history_view:
            history_view["call_datetime"] = history_view["call_datetime"].dt.strftime("%b %d, %Y · %H:%M")
        history_view = history_view.rename(columns={"_signal": "Direction", "_probability": "Probability", "_status": "Validation"})
        st.dataframe(history_view, use_container_width=True, hide_index=True)

    with st.expander("Technical details", expanded=False):
        explanation_table = _model_explanation(bundle, selected)
        if not explanation_table.empty:
            st.markdown("**Model explanation**")
            st.caption("For linear models, contributions are call-specific. For tree models, importance is global and should not be read as causal evidence.")
            st.dataframe(explanation_table.head(12), use_container_width=True, hide_index=True)
        feature_columns = [column for column in bundle.feature_columns if column in selected]
        if feature_columns:
            st.markdown("**Raw model inputs**")
            raw = selected[feature_columns].T.rename(columns={selected.index[0]: "Value"})
            raw.index = [_friendly(str(index)) for index in raw.index]
            st.dataframe(raw, use_container_width=True)
        st.download_button("Download selected call data", selected.to_csv(index=False).encode("utf-8"), file_name=f"{selected_row['symbol']}_earnings_call.csv", mime="text/csv")


def _render_home(bundle: ArtifactBundle) -> None:
    st.markdown('<div class="eyebrow">Research workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Earnings Call Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Estimate whether a company’s stock may outperform or underperform the market over the five sessions after an earnings call, then inspect why and how much to trust the result.</p>', unsafe_allow_html=True)

    intro_left, intro_right = st.columns([1.12, .88])
    with intro_left:
        st.markdown('<div class="intro-panel"><div class="card-kicker">Start here</div><h2>Turn one earnings call into a research question.</h2><p>The model reads call-language patterns and market context to estimate direction. It does not set a price target, make a trade, or replace your own analysis.</p></div>', unsafe_allow_html=True)
        cta_left, cta_right = st.columns([1, 1])
        with cta_left:
            if st.button("Explore validated calls", type="primary", use_container_width=True, key="home-explore"):
                st.session_state["_next_view"] = "Calls"
                st.rerun()
        with cta_right:
            if st.button("Learn how validation works", use_container_width=True, key="home-learn"):
                st.session_state["_next_view"] = "Reliability"
                st.rerun()
    with intro_right:
        st.markdown('<div class="narrative-panel"><strong>What to look for</strong><p>Open a validated call, read the directional signal, compare it with the normal positive-return rate, and expand the evidence only when you want the details.</p><p>Every score is labeled so a retrospective model inference cannot be mistaken for an out-of-sample result.</p></div>', unsafe_allow_html=True)

    st.subheader("How it works")
    steps = st.columns(4)
    process = [
        ("01 · Call", "Choose a company and earnings call."),
        ("02 · Signal", "See Positive, Negative, or No clear signal."),
        ("03 · Why", "Inspect tone, Q&A behavior, and market context."),
        ("04 · Trust", "Check validation status and model reliability."),
    ]
    for column, (title, copy) in zip(steps, process):
        with column:
            st.markdown(f'<div class="process-step"><strong>{_escape(title)}</strong><span>{_escape(copy)}</span></div>', unsafe_allow_html=True)

    table = bundle.feature_table
    validated_count = len(bundle.predictions) if bundle.predictions is not None else 0
    st.subheader("Current research coverage")
    metrics = st.columns(3)
    metrics[0].metric("Stored validated predictions", f"{validated_count:,}")
    metrics[1].metric("Companies in active artifact", f"{table['symbol'].nunique():,}")
    metrics[2].metric("Base positive-return rate", _format_percent(_base_rate(bundle)))
    st.caption(f"Active research context: {bundle.display_name}. Offline artifact only; no live transcripts, prices, or investment advice.")


def main() -> None:
    _inject_css()
    try:
        artifact_paths = tuple(str(path) for path in discover_artifact_dirs())
        bundles, bundle_errors = _load_bundles(artifact_paths)
    except ArtifactValidationError as exc:
        st.error("No model artifact bundles could be discovered.")
        st.code(str(exc))
        st.info("Create sibling bundles under artifacts/ or set EARNINGS_ARTIFACT_DIR to one validated bundle.")
        return

    for error in bundle_errors:
        st.warning(f"A model bundle was skipped because it failed validation: {error}")
    if not bundles:
        st.error("Every discovered model bundle failed validation.")
        return

    bundles = tuple(sorted(bundles, key=lambda item: (item.is_experimental, item.display_label.lower())))
    label_to_bundle: dict[str, ArtifactBundle] = {}
    labels: list[str] = []
    for bundle in bundles:
        label = bundle.display_label
        unique_label = label if label not in label_to_bundle else f"{label} [{bundle.model_version}]"
        labels.append(unique_label)
        label_to_bundle[unique_label] = bundle

    if "active_model_label" not in st.session_state:
        preferred = next((label for label in labels if label_to_bundle[label].predictions is not None), labels[0])
        st.session_state["active_model_label"] = preferred
    if st.session_state["active_model_label"] not in labels:
        st.session_state["active_model_label"] = labels[0]
    active_bundle = label_to_bundle[st.session_state["active_model_label"]]

    if "_next_view" in st.session_state:
        next_view = st.session_state.pop("_next_view")
        st.session_state["view"] = next_view
        st.session_state["top_navigation"] = "Calls" if next_view == "Detail" else next_view
    st.markdown('<div class="brand-row"><div class="brand">◈ Earnings Call Intelligence</div><div class="brand-meta">Offline research artifact · direction, evidence, reliability</div></div>', unsafe_allow_html=True)
    view = st.session_state.get("view", "Home")
    default_nav = "Calls" if view == "Detail" else view if view in {"Home", "Calls", "Reliability"} else "Home"
    if "top_navigation" not in st.session_state:
        st.session_state["top_navigation"] = default_nav
    navigation = st.radio("Navigate", ["Home", "Calls", "Reliability"], horizontal=True, key="top_navigation", label_visibility="collapsed")
    if view != "Detail" or navigation != "Calls":
        view = navigation
        st.session_state["view"] = view

    if view == "Home":
        _render_home(active_bundle)
    elif view == "Calls":
        _render_calls(active_bundle, bundles)
    elif view == "Reliability":
        _render_reliability()
    else:
        _render_call_detail(bundles, label_to_bundle, labels)


if __name__ == "__main__":
    main()
