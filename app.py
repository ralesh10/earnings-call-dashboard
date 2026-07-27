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
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
        :root {
            --bg: #07090e;
            --surface: #0d1118;
            --surface-raised: #131822;
            --surface-hover: #19202e;
            --border: #1e2636;
            --border-soft: #141a26;
            --border-focus: #33415c;
            --text: #eef2f9;
            --muted: #8c97ab;
            --muted-dim: #545f73;
            --gold: #f5b041;
            --gold-dim: #9a6f28;
            --gold-soft: rgba(245,176,65,.12);
            --amber: #f5b041;
            --up: #00e599;
            --up-soft: rgba(0,229,153,.12);
            --down: #ff4d6d;
            --down-soft: rgba(255,77,109,.12);
            --neutral: #38bdf8;
            --neutral-soft: rgba(56,189,248,.12);
            --font-display: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'IBM Plex Mono', ui-monospace, SFMono-Regular, monospace;
        }

        * { box-sizing: border-box; }
        .stApp { background: var(--bg); color: var(--text); font-family: var(--font-body); }
        [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none; }
        .block-container { max-width: 1440px; padding: 0 1.4rem 3.5rem; }
        h1, h2, h3, h4, h5 { font-family: var(--font-display); color: var(--text); }
        .topbar-shell { border-bottom: 1px solid var(--border); background: rgba(13,17,24,.96); margin: 0 -1.4rem 1rem; padding: .7rem 1.4rem .35rem; }
        .brand-row { display: flex; align-items: center; gap: .65rem; min-height: 34px; }
        .brand-mark { width: 28px; height: 28px; border-radius: 6px; background: linear-gradient(135deg,var(--gold),var(--gold-dim)); display: inline-flex; align-items: center; justify-content: center; color: #07090e; font: 700 12px var(--font-mono); }
        .brand-title { font: 700 15px var(--font-display); letter-spacing: -.02em; }
        .brand-badge { border: 1px solid var(--border); background: var(--surface-raised); color: var(--muted); border-radius: 4px; padding: 2px 6px; font: 10px var(--font-mono); text-transform: uppercase; }
        .brand-meta { color: var(--muted); font: 11px var(--font-mono); }
        .eyebrow { color: var(--gold); font: 11px var(--font-mono); letter-spacing: .1em; text-transform: uppercase; margin-bottom: .35rem; }
        .app-title { margin: .1rem 0 .25rem; color: var(--text); font: 700 1.85rem/1.15 var(--font-display); letter-spacing: -.03em; }
        .app-subtitle, .section-note { color: var(--muted); max-width: 900px; font-size: .93rem; line-height: 1.5; }
        .card, .card-raised, .intro-panel, .signal-panel, .probability-panel, .availability-panel, .narrative-panel, .model-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1.15rem; }
        .card-raised { background: var(--surface-raised); }
        .intro-panel, .signal-panel, .probability-panel { min-height: 200px; }
        .intro-panel h2, .signal-panel h2 { margin: .25rem 0 .5rem; font-size: 1.5rem; }
        .intro-panel p, .signal-panel p, .probability-panel p, .narrative-panel p { color: var(--muted); line-height: 1.5; }
        .card-kicker, .stat-lbl { color: var(--muted); font: 10px var(--font-mono); letter-spacing: .08em; text-transform: uppercase; }
        .card-title { color: var(--text); font-size: 1rem; font-weight: 700; line-height: 1.3; }
        .card-meta, .small-tag { color: var(--muted); font-size: .8rem; line-height: 1.45; }
        .call-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: .8rem; min-height: 0; }
        .call-card .card-value { color: var(--text); font: 600 1rem var(--font-mono); margin: .35rem 0; }
        .model-card { min-height: 176px; }
        .signal-positive { color: var(--up); font-weight: 800; }
        .signal-negative { color: var(--down); font-weight: 800; }
        .signal-neutral { color: var(--neutral); font-weight: 800; }
        .outcome-positive { color: var(--up); font-weight: 800; }
        .outcome-negative { color: var(--down); font-weight: 800; }
        .status-pill, .badge { display: inline-flex; align-items: center; gap: 4px; border-radius: 4px; padding: 2px 7px; font: 600 10px var(--font-mono); }
        .status-validated, .badge-up { background: var(--up-soft); border: 1px solid rgba(0,229,153,.28); color: var(--up); }
        .status-retro, .badge-gold { background: var(--gold-soft); border: 1px solid rgba(245,176,65,.28); color: var(--gold); }
        .status-unavailable, .badge-down { background: var(--down-soft); border: 1px solid rgba(255,77,109,.28); color: var(--down); }
        .status-neutral, .badge-neutral { background: var(--neutral-soft); border: 1px solid rgba(56,189,248,.28); color: var(--neutral); }
        .badge-muted { background: var(--surface-raised); border: 1px solid var(--border); color: var(--muted); }
        .infoicon { display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; border: 1px solid var(--muted-dim); border-radius: 50%; color: var(--muted); font: 10px var(--font-mono); cursor: help; margin-left: 3px; }
        .infoicon:hover { border-color: var(--gold); color: var(--gold); }
        .infoicon .tip { visibility: hidden; opacity: 0; position: absolute; width: 240px; margin: -8px 0 0 8px; padding: 9px 11px; background: #161c28; border: 1px solid var(--border-focus); border-radius: 6px; color: var(--text); font: 11px/1.4 var(--font-body); text-transform: none; z-index: 50; box-shadow: 0 10px 25px rgba(0,0,0,.55); }
        .infoicon:hover .tip { visibility: visible; opacity: 1; }
        .section-header { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin: 1.8rem 0 .8rem; border-bottom: 1px solid var(--border); padding-bottom: .55rem; }
        .section-header h2 { font-size: 1.1rem; }
        .section-header .sub { color: var(--muted); font: 11px var(--font-mono); }
        .hero-stats, .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-soft); }
        .stat-box { background: var(--surface-raised); padding: .7rem .8rem; border: 1px solid var(--border-soft); border-radius: 6px; }
        .stat-num { color: var(--text); font: 700 1.25rem var(--font-mono); margin-bottom: .15rem; }
        .text-up { color: var(--up); } .text-down { color: var(--down); } .text-gold { color: var(--gold); } .text-neutral { color: var(--neutral); } .text-mono { font-family: var(--font-mono); }
        .pipeline-flow, .status-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: .7rem; }
        .status-grid { grid-template-columns: repeat(3, 1fr); }
        .flow-node, .status-box { background: var(--surface-raised); border: 1px solid var(--border); border-radius: 6px; padding: .85rem; }
        .flow-step { color: var(--gold); font: 10px var(--font-mono); text-transform: uppercase; margin-bottom: .25rem; }
        .flow-title, .status-box h4 { color: var(--text); font-size: .86rem; font-weight: 700; margin: 0 0 .35rem; }
        .flow-desc, .status-box p { color: var(--muted); font-size: .75rem; line-height: 1.45; margin: 0; }
        .filter-bar { display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between; background: var(--surface-raised); border: 1px solid var(--border); border-radius: 8px; padding: .85rem 1rem; }
        .data-table-wrap { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
        .table-head { display: grid; grid-template-columns: 1.5fr 1.1fr 1.15fr 1.4fr .9fr 1fr .55fr; gap: .6rem; padding: .6rem .8rem; background: var(--surface-raised); color: var(--muted); font: 10px var(--font-mono); letter-spacing: .05em; text-transform: uppercase; }
        .screener-row { display: grid; grid-template-columns: 1.5fr 1.1fr 1.15fr 1.4fr .9fr 1fr .55fr; gap: .6rem; align-items: center; padding: .7rem .8rem; border-top: 1px solid var(--border-soft); font-size: .78rem; }
        .screener-row:hover { background: var(--surface-hover); }
        .screener-cell { min-width: 0; }
        .screener-cell strong { color: var(--text); }
        .detail-header-controls { display: flex; justify-content: space-between; align-items: end; gap: 1rem; margin: 1rem 0 1.1rem; }
        .detail-grid { display: grid; grid-template-columns: 320px 1fr; gap: 1.1rem; align-items: start; }
        .detail-left, .detail-right { display: flex; flex-direction: column; gap: 1rem; }
        .gauge-card { text-align: center; padding: 1.2rem; }
        .gauge-wrap { width: 160px; height: 82px; margin: .7rem auto .6rem; overflow: hidden; position: relative; }
        .gauge-bg, .gauge-fill { position: absolute; width: 160px; height: 160px; border-radius: 50%; border: 15px solid var(--border-soft); border-bottom-color: transparent; border-right-color: transparent; transform: rotate(-45deg); }
        .gauge-fill { border-color: var(--gold); border-bottom-color: transparent; border-right-color: transparent; }
        .gauge-val { font: 700 1.65rem var(--font-mono); margin-top: -.25rem; }
        .gauge-sub { color: var(--muted); font: 11px var(--font-mono); margin-top: .25rem; }
        .fallback-banner { display: flex; gap: .7rem; background: var(--neutral-soft); border: 1px solid rgba(56,189,248,.3); border-radius: 6px; padding: .75rem .9rem; margin-bottom: .9rem; color: var(--muted); font-size: .8rem; }
        .feature-container { display: flex; flex-direction: column; gap: 1rem; }
        .f-bar-wrap { display: flex; align-items: center; gap: .6rem; width: 100%; }
        .f-bar-label { width: 190px; color: var(--text); font-size: .8rem; flex-shrink: 0; }
        .f-bar-track { flex: 1; height: 7px; background: var(--bg); border-radius: 3px; overflow: hidden; }
        .f-bar-fill { height: 100%; border-radius: 3px; }
        .f-bar-val { width: 56px; text-align: right; font: 11px var(--font-mono); }
        .chart-box { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem; }
        .chart-header { display: flex; justify-content: space-between; align-items: center; gap: .5rem; margin-bottom: .7rem; }
        .chart-note { color: var(--muted); font: 11px/1.45 var(--font-mono); background: var(--surface-raised); border-left: 3px solid var(--gold); border-radius: 4px; padding: .55rem .7rem; }
        .availability-panel { border-color: rgba(245,176,65,.35); background: var(--gold-soft); }
        .availability-panel strong { color: var(--gold); }
        .availability-panel ul { color: var(--muted); margin: .45rem 0 0 1.1rem; }
        .narrative-panel { border-left: 3px solid var(--gold); background: var(--gold-soft); }
        .narrative-panel strong { color: var(--gold); }
        .model-status { display: inline-flex; align-items: center; justify-content: center; gap: .35rem; min-width: 155px; padding: .3rem .55rem; background: var(--up-soft); border: 1px solid rgba(0,229,153,.25); border-radius: 4px; color: var(--up); font: 11px var(--font-mono); }
        .model-status::before { content: ''; width: 6px; height: 6px; border-radius: 50%; background: var(--up); box-shadow: 0 0 6px var(--up); }
        .tape-wrap { margin: 0 -1.4rem 1rem; background: var(--surface); border-bottom: 1px solid var(--border-soft); border-top: 1px solid var(--border-soft); overflow: hidden; white-space: nowrap; height: 30px; display: flex; align-items: center; }
        .tape { display: inline-flex; gap: 2rem; min-width: max-content; animation: tape 45s linear infinite; }
        .tape-wrap:hover .tape { animation-play-state: paused; }
        @keyframes tape { from { transform: translateX(0); } to { transform: translateX(-50%); } }
        .tape-item { color: var(--muted); font: 11px var(--font-mono); display: inline-flex; gap: .35rem; align-items: center; }
        .tape-item b { color: var(--text); } .tape-item .up { color: var(--up); } .tape-item .down { color: var(--down); } .tape-item .neutral { color: var(--neutral); }
        button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible, [role="button"]:focus-visible { outline: 2px solid var(--gold) !important; outline-offset: 2px; }
        div[data-testid="stRadio"] > label, div[data-testid="stSelectbox"] > label { color: var(--muted) !important; font: 10px var(--font-mono) !important; text-transform: uppercase; letter-spacing: .06em; }
        div[data-testid="stRadio"] div[role="radiogroup"] { gap: .2rem; }
        div[data-testid="stRadio"] div[role="radiogroup"] label { color: var(--muted) !important; font: 600 12px var(--font-body) !important; padding: .25rem .65rem; border-radius: 4px; }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover { background: var(--surface-hover); color: var(--text) !important; }
        @media (max-width: 1024px) { .pipeline-flow, .grid-2, .detail-grid { grid-template-columns: 1fr 1fr; } .hero-stats, .stat-grid { grid-template-columns: 1fr 1fr; } .status-grid { grid-template-columns: 1fr; } .f-bar-label { width: 155px; } }
        @media (max-width: 760px) { .block-container { padding: 0 .75rem 2rem; } .topbar-shell, .tape-wrap { margin-left: -.75rem; margin-right: -.75rem; padding-left: .75rem; padding-right: .75rem; } .pipeline-flow, .grid-2, .detail-grid, .hero-stats, .stat-grid { grid-template-columns: 1fr; } .table-head { display: none; } .screener-row { grid-template-columns: 1fr 1fr; } .screener-row .screener-cell:nth-child(4), .screener-row .screener-cell:nth-child(5), .screener-row .screener-cell:nth-child(6) { display: none; } .f-bar-label { width: 130px; font-size: .72rem; } .detail-header-controls { align-items: stretch; flex-direction: column; } }
        .probability-shell { margin: .8rem 0 .55rem; }
        .probability-track { position: relative; height: 18px; border-radius: 999px; background: var(--surface-raised); overflow: visible; border: 1px solid var(--border-focus); }
        .probability-neutral-zone { position: absolute; top: 0; bottom: 0; border-radius: 999px; background: rgba(241,197,109,.2); }
        .probability-interval { position: absolute; top: 3px; bottom: 3px; border-radius: 999px; background: rgba(184,193,255,.45); }
        .probability-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px; background: linear-gradient(90deg, var(--gold-dim), var(--gold)); }
        .probability-marker { position: absolute; top: -5px; width: 3px; height: 28px; border-radius: 3px; background: #fff; box-shadow: 0 0 0 2px rgba(17,19,24,.85); }
        .probability-base-marker { position: absolute; top: -2px; width: 2px; height: 22px; border-radius: 2px; background: var(--amber); }
        .probability-labels { display: flex; justify-content: space-between; gap: .5rem; color: var(--muted); font: .75rem var(--font-mono); margin-top: .45rem; }
        .probability-legend { display: flex; gap: .85rem; flex-wrap: wrap; color: var(--muted); font: .77rem var(--font-mono); margin-top: .6rem; }
        .legend-dot { display: inline-block; width: .58rem; height: .58rem; border-radius: 50%; margin-right: .25rem; }
        .legend-model { background: #fff; }
        .legend-base { background: var(--amber); }
        .legend-neutral { background: rgba(241,197,109,.55); }
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


def _signal(probability: float | None, center: float, threshold: float | None = None) -> tuple[str, str, str]:
    if probability is None:
        return "Unavailable", "neutral", "This call does not have a usable model score."
    decision_threshold = center if threshold is None else threshold
    if probability >= decision_threshold:
        return "Positive", "positive", "The model probability is above its binary decision threshold; confidence reflects its distance from the typical positive-return rate."
    return "Negative", "negative", "The model probability is below its binary decision threshold; confidence reflects its distance from the typical positive-return rate."


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
        threshold = float(bundle.schema.get("prediction_threshold", .5))
        label, tone, _ = _signal(probability, center, threshold)
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
    threshold = float(bundle.schema.get("prediction_threshold", .5))
    probabilities: list[float | None] = []
    statuses: list[str] = []
    signals: list[str] = []
    tones: list[str] = []
    convictions: list[str] = []
    for _, row in table.iterrows():
        probability, status, _ = _predict(bundle, row.to_frame().T)
        signal, tone, _ = _signal(probability, center, threshold)
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


def _bundle_metric(bundle: ArtifactBundle, split: str, name: str) -> float | None:
    if bundle.metrics is None or bundle.metrics.empty or "split" not in bundle.metrics or name not in bundle.metrics:
        return None
    rows = bundle.metrics[bundle.metrics["split"].eq(split)]
    return _as_float(rows.iloc[0][name]) if not rows.empty else None


def _spotlight_features(row: pd.Series) -> list[tuple[str, float | None, str, str]]:
    candidates = [
        ("Q&A historical surprise", _first_feature(row, ("qa_sent_mean_z", "qa_sent_mean")), "var(--up)", "Q&A tone relative to the company’s available history."),
        ("Presentation tone", _first_feature(row, ("pres_sent_mean_z", "pres_net_sentiment")), "var(--neutral)", "Prepared presentation sentiment or historical surprise."),
        ("Q&A / presentation gap", _first_feature(row, ("qa_minus_pres_sent_mean", "sentiment_mismatch_pos")), "var(--gold)", "Difference between unscripted Q&A and prepared remarks."),
        ("Market momentum", _first_feature(row, ("momentum_20d", "momentum_5d")), "var(--neutral)", "Recent market movement used as context."),
    ]
    return [(label, value, color, tip) for label, value, color, tip in candidates if value is not None]


def _feature_bar_html(label: str, value: float | None, color: str, tip: str, scale: float = 2.0) -> str:
    if value is None:
        return ""
    width = min(100.0, max(8.0, abs(value) / scale * 100))
    sign = "+" if value > 0 else ""
    return (
        f'<div class="f-bar-wrap"><span class="f-bar-label">{_escape(label)} '
        f'<span class="infoicon" title="{_escape(tip)}">i</span></span>'
        f'<div class="f-bar-track"><div class="f-bar-fill" style="width:{width:.1f}%;background:{color};"></div></div>'
        f'<span class="f-bar-val" style="color:{color}">{sign}{value:.2f}</span></div>'
    )


def _render_ticker(table: pd.DataFrame) -> None:
    if table.empty:
        return
    items: list[str] = []
    for _, row in table.head(12).iterrows():
        tone = str(row.get("_tone", "neutral"))
        css = "up" if tone == "positive" else "down" if tone == "negative" else "neutral"
        signal = str(row.get("_signal", "Unavailable"))
        probability = _format_percent(_as_float(row.get("_probability")))
        date = pd.to_datetime(row.get("call_datetime"), errors="coerce")
        quarter = row.get("quarter")
        date_text = str(quarter) if quarter not in (None, "", "nan") else (date.strftime("%b %Y") if pd.notna(date) else "—")
        items.append(
            f'<span class="tape-item"><b>{_escape(row.get("symbol", "—"))}</b> '
            f'<span class="{css}">{_escape(signal)} {probability}</span> ({date_text})</span>'
        )
    tape = "".join(items)
    st.markdown(f'<div class="tape-wrap"><div class="tape">{tape}{tape}</div></div>', unsafe_allow_html=True)


def _open_mockup_call(row: pd.Series) -> None:
    st.session_state["selected_call_key"] = _call_key(str(row["symbol"]), row["call_datetime"])
    st.session_state["_next_mockup_view"] = "Call Detail Terminal"
    st.rerun()


def _render_mockup_overview(bundle: ArtifactBundle, table: pd.DataFrame | None = None) -> None:
    table = table if table is not None else _prepare_table(bundle)
    base_rate = _base_rate(bundle)
    center = base_rate or float(bundle.schema.get("prediction_threshold", .5))
    validated = table[table["_status"].isin(VALIDATED_STATUSES) & table["_probability"].notna()]
    spotlight = validated.copy()
    if spotlight.empty:
        spotlight = table[table["_probability"].notna()].copy()
    if not spotlight.empty:
        spotlight["_distance"] = (spotlight["_probability"] - center).abs()
        spotlight_row = spotlight.sort_values(["_distance", "call_datetime"], ascending=[False, False]).iloc[0]
    else:
        spotlight_row = None

    auc = _bundle_metric(bundle, "walk_forward_aggregate", "auc")
    auc_label = "Walk-forward AUC"
    if auc is None:
        auc = _bundle_metric(bundle, "final_holdout", "auc")
        auc_label = "Latest holdout AUC"
    mcc = _bundle_metric(bundle, "walk_forward_aggregate", "mcc")
    company_count = table["symbol"].nunique() if "symbol" in table else 0
    metric_cards = (
        f'<div class="hero-stats">'
        f'<div class="stat-box"><div class="stat-num">{len(table):,}</div><div class="stat-lbl">Events in artifact <span class="infoicon" title="Calls available in the active feature artifact.">i</span></div></div>'
        f'<div class="stat-box"><div class="stat-num">{company_count:,}</div><div class="stat-lbl">Companies covered</div></div>'
        f'<div class="stat-box"><div class="stat-num text-up">{_metric_text(pd.Series({"auc": auc}), "auc")}</div><div class="stat-lbl">{_escape(auc_label)} <span class="infoicon" title="Area under the ROC curve. 0.50 is close to no-information ranking.">i</span></div></div>'
        f'<div class="stat-box"><div class="stat-num text-gold">{_metric_text(pd.Series({"mcc": mcc}), "mcc")}</div><div class="stat-lbl">MCC <span class="infoicon" title="A balanced directional-quality metric. Higher is better; zero is roughly no correlation.">i</span></div></div>'
        f'</div>'
    )
    st.markdown(
        f'<div class="card-raised"><div class="eyebrow">Executive dialogue &amp; return drift</div>'
        f'<h1 style="font:700 1.65rem/1.2 var(--font-display);margin:.25rem 0 .55rem;">Predicting five-session post-earnings abnormal returns through linguistic drift</h1>'
        f'<p class="app-subtitle">This system compares earnings presentations and analyst Q&amp;A with company history and market context, then estimates the chance of a positive five-session abnormal return. It is a research signal, not a price target or trade recommendation.</p>'
        f'{metric_cards}</div>',
        unsafe_allow_html=True,
    )
    cta_left, cta_right = st.columns([1, 1])
    with cta_left:
        if st.button("Launch Screener →", type="primary", use_container_width=True, key="overview-launch-screener"):
            st.session_state["_next_mockup_view"] = "Screener & Signals"
            st.rerun()
    with cta_right:
        if st.button("Compare Architecture Lineage", use_container_width=True, key="overview-lineage"):
            st.session_state["_next_mockup_view"] = "Model Reliability & Lineage"
            st.rerun()

    st.markdown('<div class="section-header"><h2>Signal spotlight</h2><span class="sub">Highest separation from the active artifact base rate</span></div>', unsafe_allow_html=True)
    if spotlight_row is None:
        st.info("No scored calls are available for a spotlight in the active artifact.")
    else:
        timestamp = pd.to_datetime(spotlight_row["call_datetime"], errors="coerce")
        date_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Date unavailable"
        signal = str(spotlight_row["_signal"])
        tone = str(spotlight_row["_tone"])
        feature_html = "".join(_feature_bar_html(*item) for item in _spotlight_features(spotlight_row)[:3])
        st.markdown(
            f'<div class="card"><div class="eyebrow" style="color:var(--neutral)">Signal spotlight</div>'
            f'<div style="display:flex;justify-content:space-between;gap:.8rem;align-items:flex-start;">'
            f'<div><div class="card-title" style="font:600 1rem var(--font-mono);">{_escape(spotlight_row.get("symbol", "—"))} — {_escape(spotlight_row.get("company_name", "Unknown"))}</div>'
            f'<div class="card-meta">{_escape(date_text)} · {_escape(_phase_label(spotlight_row))} · {_status_pill(str(spotlight_row["_status"]))}</div></div>'
            f'<span class="badge {"badge-up" if tone == "positive" else "badge-down" if tone == "negative" else "badge-neutral"}">{_escape(signal)} · {_format_percent(_as_float(spotlight_row["_probability"]))}</span></div>'
            f'<p style="color:var(--muted);border-left:2px solid var(--gold);padding-left:.65rem;margin:.8rem 0;">'
            f'<strong>Model rationale:</strong> { _escape(_signal(_as_float(spotlight_row["_probability"]), center, float(bundle.schema.get("prediction_threshold", .5)))[2]) }</p>'
            f'<div class="feature-container">{feature_html}</div></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open full terminal breakdown →", use_container_width=True, key="overview-open-spotlight"):
            _open_mockup_call(spotlight_row)

    st.markdown('<div class="section-header"><h2>End-to-end ML pipeline architecture</h2><span class="sub">How transcript signals become a directional estimate</span></div>', unsafe_allow_html=True)
    stages = [
        ("Stage 01", "Transcript ingestion & splitting", "Presentation and analyst Q&A are separated before feature extraction."),
        ("Stage 02", "Sentence NLP features", "Sentiment, entropy, sentence position, and Q&A behavior are summarized."),
        ("Stage 03", "Historical normalization", "Language measures are compared with the company’s available history where supported."),
        ("Stage 04", "Direction classifier", "The active model estimates the probability of a positive five-session abnormal return."),
    ]
    stage_html = "".join(f'<div class="flow-node"><div class="flow-step">{step}</div><div class="flow-title">{title}</div><div class="flow-desc">{desc}</div></div>' for step, title, desc in stages)
    st.markdown(f'<div class="pipeline-flow">{stage_html}</div>', unsafe_allow_html=True)

    counts = table["_status"].value_counts().to_dict()
    status_cards = [
        ("Validated", "Out-of-sample folds", counts.get("Out-of-sample holdout", 0) + counts.get("Walk-forward validated", 0), "Stored validation provenance is available for this call."),
        ("Exploratory", "Retrospective archive", counts.get("Retrospective inference", 0), "The model can score the row, but it was not independently held out."),
        ("Unavailable", "Filtered events", counts.get("Unavailable", 0), "No usable score is available for the active model."),
    ]
    status_html = "".join(
        f'<div class="status-box"><h4><span class="badge {"badge-up" if label == "Validated" else "badge-neutral" if label == "Exploratory" else "badge-muted"}">{label}</span> {title}</h4><div class="stat-num">{count:,}</div><p>{desc}</p></div>'
        for label, title, count, desc in status_cards
    )
    st.markdown('<div class="section-header"><h2>Dataset coverage &amp; validation discipline</h2><span class="sub">Every visible score carries its provenance</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="status-grid">{status_html}</div>', unsafe_allow_html=True)


def _render_mockup_screener(bundle: ArtifactBundle, bundles: tuple[ArtifactBundle, ...], table: pd.DataFrame | None = None) -> None:
    table = table if table is not None else _prepare_table(bundle)
    base_rate = _base_rate(bundle) or .5
    st.markdown('<div class="eyebrow">Market intelligence</div><div class="app-title">Screener &amp; Signals</div><p class="app-subtitle">Search earnings calls by company, inspect the active model signal, and keep validation status in view.</p>', unsafe_allow_html=True)
    filter_left, filter_right = st.columns([3.6, 1.4])
    with filter_left:
        dir_filter = st.radio("Signal direction", ["All signals", "Up only", "Down only"], horizontal=True, key="mockup_dir_filter")
    with filter_right:
        query = st.text_input("Search ticker or company", placeholder="Search ticker or company", key="mockup_screener_search")
    filter_a, filter_b = st.columns([1, 1])
    with filter_a:
        conf_filter = st.radio("Confidence band", ["All", "High", "Medium", "Low"], horizontal=True, key="mockup_conf_filter")
    with filter_b:
        status_filter = st.radio("Coverage status", ["Validated", "Exploratory", "Unavailable", "All"], horizontal=True, key="mockup_status_filter")

    filtered = table
    if query:
        q = query.upper()
        names = filtered.get("company_name", pd.Series(index=filtered.index, dtype=str)).astype(str)
        filtered = filtered[filtered["symbol"].str.upper().str.contains(q, na=False) | names.str.upper().str.contains(q, na=False)]
    if dir_filter == "Up only":
        filtered = filtered[filtered["_signal"].eq("Positive")]
    elif dir_filter == "Down only":
        filtered = filtered[filtered["_signal"].eq("Negative")]
    if conf_filter != "All":
        filtered = filtered[filtered["_conviction"].eq(conf_filter)]
    if status_filter == "Validated":
        filtered = filtered[filtered["_status"].isin(VALIDATED_STATUSES)]
    elif status_filter == "Exploratory":
        filtered = filtered[filtered["_status"].eq("Retrospective inference")]
    elif status_filter == "Unavailable":
        filtered = filtered[filtered["_status"].eq("Unavailable")]

    if filtered.empty:
        st.warning("No calls match those filters.")
        return

    page_size = 14
    page_count = max(1, int(np.ceil(len(filtered) / page_size)))
    current_page = int(st.session_state.get("mockup_screener_page", 1))
    if current_page > page_count:
        st.session_state["mockup_screener_page"] = 1
        current_page = 1
    current_page = int(st.number_input("Page", min_value=1, max_value=page_count, value=current_page, step=1, key="mockup_screener_page"))
    visible = filtered.iloc[(current_page - 1) * page_size: current_page * page_size]
    st.markdown(f'<div style="display:flex;justify-content:space-between;color:var(--muted);font:11px var(--font-mono);margin:.7rem 0;"><span>Showing {len(visible)} of {len(filtered)} matching calls</span><span>Model: {_escape(bundle.display_name)}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="data-table-wrap"><div class="table-head"><div>Company &amp; ticker</div><div>Earnings call</div><div>Dataset status</div><div>Active model signal</div><div>Confidence</div><div>Historical 5D outcome</div><div>Action</div></div>', unsafe_allow_html=True)
    for index, (_, row) in enumerate(visible.iterrows()):
        timestamp = pd.to_datetime(row["call_datetime"], errors="coerce")
        date_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Undated"
        target = str(bundle.schema.get("target_column", "abnormal_return_5d"))
        actual = _as_float(row.get(target))
        actual_html = "—" if actual is None else f'<span class="{"text-up" if actual > 0 else "text-down"}">{_format_percent(actual, signed=True)}</span>'
        tone = str(row["_tone"])
        signal_html = f'<span class="badge {"badge-up" if tone == "positive" else "badge-down" if tone == "negative" else "badge-neutral"}">{_escape(row["_signal"])} · {_format_percent(_as_float(row["_probability"]))}</span>'
        status = _status_pill(str(row["_status"]))
        cols = st.columns([1.5, 1.1, 1.15, 1.4, .9, 1, .55])
        with cols[0]:
            st.markdown(f'<div class="screener-cell"><strong>{_escape(row.get("company_name", row["symbol"]))}</strong><br><span class="card-meta">{_escape(row["symbol"])}</span></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="screener-cell text-mono">{_escape(date_text)}<br><span class="card-meta">{_escape(_phase_label(row))}</span></div>', unsafe_allow_html=True)
        with cols[2]:
            st.markdown(status, unsafe_allow_html=True)
        with cols[3]:
            st.markdown(signal_html, unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f'<span class="text-mono">{_escape(row["_conviction"])}</span>', unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f'<span class="text-mono">{actual_html}</span>', unsafe_allow_html=True)
        with cols[6]:
            if st.button("View", key=f"mockup-view-{_call_key(str(row['symbol']), row['call_datetime'])}", use_container_width=True):
                _open_mockup_call(row)
    st.markdown('</div>', unsafe_allow_html=True)
    prev_col, page_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button("← Previous", disabled=current_page <= 1, key="mockup-prev", use_container_width=True):
            st.session_state["mockup_screener_page"] = current_page - 1
            st.rerun()
    with page_col:
        st.markdown(f'<div style="text-align:center;color:var(--muted);font:11px var(--font-mono);padding:.45rem;">Page {current_page} of {page_count}</div>', unsafe_allow_html=True)
    with next_col:
        if st.button("Next →", disabled=current_page >= page_count, key="mockup-next", use_container_width=True):
            st.session_state["mockup_screener_page"] = current_page + 1
            st.rerun()


def _render_mockup_detail(bundles: tuple[ArtifactBundle, ...], bundle: ArtifactBundle, table: pd.DataFrame | None = None) -> None:
    if st.button("← Back to Screener & Signals", key="mockup-back-screener"):
        st.session_state["_next_mockup_view"] = "Screener & Signals"
        st.rerun()
    table = table if table is not None else _prepare_table(bundle)
    if table.empty:
        st.warning("This artifact has no calls to display.")
        return
    selected_key = st.session_state.get("selected_call_key")
    selected_symbol = str(selected_key).split("|", 1)[0] if selected_key else str(table.iloc[0]["symbol"])
    symbols = sorted(table["symbol"].unique().tolist())
    if selected_symbol not in symbols:
        selected_symbol = symbols[0]
    company_names = table.groupby("symbol")["company_name"].first().to_dict() if "company_name" in table else {}
    company_label = lambda symbol: f"{company_names.get(symbol, symbol)} ({symbol})"
    co_col, call_col = st.columns([1, 1])
    with co_col:
        selected_symbol = st.selectbox("Company", symbols, index=symbols.index(selected_symbol), format_func=company_label, key="mockup_detail_company")
    company_calls = table[table["symbol"] == selected_symbol].sort_values("call_datetime", ascending=False)
    call_indices = list(company_calls.index)
    call_labels = []
    for index in call_indices:
        row = company_calls.loc[index]
        stamp = pd.to_datetime(row["call_datetime"], errors="coerce")
        call_labels.append(f"{stamp.strftime('%b %d, %Y · %H:%M') if pd.notna(stamp) else 'Undated'} · {row['_status']}")
    default_index = 0
    if selected_key:
        for position, index in enumerate(call_indices):
            if _call_key(selected_symbol, company_calls.loc[index]["call_datetime"]) == selected_key:
                default_index = position
                break
    with call_col:
        selected_call_label = st.selectbox("Call", call_labels, index=default_index, key=f"mockup_detail_call_{selected_symbol}")
    selected_index = call_indices[call_labels.index(selected_call_label)]
    selected = company_calls.loc[[selected_index]].copy()
    row = selected.iloc[0]
    st.session_state["selected_call_key"] = _call_key(str(row["symbol"]), row["call_datetime"])

    probability = _as_float(row["_probability"])
    base_rate = _base_rate(bundle)
    center = base_rate or float(bundle.schema.get("prediction_threshold", .5))
    signal, tone, explanation = _signal(probability, center, float(bundle.schema.get("prediction_threshold", .5)))
    status = str(row["_status"])
    timestamp = pd.to_datetime(row["call_datetime"], errors="coerce")
    date_text = timestamp.strftime("%b %d, %Y · %H:%M") if pd.notna(timestamp) else "Date unavailable"
    st.markdown(f'<div class="detail-header-controls"><div><div class="eyebrow">{_escape(row["symbol"])} · {_escape(_phase_label(row))} · {_status_pill(status)}</div><div class="app-title">{_escape(row.get("company_name", row["symbol"]))} — {date_text}</div><div class="card-meta">Five-session abnormal-return research signal · {_escape(_status_description(status))}</div></div></div>', unsafe_allow_html=True)

    if status == "Unavailable":
        st.markdown('<div class="fallback-banner"><span class="text-neutral">i</span><div><strong>No prediction available in the active model.</strong><br>Choose another artifact or inspect the call as an unavailable record. No probability is being inferred.</div></div>', unsafe_allow_html=True)
    elif status == "Retrospective inference":
        st.markdown('<div class="fallback-banner"><span class="text-gold">!</span><div><strong>Exploratory archive record.</strong><br>This score is a retrospective model inference, not an out-of-sample validation result.</div></div>', unsafe_allow_html=True)

    detail_left, detail_right = st.columns([.82, 1.7])
    with detail_left:
        pct = max(0, min(100, (probability or 0) * 100))
        gauge_deg = -45 + ((probability or 0) * 180)
        gauge_value = _format_percent(probability)
        gauge_color = "var(--up)" if tone == "positive" else "var(--down)" if tone == "negative" else "var(--neutral)"
        st.markdown(
            f'<div class="card gauge-card" role="img" aria-label="Model probability {gauge_value}, direction {signal}, confidence {row["_conviction"]}">'
            f'<div class="eyebrow">Active model direction signal</div><div class="gauge-wrap"><div class="gauge-bg"></div><div class="gauge-fill" style="transform:rotate({gauge_deg:.1f}deg);border-color:{gauge_color};border-bottom-color:transparent;border-right-color:transparent;"></div></div>'
            f'<div class="gauge-val" style="color:{gauge_color}">{gauge_value}</div><div class="gauge-sub">{_escape(signal)} · {_escape(row["_conviction"])} confidence</div><div style="margin-top:.65rem;">{_status_pill(status)}</div>'
            f'<div class="card-meta" style="margin-top:.7rem;">Base rate {_format_percent(base_rate)} · Difference {_format_percent((probability - center) if probability is not None else None, signed=True)}</div></div>',
            unsafe_allow_html=True,
        )
        target = str(bundle.schema.get("target_column", "abnormal_return_5d"))
        actual = _as_float(row.get(target))
        actual_text = "Unavailable" if actual is None else ("Positive" if actual > 0 else "Negative")
        actual_class = "text-up" if actual is not None and actual > 0 else "text-down" if actual is not None else ""
        st.markdown(f'<div class="card"><div class="eyebrow">Market outcome · historical only</div><div class="stat-num {actual_class}">{actual_text}</div><div class="card-meta">{_format_percent(actual, signed=True)} abnormal return over five sessions. This is not a prediction.</div></div>', unsafe_allow_html=True)
        pres_count = _first_feature(row, ("pres_n_sentences",))
        qa_count = _first_feature(row, ("qa_n_sentences",))
        st.markdown(f'<div class="card"><div class="eyebrow">Transcript metadata</div><div style="display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem;"><span>Presentation sentences</span><strong class="text-mono">{_format_number(pres_count)}</strong></div><div style="display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem;margin-top:.45rem;"><span>Q&amp;A sentences</span><strong class="text-mono">{_format_number(qa_count)}</strong></div><div style="display:flex;justify-content:space-between;color:var(--muted);font-size:.8rem;margin-top:.45rem;"><span>Coverage status</span><strong>{_escape(status)}</strong></div></div>', unsafe_allow_html=True)

    with detail_right:
        st.markdown('<div class="card"><div class="section-header" style="margin:0 0 1rem;"><h2>Why the model responded</h2><span class="sub">Active artifact features</span></div><div class="feature-container">', unsafe_allow_html=True)
        feature_html = "".join(_feature_bar_html(*item) for item in _spotlight_features(row))
        if feature_html:
            st.markdown(feature_html, unsafe_allow_html=True)
        else:
            st.info("Human-readable feature evidence is unavailable for this artifact.")
        st.markdown('</div><p class="card-meta" style="margin:.9rem 0 0;">Feature associations describe what the model used; they are not proof of causation.</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-box"><div class="chart-header"><div><div class="eyebrow" style="margin:0;">Market event window</div><div class="card-meta">Normalized around the earnings call and five-session evaluation window</div></div><span class="badge-muted">HISTORICAL SIMULATION</span></div>', unsafe_allow_html=True)
        event_available = _render_event_chart(row)
        if not event_available:
            st.markdown('<div class="chart-note">Price and benchmark series are not included in the current offline artifact. The chart will appear when source-linked event-window data is added.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Compare model outputs", expanded=False):
        _render_call_comparison(_collect_model_results(bundles, str(row["symbol"]), row["call_datetime"]))
    with st.expander("Source-linked transcript evidence", expanded=False):
        evidence = row.get("transcript_evidence")
        if evidence not in (None, "", "nan") and not (isinstance(evidence, float) and pd.isna(evidence)):
            st.write(evidence)
        else:
            st.info("Transcript excerpts are not stored in the current artifact. No generated quote is shown.")
    with st.expander("Company call history", expanded=False):
        history = table[table["symbol"] == str(row["symbol"])].sort_values("call_datetime", ascending=False)
        view = history[[c for c in ("symbol", "company_name", "call_datetime", "_signal", "_probability", "_status") if c in history]].copy()
        if "call_datetime" in view:
            view["call_datetime"] = view["call_datetime"].dt.strftime("%b %d, %Y · %H:%M")
        st.dataframe(view.rename(columns={"_signal":"Direction","_probability":"Probability","_status":"Validation"}), use_container_width=True, hide_index=True)
    with st.expander("Technical details", expanded=False):
        explanation_table = _model_explanation(bundle, selected)
        if not explanation_table.empty:
            st.dataframe(explanation_table.head(12), use_container_width=True, hide_index=True)
        st.download_button("Download selected call data", selected.to_csv(index=False).encode("utf-8"), file_name=f"{row['symbol']}_earnings_call.csv", mime="text/csv")

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
        direction_filter = st.selectbox("Direction", ["All directions", "Positive", "Negative"], key="calls_direction_filter")
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
    st.markdown('<div class="eyebrow">Model evaluation suite</div><div class="app-title">Model Reliability &amp; Lineage</div>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">Compare architectural iterations, validation lineage, and the limits of the evidence without treating one metric or one holdout as a definitive winner.</p>', unsafe_allow_html=True)
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

    st.subheader("Evaluation table")
    evaluation_rows = []
    for model, title, description, badge in model_config:
        walk = _comparison_row(metrics, model, "walk_forward_aggregate")
        holdout = _comparison_row(metrics, model, "final_holdout")
        evaluation_rows.append({
            "Model": title,
            "Type": badge,
            "Walk-forward AUC": _metric_text(walk, "auc"),
            "Latest holdout AUC": _metric_text(holdout, "auc"),
            "Walk-forward Brier": _metric_text(walk, "brier"),
            "95% AUC range": f"{_metric_text(walk, 'auc_ci_lower_95')}–{_metric_text(walk, 'auc_ci_upper_95')}" if walk is not None and "auc_ci_lower_95" in walk else "—",
            "Events": str(_as_float(walk["n"]) if walk is not None and "n" in walk else "—"),
        })
    st.dataframe(pd.DataFrame(evaluation_rows), use_container_width=True, hide_index=True)

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
        xaxis={"title": "AUC · higher is better", "range": [0, 1], "fixedrange": True, "gridcolor": "#303642", "zeroline": False},
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
    signal, tone, explanation = _signal(probability, center, float(bundle.schema.get("prediction_threshold", .5)))
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
        ("02 · Signal", "See the model’s Positive or Negative prediction."),
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

    if "_next_mockup_view" in st.session_state:
        st.session_state["mockup_view"] = st.session_state.pop("_next_mockup_view")
    if "mockup_view" not in st.session_state:
        st.session_state["mockup_view"] = "Overview"

    walk_forward_auc = _bundle_metric(active_bundle, "walk_forward_aggregate", "auc")
    topbar_metric_label = "Walk-forward AUC"
    if walk_forward_auc is None:
        walk_forward_auc = _bundle_metric(active_bundle, "final_holdout", "auc")
        topbar_metric_label = "Latest holdout AUC"
    st.markdown(
        '<div class="topbar-shell"><div class="brand-row">'
        '<span class="brand-mark">EC</span>'
        '<span class="brand-title">Earnings Call Intelligence</span>'
        '<span class="brand-badge">research prototype</span>'
        '<span class="brand-meta">direction · evidence · reliability</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    nav_col, model_col, status_col = st.columns([5.6, 2.8, 1.8], vertical_alignment="center")
    with nav_col:
        navigation = st.radio(
            "Navigate",
            ["Overview", "Screener & Signals", "Call Detail Terminal", "Model Reliability & Lineage"],
            horizontal=True,
            key="mockup_view",
            label_visibility="collapsed",
        )
    with model_col:
        selected_model_label = st.selectbox(
            "Active model",
            labels,
            key="active_model_label",
            label_visibility="collapsed",
            help="This model controls the probabilities, provenance labels, and feature values shown throughout the app.",
        )
    with status_col:
        auc_text = _metric_text(pd.Series({"auc": walk_forward_auc}), "auc")
        st.markdown(f'<div class="model-status"><span>{_escape(topbar_metric_label)}</span><strong>{auc_text}</strong></div>', unsafe_allow_html=True)

    active_bundle = label_to_bundle[selected_model_label]
    active_table = _prepare_table(active_bundle)
    _render_ticker(active_table)

    if navigation == "Overview":
        _render_mockup_overview(active_bundle, active_table)
    elif navigation == "Screener & Signals":
        _render_mockup_screener(active_bundle, bundles, active_table)
    elif navigation == "Call Detail Terminal":
        _render_mockup_detail(bundles, active_bundle, active_table)
    else:
        _render_reliability()


if __name__ == "__main__":
    main()
