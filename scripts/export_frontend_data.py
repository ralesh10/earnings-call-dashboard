"""Export validated artifact bundles into the static frontend data contract."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import (  # noqa: E402
    VALIDATED_STATUSES,
    _base_rate,
    _comparison_row,
    _conviction,
    _evidence_sections,
    _load_bundles,
    _load_comparison_manifest,
    _load_comparison_metrics,
    _predict,
    _signal,
)
from artifact_contract import discover_artifact_dirs  # noqa: E402


def _number(value: Any) -> float | int | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(value):
        return None
    return int(value) if value.is_integer() else value


def _text(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    value = str(value)
    return None if value in {"", "nan", "NaT"} else value


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except ValueError:
            pass
    return value


def _model_key(bundle: Any) -> str:
    if "xgboost" in bundle.display_name.lower() or bundle.schema.get("model_family") == "xgboost":
        return "sentence_hist"
    if "logistic" in bundle.display_name.lower() or bundle.schema.get("model_family") == "logistic_regression":
        return "finbert"
    return bundle.model_version


def _format_value(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "Unavailable"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}{suffix}"


def _feature_bars(row: pd.Series) -> list[dict[str, Any]]:
    candidates = [
        ("Q&A historical surprise", ("qa_sent_mean_z", "qa_sent_mean"), "up", "Current Q&A tone relative to the company’s available history.", "score"),
        ("Presentation historical surprise", ("pres_sent_mean_z", "pres_net_sentiment"), "neutral", "Prepared presentation tone, compared with company history when a z-score is available.", "score"),
        ("Q&A / presentation gap", ("qa_minus_pres_sent_mean", "sentiment_mismatch_pos"), "gold", "Difference between Q&A and prepared remarks, measured on the sentiment-score scale.", "score"),
        ("Q&A sentiment slope", ("qa_slope_z", "qa_slope"), "up", "Change in Q&A tone across the call; shown as a historical z-score when available.", "score"),
        ("Evasion index", ("evasion_index",), "neutral", "Bounded index from 0 to 1; higher values indicate more evasive language in the source artifact.", "index"),
        ("Market momentum", ("momentum_20d", "momentum_5d"), "neutral", "Recent market movement used as context, shown as a percentage change.", "percent"),
    ]
    bars: list[dict[str, Any]] = []
    for label, names, color, description, value_kind in candidates:
        value = None
        selected_name = None
        for name in names:
            if name in row.index:
                value = _number(row.get(name))
                if value is not None:
                    selected_name = name
                    break
        if value is None:
            continue
        width = min(100, max(10, abs(float(value)) / 2 * 100))
        if value_kind == "index":
            display = f"{float(value):.2f} / 1"
            unit = "0–1"
        elif value_kind == "percent":
            display = f"{float(value):+.1%}"
            unit = "%"
        else:
            unit = "σ" if selected_name and selected_name.endswith("_z") else "score"
            display = _format_value(float(value), f" {unit}" if unit == "score" else unit)
        bars.append({"label": label, "value": value, "display": display, "unit": unit, "color": color, "width": width, "description": description})
    return bars[:4]


def _feature_groups(row: pd.Series) -> list[dict[str, Any]]:
    groups = []
    for title, summary, description, details in _evidence_sections(row):
        readable_details = []
        for detail in details:
            detail = re.sub(r"(Historical surprise:\s*[+-]?\d+(?:\.\d+)?)$", r"\1σ", detail)
            detail = re.sub(r"^(Presentation sentiment|Q&A sentiment|Presentation/Q&A gap|Sentiment slope):\s*([+-]?\d+(?:\.\d+)?)$", r"\1: \2 score", detail)
            readable_details.append(detail)
        groups.append({"title": title, "summary": summary, "description": description, "details": readable_details})
    return groups


def _prediction_for(bundle: Any, row: pd.Series) -> dict[str, Any]:
    frame = row.to_frame().T
    probability, status, source = _predict(bundle, frame)
    center = _base_rate(bundle) or float(bundle.schema.get("prediction_threshold", 0.5))
    signal, tone, explanation = _signal(probability, center)
    target = str(bundle.schema.get("target_column", "abnormal_return_5d"))
    actual = _number(row.get(target))
    difference = None if probability is None else probability - center
    confidence_description = (
        "Low confidence: the model is within 5 percentage points of the typical positive-return rate."
        if difference is not None and abs(difference) < .05 else
        "Medium confidence: the model is 5–15 percentage points from the typical positive-return rate."
        if difference is not None and abs(difference) < .15 else
        "High confidence: the model is at least 15 percentage points from the typical positive-return rate."
        if difference is not None else
        "Confidence is unavailable because no probability was produced."
    )
    return {
        "prob": probability,
        "status": status,
        "statusDescription": {
            "Out-of-sample holdout": "This call was kept outside the training data used for the stored prediction.",
            "Walk-forward validated": "This prediction was evaluated using only information available before the call’s evaluation period.",
            "Retrospective inference": "The model scored this historical call, but the call was not independently held out for validation.",
            "Unavailable": "No usable model score is available for this call.",
        }.get(status, "Validation provenance was not provided by the artifact."),
        "source": source,
        "signal": signal,
        "tone": tone,
        "explanation": explanation,
        "confidence": _conviction(probability, center).upper(),
        "baseRate": center,
        "differenceFromBaseRate": difference,
        "confidenceDescription": confidence_description,
        "statusCategory": "Validated" if status in VALIDATED_STATUSES else "Exploratory" if status == "Retrospective inference" else "Unavailable",
        "actualReturn": actual,
        "featureBars": _feature_bars(row),
        "featureGroups": _feature_groups(row),
    }


def _call_record(bundle: Any, rows_by_key: dict[tuple[str, str], dict[str, Any]], row: pd.Series) -> None:
    symbol = str(row["symbol"])
    timestamp = pd.to_datetime(row["call_datetime"], errors="coerce")
    key = (symbol, timestamp.isoformat())
    existing = rows_by_key.setdefault(key, {
        "id": f"{symbol}|{timestamp.isoformat()}",
        "sym": symbol,
        "co": _text(row.get("company_name")) or symbol,
        "year": int(_number(row.get("year")) or timestamp.year),
        "q": int(_number(row.get("quarter")) or 0),
        "date": timestamp.strftime("%Y-%m-%d") if pd.notna(timestamp) else "",
        "datetime": timestamp.isoformat() if pd.notna(timestamp) else None,
        "timing": _text(row.get("call_phase")) or ("Before open" if pd.notna(timestamp) and timestamp.hour < 12 else "After close"),
        "ret": _number(row.get(bundle.schema.get("target_column", "abnormal_return_5d"))),
        "rawReturn": _number(row.get("return")),
        "presLen": f"{int(_number(row.get('pres_n_sentences')) or 0):,} sentences" if _number(row.get("pres_n_sentences")) is not None else "Unavailable",
        "qaLen": f"{int(_number(row.get('qa_n_sentences')) or 0):,} Q&A sentences" if _number(row.get("qa_n_sentences")) is not None else "Unavailable",
        "priceSeries": _json_value(row.get("price_series")),
        "benchmarkSeries": _json_value(row.get("benchmark_series")),
        "models": {},
    })
    prediction = _prediction_for(bundle, row)
    model_key = _model_key(bundle)
    existing["models"][model_key] = prediction
    if model_key == "sentence_hist":
        existing["ret"] = prediction["actualReturn"]
    existing["status"] = existing["models"].get("sentence_hist", {}).get("status", "Unavailable").lower().replace(" ", "_")


def _metric_record(metrics: pd.DataFrame | None, model: str, title: str, description: str, badge: str) -> dict[str, Any]:
    walk = _comparison_row(metrics, model, "walk_forward_aggregate")
    holdout = _comparison_row(metrics, model, "final_holdout")

    def value(row: pd.Series | None, name: str) -> Any:
        return _number(row.get(name)) if row is not None else None

    return {
        "key": model,
        "title": title,
        "description": description,
        "badge": badge,
        "walkForwardAuc": value(walk, "auc"),
        "holdoutAuc": value(holdout, "auc"),
        "walkForwardBrier": value(walk, "brier"),
        "holdoutBrier": value(holdout, "brier"),
        "mcc": value(walk, "mcc") or value(holdout, "mcc"),
        "ciLower": value(walk, "auc_ci_lower_95") or value(walk, "auc_lower_95"),
        "ciUpper": value(walk, "auc_ci_upper_95") or value(walk, "auc_upper_95"),
        "events": value(walk, "n") or value(holdout, "n"),
        "positiveRate": value(walk, "positive_rate") or value(holdout, "positive_rate"),
    }


def main() -> None:
    bundles, errors = _load_bundles(tuple(str(path) for path in discover_artifact_dirs()))
    if errors or not bundles:
        raise SystemExit("Unable to load validated bundles: " + "; ".join(errors))
    bundles = tuple(sorted(bundles, key=lambda item: (item.is_experimental, item.display_label.lower())))

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for bundle in bundles:
        for _, row in bundle.feature_table.iterrows():
            _call_record(bundle, rows_by_key, row)
    calls = sorted(rows_by_key.values(), key=lambda item: item.get("datetime") or "", reverse=True)

    metrics = _load_comparison_metrics()
    model_config = [
        ("sentence_plus_historical_xgboost_depth1_trees100", "Rich XGBoost", "Language, sentence-position, and historical context features.", "Richer candidate"),
        ("original_logistic", "Original Logistic", "Smaller sentiment model used as the reference.", "Reference"),
        ("market_only_logistic", "Market-only baseline", "Recent market behavior without transcript language.", "Context baseline"),
    ]
    model_records = []
    for key, title, description, badge in model_config:
        model_records.append(_metric_record(metrics, key, title, description, badge))

    selected = next((bundle for bundle in bundles if bundle.predictions is not None), bundles[0])
    output = {
        "version": 1,
        "generatedAt": pd.Timestamp.now(tz="UTC").isoformat(),
        "defaultModel": _model_key(selected),
        "models": [
            {"key": _model_key(bundle), "label": bundle.display_label, "displayName": bundle.display_name, "experimental": bundle.is_experimental, "baseRate": _base_rate(bundle), "events": len(bundle.feature_table), "companies": int(bundle.feature_table["symbol"].nunique())}
            for bundle in bundles
        ],
        "calls": calls,
        "reliability": {"models": model_records, "manifest": _json_value(_load_comparison_manifest())},
    }
    output_path = ROOT / "frontend" / "data" / "app-data.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    print(f"Exported {len(calls):,} calls and {len(model_records)} reliability records to {output_path}")


if __name__ == "__main__":
    main()
