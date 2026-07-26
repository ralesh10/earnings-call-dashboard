"""Presentation dashboard for validated earnings-call model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from artifact_contract import (
    ArtifactBundle,
    ArtifactValidationError,
    discover_artifact_dirs,
    load_artifact_bundle,
)


st.set_page_config(page_title="Earnings Call Intelligence", page_icon="◈", layout="wide")


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1020; color: #e8edf7; }
        [data-testid="stSidebar"] { background: #11182b; border-right: 1px solid #263352; }
        .block-container { max-width: 1400px; padding-top: 2rem; }
        .eyebrow { color: #7dd3fc; letter-spacing: .14em; text-transform: uppercase; font-size: .72rem; font-weight: 700; }
        .hero { padding: 1.5rem 1.7rem; border: 1px solid #2c3d62; border-radius: 18px; background: linear-gradient(135deg, #17233d, #10182c); margin: .5rem 0 1.2rem; }
        .hero h1 { margin: .2rem 0 .4rem; color: #f8fafc; font-size: 2.25rem; }
        .hero p { color: #aab8d1; margin: 0; }
        .pill { display: inline-block; padding: .28rem .65rem; border-radius: 999px; background: #173c50; color: #8be5ff; font-size: .78rem; font-weight: 700; }
        .experimental-badge { display: inline-block; padding: .25rem .55rem; border-radius: 999px; background: #4a3516; color: #f6c85f; font-size: .72rem; font-weight: 800; letter-spacing: .05em; }
        .signal-card { border: 1px solid #2c3d62; border-radius: 16px; padding: 1rem 1.1rem; background: #121b31; min-height: 128px; }
        .muted { color: #9eacc5; font-size: .9rem; }
        div[data-testid="stMetricValue"] { color: #f8fafc; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_bundle(path: str) -> ArtifactBundle:
    return load_artifact_bundle(path)


@st.cache_resource(show_spinner=False)
def _load_bundles(paths: tuple[str, ...]) -> tuple[tuple[ArtifactBundle, ...], tuple[str, ...]]:
    """Load every discovered bundle once, preserving clear errors for bad candidates."""
    bundles: list[ArtifactBundle] = []
    errors: list[str] = []
    for path in paths:
        try:
            bundles.append(load_artifact_bundle(path))
        except ArtifactValidationError as exc:
            errors.append(f"{path}: {exc}")
    return tuple(bundles), tuple(errors)


def _friendly(name: str) -> str:
    replacements = {
        "pres_": "Presentation ", "qa_": "Q&A ", "_z": " surprise",
        "_mean": " mean", "_frac": " fraction", "_entropy": " entropy",
        "_slope": " slope", "_history_count": " history count",
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


def _signal(probability: float, threshold: float) -> tuple[str, str, str]:
    if probability >= threshold + 0.05:
        return "OUTPERFORM", "#55d187", "The model assigns a higher probability to a positive abnormal return."
    if probability <= threshold - 0.05:
        return "UNDERPERFORM", "#ff7b8b", "The model assigns a lower probability to a positive abnormal return."
    return "NEUTRAL", "#f6c85f", "The model output is close to the decision boundary; conviction is limited."


def _predict(bundle: ArtifactBundle, row: pd.DataFrame) -> tuple[float, str]:
    if bundle.predictions is not None:
        predictions = bundle.predictions.copy()
        predictions["symbol"] = predictions["symbol"].astype(str)
        predictions["call_datetime"] = pd.to_datetime(predictions["call_datetime"], errors="coerce")
        key = row.iloc[0]
        matched = predictions[
            (predictions["symbol"] == str(key["symbol"]))
            & (predictions["call_datetime"] == pd.to_datetime(key["call_datetime"]))
        ]
        if not matched.empty:
            return float(matched.iloc[-1]["probability"]), "Stored out-of-sample prediction"

    features = bundle.feature_columns
    probabilities = bundle.model.predict_proba(row[features])[:, 1]
    probability = float(probabilities[0])
    if not 0 <= probability <= 1:
        raise ArtifactValidationError("The active model returned a probability outside [0, 1].")
    return probability, "Active model inference"


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
        return pd.DataFrame({"Feature": [_friendly(name) for name in features], "Contribution": contributions}).sort_values(
            "Contribution", key=abs, ascending=False
        )
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({
            "Feature": [_friendly(name) for name in features],
            "Importance": estimator.feature_importances_,
        }).sort_values("Importance", ascending=False)
    return pd.DataFrame()


def _metric_snapshot(bundle: ArtifactBundle) -> pd.DataFrame | None:
    if bundle.metrics is None or bundle.metrics.empty:
        return None
    metrics = bundle.metrics.copy()
    if "split" in metrics:
        filtered = metrics[metrics["split"].astype(str).isin(["final_holdout", "walk_forward_aggregate"])]
        if not filtered.empty:
            return filtered
    return metrics.head(10)


@st.cache_data(show_spinner=False)
def _load_comparison_metrics() -> pd.DataFrame | None:
    """Load the frozen same-sample comparison used by the presentation tab."""
    path = Path(__file__).resolve().parent / "artifacts" / "model_comparison" / "metrics.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _comparison_label(model: str) -> str:
    labels = {
        "original_logistic": "Original Logistic",
        "sentence_plus_historical_xgboost_depth1_trees100": "Rich XGBoost * Experimental",
    }
    return labels.get(str(model), str(model).replace("_", " ").title())


def _comparison_frame(metrics: pd.DataFrame, split: str) -> pd.DataFrame:
    metric_map = {
        "auc": "AUC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "balanced_accuracy": "Balanced accuracy",
        "mcc": "MCC",
        "log_loss": "Log loss",
        "brier": "Brier score",
    }
    compared_models = {
        "original_logistic",
        "sentence_plus_historical_xgboost_depth1_trees100",
    }
    frame = metrics[
        metrics["split"].eq(split) & metrics["model"].isin(compared_models)
    ].copy()
    frame["Model"] = frame["model"].map(_comparison_label)
    columns = [column for column in metric_map if column in frame]
    rows = []
    for _, row in frame.iterrows():
        for column in columns:
            rows.append({"Model": row["Model"], "Metric": metric_map[column], "Value": float(row[column])})
    return pd.DataFrame(rows)


def _render_comparison_chart(frame: pd.DataFrame, title: str) -> None:
    st.markdown(f"#### {title}")
    if frame.empty:
        st.info("No metrics are available for this evaluation split.")
        return
    try:
        import plotly.express as px
        figure = px.bar(
            frame,
            x="Metric",
            y="Value",
            color="Model",
            barmode="group",
            text_auto=".3f",
            color_discrete_sequence=["#7dd3fc", "#f6c85f"],
        )
        figure.update_layout(
            height=430,
            margin={"t": 20, "b": 30, "l": 10, "r": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(18,27,49,0.75)",
            font={"color": "#dbeafe"},
            legend={"orientation": "h", "y": 1.08},
            yaxis={"range": [0, 1], "gridcolor": "#2c3d62"},
        )
        st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    except ImportError:
        st.dataframe(frame.pivot(index="Metric", columns="Model", values="Value"), use_container_width=True)


def _render_model_comparison() -> None:
    metrics = _load_comparison_metrics()
    st.subheader("Model comparison")
    st.caption("Controlled comparison on the same 886 Industrials events and the same 2019–2022 walk-forward folds.")
    if metrics is None:
        st.warning("The controlled comparison artifact is unavailable.")
        return

    walk = _comparison_frame(metrics, "walk_forward_aggregate")
    holdout = _comparison_frame(metrics, "final_holdout")
    _render_comparison_chart(walk, "Walk-forward aggregate")
    _render_comparison_chart(holdout, "Exploratory final holdout")

    st.markdown("#### Direct metric differences")
    selected_metrics = ["auc", "accuracy", "precision", "recall", "f1", "balanced_accuracy", "mcc", "log_loss", "brier"]
    rows = []
    for split, label in (("walk_forward_aggregate", "Walk-forward"), ("final_holdout", "Holdout")):
        subset = metrics[metrics["split"].eq(split)].copy()
        original = subset[subset["model"].eq("original_logistic")].iloc[0]
        rich = subset[subset["model"].eq("sentence_plus_historical_xgboost_depth1_trees100")].iloc[0]
        for metric in selected_metrics:
            if metric in subset:
                rows.append({
                    "Evaluation": label,
                    "Metric": metric.replace("_", " ").title(),
                    "Original": float(original[metric]),
                    "Rich experimental": float(rich[metric]),
                    "Rich − original": float(rich[metric] - original[metric]),
                })
    difference_table = pd.DataFrame(rows)
    st.dataframe(
        difference_table.style.format({
            "Original": "{:.3f}",
            "Rich experimental": "{:.3f}",
            "Rich − original": "{:+.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Concise interpretation")
    left, right = st.columns(2)
    with left:
        st.markdown("**How it works**")
        st.caption(
            "A standardized Logistic Regression using the original 11 aggregate language features: "
            "presentation and Q&A sentiment probabilities, net sentiment, presentation/Q&A mismatches, and an evasion index."
        )
        st.markdown("**Original Logistic Regression**")
        st.markdown(
            "- **Pros:** simple, interpretable, and strongest latest-holdout AUC in this comparison.\n"
            "- **Cons:** weak pooled walk-forward ranking and only coarse baseline sentiment features.\n"
            "- **Read:** the safer reference model, but its temporal performance is limited."
        )
    with right:
        st.markdown("**How it works**")
        st.caption(
            "A shallow XGBoost model built from 39 features extracted from individual transcript sentences, including sentiment statistics, "
            "beginning/middle/end tone, sentiment slope, presentation-versus-Q&A differences, and prior-only company-history surprise z-scores."
        )
        st.markdown("**Rich XGBoost · Experimental**")
        st.markdown(
            "- **Pros:** higher walk-forward AUC, accuracy, F1, and lower log loss; uses sentence-level and company-history features.\n"
            "- **Cons:** more complex, fewer complete-language events, and slightly lower holdout AUC than the original.\n"
            "- **Read:** stronger research candidate, but not a proven replacement."
        )
    st.info(
        "Interpretation: the richer model looks better across the historical walk-forward folds, while the original baseline is marginally better on the latest exploratory holdout. "
        "That disagreement is why the richer model is labeled experimental rather than presented as definitively superior."
    )


def _render_gauge(probability: float, threshold: float) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.progress(probability)
        st.metric("Positive-return probability", f"{probability:.1%}")
        return
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"color": "#f8fafc", "size": 34}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8292ae"},
            "bar": {"color": "#38bdf8"},
            "bgcolor": "#1a2742",
            "bordercolor": "#34486e",
            "steps": [
                {"range": [0, (threshold - 0.05) * 100], "color": "#2d1d35"},
                {"range": [(threshold - 0.05) * 100, (threshold + 0.05) * 100], "color": "#27334a"},
                {"range": [(threshold + 0.05) * 100, 100], "color": "#173c50"},
            ],
        },
    ))
    figure.update_layout(
        height=190,
        margin={"t": 15, "b": 0, "l": 18, "r": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#dbeafe"},
    )
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


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

    # Keep the stable baseline first, while allowing future bundles to appear
    # without changing frontend code.
    bundles = tuple(sorted(bundles, key=lambda item: (item.is_experimental, item.display_label.lower())))
    labels = [bundle.display_label for bundle in bundles]
    # Avoid an accidental selector collision if two artifact folders use the
    # same display name.
    label_to_bundle: dict[str, ArtifactBundle] = {}
    unique_labels: list[str] = []
    for bundle, label in zip(bundles, labels):
        unique_label = label if label not in label_to_bundle else f"{label} [{bundle.model_version}]"
        unique_labels.append(unique_label)
        label_to_bundle[unique_label] = bundle
    with st.sidebar:
        selected_model_label = st.selectbox("Model", unique_labels, help="Choose the validated model artifact used for this view.")
    bundle = label_to_bundle[selected_model_label]

    table = bundle.feature_table.copy()
    if "symbol" not in table or "call_datetime" not in table:
        st.error("The artifact feature table must contain symbol and call_datetime columns.")
        return
    table["symbol"] = table["symbol"].astype(str)
    table["call_datetime"] = pd.to_datetime(table["call_datetime"], errors="coerce")

    st.markdown('<div class="eyebrow">Market intelligence · research prototype</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero"><span class="pill">{bundle.display_label.upper()}</span>'
        '<h1>Earnings Call Intelligence</h1>'
        '<p>Explore how management language compares with a company\'s own history and how the active model ranks post-call outcomes.</p></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Explore a call")
        st.caption(f"Artifact: `{bundle.model_version}`")
        if bundle.is_experimental:
            st.markdown('<span class="experimental-badge">* EXPERIMENTAL</span>', unsafe_allow_html=True)
            st.caption("Candidate richer-feature model; use for comparison, not as a confirmed replacement.")
        symbols = sorted(table["symbol"].dropna().unique())
        query = st.text_input("Search ticker", placeholder="e.g. ADP")
        visible = [symbol for symbol in symbols if query.upper() in symbol.upper()] or symbols
        symbol = st.selectbox("Company", visible)
        company = table[table["symbol"] == symbol].sort_values("call_datetime")
        options = []
        labels_seen: dict[str, int] = {}
        for index, row in company.iterrows():
            date = row["call_datetime"]
            quarter = row.get("quarter", "")
            date_text = date.strftime("%b %d, %Y") if pd.notna(date) else "Undated call"
            label = f"{date_text} · Q{quarter}"
            labels_seen[label] = labels_seen.get(label, 0) + 1
            if labels_seen[label] > 1:
                label = f"{label} · event {labels_seen[label]}"
            options.append((index, label))
        if not options:
            st.warning("No dated calls are available for this company in the active artifact.")
            return
        selected_label = st.selectbox("Earnings call", [label for _, label in options], index=len(options) - 1)
        selected_index = {label: index for index, label in options}[selected_label]
        selected = company.loc[[selected_index]].copy()
        st.divider()
        st.caption("Offline artifact demo. No live transcripts, prices, or investment advice.")

    selected_row = selected.iloc[0]
    try:
        probability, prediction_source = _predict(bundle, selected)
    except Exception as exc:
        st.error("The active model could not score this call.")
        st.code(str(exc))
        return
    threshold = float(bundle.schema["prediction_threshold"])
    signal, color, explanation = _signal(probability, threshold)
    call_date = selected_row["call_datetime"]
    call_date_text = call_date.strftime("%b %d, %Y · %H:%M") if pd.notna(call_date) else "Date unavailable"
    phase = str(selected_row.get("call_phase", "event timing unavailable")).replace("_", " ").title()
    company_name = selected_row.get("company_name", symbol)

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown(f"### {company_name}")
        st.markdown(f"`{symbol}`  ·  {call_date_text}  ·  {phase}")
        st.markdown(f"### :{('green' if color == '#55d187' else 'red' if color == '#ff7b8b' else 'orange')}[{signal}]")
        st.write(explanation)
        st.caption(f"{prediction_source} · decision threshold {threshold:.0%}")
    with right:
        _render_gauge(probability, threshold)

    st.divider()
    cards = st.columns(4)
    card_values = [
        ("Positive-return probability", f"{probability:.1%}"),
        ("Presentation net tone", _as_float(selected_row.get("pres_net_sentiment"))),
        ("Q&A net tone", _as_float(selected_row.get("qa_net_sentiment"))),
        ("Presentation/Q&A gap", _as_float(selected_row.get("sentiment_mismatch_pos"))),
    ]
    for card, (label, value) in zip(cards, card_values):
        card.metric(label, value if isinstance(value, str) else ("—" if value is None else f"{value:+.2f}"))

    signal_tab, language_tab, history_tab, research_tab, comparison_tab = st.tabs([
        "Signal", "Language profile", "Call history", "Research context", "Model comparison"
    ])
    with signal_tab:
        st.subheader("What the model is seeing")
        profile_columns = [
            column for column in bundle.feature_columns
            if column in selected and any(token in column for token in ("sent", "mismatch", "evasion", "entropy", "slope", "momentum", "volatility", "beta"))
        ]
        if profile_columns:
            profile = selected[profile_columns].T.rename(columns={selected.index[0]: "Value"})
            profile.index = [_friendly(str(index)) for index in profile.index]
            st.dataframe(profile, use_container_width=True)
        explanation_table = _model_explanation(bundle, selected)
        if not explanation_table.empty:
            st.subheader("Model explanation")
            st.dataframe(explanation_table.head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Feature explanation is unavailable for this model family.")

    with language_tab:
        st.subheader("Presentation versus Q&A")
        timeline_columns = [column for column in (
            "pres_begin_mean", "pres_middle_mean", "pres_end_mean",
            "qa_begin_mean", "qa_middle_mean", "qa_end_mean",
        ) if column in selected]
        if len(timeline_columns) == 6:
            timeline = pd.DataFrame({
                "Presentation": [selected_row.get("pres_begin_mean"), selected_row.get("pres_middle_mean"), selected_row.get("pres_end_mean")],
                "Q&A": [selected_row.get("qa_begin_mean"), selected_row.get("qa_middle_mean"), selected_row.get("qa_end_mean")],
            }, index=["Beginning", "Middle", "End"])
            st.line_chart(timeline)
        else:
            st.info("Sentence-position features are not present in this baseline artifact. They will appear automatically in a richer model bundle.")

        dictionary_columns = [column for column in bundle.feature_columns if column in selected and any(token in column for token in ("uncertainty", "litigious", "modal", "constraining", "guidance", "expectations", "eps"))]
        if dictionary_columns:
            details = selected[dictionary_columns].T.rename(columns={selected.index[0]: "Value"})
            details.index = [_friendly(str(index)) for index in details.index]
            st.dataframe(details, use_container_width=True)

    with history_tab:
        st.subheader(f"{symbol} call history")
        history_columns = [column for column in (
            "symbol", "company_name", "call_datetime", "quarter", "pres_net_sentiment", "qa_net_sentiment",
        ) if column in company]
        st.dataframe(company[history_columns].sort_values("call_datetime", ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Download selected call data",
            selected.to_csv(index=False).encode("utf-8"),
            file_name=f"{symbol}_earnings_call.csv",
            mime="text/csv",
        )
        st.subheader("Historical backtest context")
        target_column = str(bundle.schema.get("target_column", "abnormal_return_5d"))
        realized = _as_float(selected_row.get(target_column))
        if realized is None:
            st.info("No realized outcome is available for this call.")
        elif realized > 0:
            st.success(f"Historical outcome: outperformed the benchmark ({realized:+.2%}). This value was not used as a model input.")
        else:
            st.error(f"Historical outcome: underperformed the benchmark ({realized:+.2%}). This value was not used as a model input.")

    with research_tab:
        st.subheader("Model and validation context")
        manifest = bundle.manifest
        cols = st.columns(3)
        cols[0].metric("Model family", str(manifest.get("model_family", bundle.schema.get("model_family", "Unknown"))))
        cols[1].metric("Target", str(manifest.get("target_display", bundle.schema.get("target_column", "Unknown"))))
        cols[2].metric("Artifact version", bundle.model_version)
        snapshot = _metric_snapshot(bundle)
        if snapshot is not None:
            st.dataframe(snapshot, use_container_width=True, hide_index=True)
        st.markdown(
            "This is an offline research prototype. The probability ranks the likelihood of a positive five-session abnormal return under the stored model. "
            "It is not a guarantee, a price target, or investment advice."
        )

    with comparison_tab:
        _render_model_comparison()


if __name__ == "__main__":
    main()
