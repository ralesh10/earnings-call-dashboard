# Earnings Call Intelligence Dashboard

An offline Streamlit research workspace for comparing earnings-call language with a company's historical market outcomes. The app loads one validated, versioned model artifact bundle as the active context while keeping sibling bundles available for contextual comparison; it does not fetch transcripts, prices, or secrets at runtime.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app discovers validated sibling bundles under `artifacts/`. The richer sentence-plus-historical model is the default research context because it contains stored out-of-sample predictions; it is marked `* Experimental` and lives in `artifacts/experimental_rich/`. The stable reference remains available from call detail.

The original root-level files are retained only as migration/reference copies and are not loaded by the application.

## Swap the active model

To deploy one bundle without showing the local model catalog, point the app at another validated bundle without changing frontend code:

```bash
EARNINGS_ARTIFACT_DIR=artifacts/final_model streamlit run app.py
```

The directory must contain:

```text
model.joblib
feature_schema.json
feature_table.csv
run_manifest.json
```

`predictions.csv` and `metrics.csv` are optional. If stored predictions contain a matching `symbol` and `call_datetime`, the dashboard displays that stored probability; otherwise it uses `model.predict_proba` on the selected feature row.

`feature_schema.json` must declare an ordered, unique `feature_columns` list. Target, future-return, and realized-outcome fields are rejected as model features. The target may remain in `feature_table.csv` so historical backtest context can be displayed separately.

Bundles can include `status: "experimental"` in their schema or manifest. Those models are labeled with `* Experimental` when selected for call detail. The permanent sidebar has been removed so model choice stays in context with the call being investigated.

## Dashboard flow

- **Home:** a compact explanation of the product, its five-session target, the validation concept, and links into the research flow.
- **Calls:** a company-first explorer that defaults to validated calls, groups sequential calls by company, supports search/filtering, and paginates older results.
- **Call detail:** the primary research view with a directional signal, confidence, base-rate comparison, progressive-disclosure evidence, optional event-window price charts, model comparison, backtest context, and technical details.
- **Reliability:** compact model cards, a labeled AUC comparison chart, Brier score, sample context, methodology, and the current nuanced comparison result.

Prediction provenance is shown per call as **Out-of-sample holdout**, **Walk-forward validated**, **Retrospective inference**, or **Unavailable**. Missing optional price series, transcript evidence, and feature groups are called out instead of being replaced with fabricated content.

The model lab prioritizes walk-forward AUC, latest holdout AUC, Brier score, sample size, and evaluation period. Accuracy, precision, recall, F1, log loss, and other raw metrics remain available under the technical details expander.

Historical abnormal returns are shown only in the clearly labeled backtest section; they are never passed to the model as inputs.

## Deployment

Streamlit Community Cloud can use `app.py` as the entrypoint. Commit the selected artifact directory and `requirements.txt`; no API keys are required for the offline demo. Configure `EARNINGS_ARTIFACT_DIR` only if the active bundle is not the default path.
