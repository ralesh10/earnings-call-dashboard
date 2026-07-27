# Earnings Call Intelligence Dashboard

An offline Streamlit research workspace for comparing earnings-call language with a company's historical market outcomes. The app loads one validated, versioned model artifact bundle as the active context while keeping sibling bundles available for contextual comparison; it does not fetch transcripts, prices, or secrets at runtime.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The production-facing UI is now also available as a mockup-native static frontend under [`frontend/`](/home/jia/Projects/AI4ALL_Final/earnings-call-dashboard/frontend/). It removes Streamlit’s native controls and spreadsheet components while preserving the Python artifact contract. Refresh its real-data export with `python scripts/export_frontend_data.py`, then preview it with `python3 -m http.server 4173 --directory frontend`.

The app discovers validated sibling bundles under `artifacts/`. The richer sentence-plus-historical model is the default research context because it contains stored out-of-sample predictions; it is marked `* Experimental` and lives in `artifacts/experimental_rich/`. The stable reference remains available from call detail.

The original root-level files are retained only as migration/reference copies and are not loaded by the application.

## Static frontend deployment

For a close one-to-one visual implementation, deploy `frontend/` as the Vercel project root. No Node build is required. The checked-in `frontend/data/app-data.json` is generated from the validated artifacts; rerun the exporter and redeploy whenever model artifacts change. Streamlit remains available as the Python research/debug surface during the migration.

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

The interface follows the terminal-style mockup while keeping every visible score backed by the selected artifact:

- **Overview:** explains the five-session target, shows active-artifact coverage, and highlights one real scored call.
- **Screener & Signals:** a validated-first company and call explorer with search, signal/confidence/status filters, and pagination.
- **Call Detail Terminal:** the primary research view with a directional signal, confidence, base-rate comparison, progressive-disclosure evidence, optional event-window price charts, model comparison, backtest context, and technical details.
- **Model Reliability & Lineage:** compact model cards, a labeled nonnegative AUC slope chart, Brier score, sample context, methodology, and the current nuanced comparison result.

The top bar controls the active model. The ticker tape, signal labels, validation provenance, probabilities, historical outcomes, feature values, and reliability metrics all come from the artifact files; mockup example companies and sample values are not used.

Prediction provenance is shown per call as **Out-of-sample holdout**, **Walk-forward validated**, **Retrospective inference**, or **Unavailable**. Missing optional price series, transcript evidence, and feature groups are called out instead of being replaced with fabricated content.

The model lab prioritizes walk-forward AUC, latest holdout AUC, Brier score, sample size, and evaluation period. Accuracy, precision, recall, F1, log loss, and other raw metrics remain available under the technical details expander.

Historical abnormal returns are shown only in the clearly labeled backtest section; they are never passed to the model as inputs.

## Deployment

Streamlit Community Cloud can use `app.py` as the entrypoint. Commit the selected artifact directory and `requirements.txt`; no API keys are required for the offline demo. Configure `EARNINGS_ARTIFACT_DIR` only if the active bundle is not the default path.
