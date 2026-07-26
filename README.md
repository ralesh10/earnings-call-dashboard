# Earnings Call Intelligence Dashboard

An offline Streamlit presentation dashboard for comparing earnings-call language with a company's historical market outcomes. The app loads one validated, versioned model artifact bundle at a time; it does not fetch transcripts, prices, or secrets at runtime.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app discovers validated sibling bundles under `artifacts/` and provides a model dropdown. The stable reference is `artifacts/original_baseline/`; the richer sentence-plus-historical model is marked `* Experimental` and lives in `artifacts/experimental_rich/`.

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

Bundles can include `status: "experimental"` in their schema or manifest. Those models are labeled with `* Experimental` in the selector and receive a warning in the sidebar.

## Dashboard sections

- **Signal:** probability gauge, signal label, and active model metadata.
- **Language profile:** presentation/Q&A comparison and richer sentence/dictionary features when available.
- **Call history:** prior calls and a downloadable selected-call record.
- **Research context:** stored validation metrics, target definition, and limitations.
- **Model comparison:** frozen same-sample walk-forward and exploratory holdout comparisons for the original and experimental models, including AUC, accuracy, precision, recall, F1, log loss, and Brier score.

Historical abnormal returns are shown only in the clearly labeled backtest section; they are never passed to the model as inputs.

## Deployment

Streamlit Community Cloud can use `app.py` as the entrypoint. Commit the selected artifact directory and `requirements.txt`; no API keys are required for the offline demo. Configure `EARNINGS_ARTIFACT_DIR` only if the active bundle is not the default path.
