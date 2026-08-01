# Earnings Call Intelligence Dashboard

The Earnings Call Intelligence Dashboard helps users explore whether the language used during earnings calls is associated with a company’s short-term market performance.

## Live app

[Open the deployed dashboard](https://earnings-call-dashboard.vercel.app/)

The app is a static frontend deployed on Vercel. It reads validated, pre-exported model artifacts from `frontend/data/app-data.json`; it does not fetch transcripts, market data, or credentials in the browser.

## What the dashboard shows

- **Overview:** a quick explanation of the five-session outcome and the current model context.
- **Screener & Signals:** searchable companies and earnings calls with directional predictions and confidence indicators.
- **Call Detail:** supporting transcript evidence, feature values, historical context, and optional price-window charts.
- **Model Reliability:** walk-forward and holdout performance, probability quality, sample coverage, and model comparison.

## Performance snapshot

The current richer model is a shallow XGBoost model using language, sentence-position, and historical context features. Its target is whether the company’s five-session abnormal return—company return minus S&P 500 return—is positive or negative.

- **Walk-forward AUC:** 0.631 across 276 evaluation calls from 2019–2022
- **Later holdout AUC:** 0.627 across 185 calls from 2023 onward
- **Brier score:** 0.237 walk-forward and 0.241 on the holdout; lower is better, with about 0.25 representing an uninformative 50/50 probability
- **Coverage:** 61 companies

The results suggest a modest ranking signal: the model is generally better than random at placing positive outcomes above negative ones, but it is not a highly accurate predictor. Performance varies by period, and the richer model is strongest in walk-forward evaluation rather than every individual comparison. These results are research evidence, not a trading strategy or investment advice.

## Run locally

From this directory:

```bash
python3 -m http.server 4173 --directory frontend
```

Then open <http://127.0.0.1:4173/>.

## Refresh the exported data

When model artifacts change, regenerate the frontend data from this directory:

```bash
python scripts/export_frontend_data.py
```

The exporter reads the validated bundles under `artifacts/` and writes `frontend/data/app-data.json`. Commit the updated JSON before deploying.

## Deploy

Use the repository root as the Vercel project root and frontend/ as the output directory. The Python /api/chat Function is discovered from the root-level api/ directory.

Optional historical price windows can be generated offline with `scripts/enrich_price_windows.py` using Alpaca credentials. Credentials stay local; the browser only receives the exported chart data.

## Data and evaluation notes

The dashboard uses time-aware validation: earlier years are used to evaluate later years, and a later holdout is reported separately. Predictions, confidence labels, feature values, and reliability metrics come from the checked-in artifacts. Missing transcript evidence, feature groups, or price history is shown as unavailable rather than filled with placeholder data.

## Research assistant

The static dashboard includes a bottom-right research assistant. It sends each question to the same-project POST /api/chat serverless function; the browser never receives the OpenAI API key. Answers are grounded only in the curated project record:

- README.md, RESEARCH_README.md, and RESEARCH_SUMMARY.md
- data/validation_summary.csv and data/validation_scorecard.csv
- the stored comparison and experimental model metric CSVs under data/artifacts/ and artifacts/

The assistant returns the supporting filenames with each answer and states when the project sources do not contain enough information. It is a documentation aid for research and education, not investment advice.

### Run the assistant locally

The Vercel Function uses the minimal dependencies in requirements.txt. For the Streamlit dashboard and model tooling, install requirements-dashboard.txt. Set OPENAI_API_KEY in the environment, then run the frontend/API through a Vercel-compatible development server. The frontend calls /api/chat relative to its origin:

~~~bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
vercel dev
~~~

For deployment, configure the Vercel project root as this repository and use frontend/ as the output directory. Add OPENAI_API_KEY as a server-side Vercel environment variable. Do not put the key in frontend/, app.js, or any browser-exposed file.
