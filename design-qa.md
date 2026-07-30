# Design QA

## Status

Blocked for screenshot-level verification in this workspace.

The static frontend was served successfully and its HTML, JavaScript, and exported JSON were fetched over HTTP. Headless Firefox is installed, but it did not produce a screenshot in this sandbox, so responsive screenshots, 200% zoom, keyboard focus visuals, and final pixel-level comparison against `AI4ALL_Mockup.html` still need to be verified in a normal local browser preview.

## Verified without a browser

- The legacy Streamlit app still loads without exceptions.
- The static frontend serves the mockup shell, `app.js`, and `data/app-data.json`.
- The data export contains real companies, probabilities, provenance, feature groups, outcomes, and reliability values.
- The frontend code contains no mockup `CALLS_DATA`, no sample CTAS/FinBERT values, and no random market chart generation.
- Missing price series produce an availability message instead of a generated market path.
