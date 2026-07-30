# Earnings Call Intelligence frontend

This directory is the mockup-native static frontend. It intentionally does not use Streamlit widgets: `index.html` preserves the finalized visual system and `app.js` supplies the interactions.

## Refresh real artifact data

From the repository root:

```bash
python scripts/export_frontend_data.py
```

The exporter reads the validated model bundles and writes `frontend/data/app-data.json`. The frontend never uses the mockup’s sample companies, sample probabilities, or generated chart path.

## Preview locally

```bash
python3 -m http.server 4173 --directory frontend
```

Then open `http://127.0.0.1:4173/`.

## Deploy to Vercel

Create a Vercel project with this directory as the project root. It is a static site: use no build command and serve the project root as the output. Re-run the exporter whenever the artifact files change, commit the updated JSON, and redeploy.
