# Earnings Call Intelligence frontend

This directory contains the static frontend: `index.html` defines the visual system and `app.js` supplies the interactions.

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

Configure the Vercel project with the repository root as its project root and frontend/ as its output directory. The root-level Python /api/chat Function is deployed alongside the static site. Re-run the exporter whenever the artifact files change, commit the updated JSON, and redeploy.

The repository also contains a Python /api/chat function for the optional research assistant. When deploying the combined dashboard, set the Vercel project root to the repository root and configure frontend/ as the static output directory. The function requires the server-side OPENAI_API_KEY; the browser only calls the relative /api/chat endpoint.
