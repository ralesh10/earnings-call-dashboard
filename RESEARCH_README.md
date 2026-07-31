# Earnings-call research record

This repository now contains the consolidated, reproducible research notebook alongside the dashboard application.

## Run the verification notebook

Open [`notebooks/earnings_call_research_record.ipynb`](notebooks/earnings_call_research_record.ipynb) in Jupyter or upload that file to Colab. The notebook defaults to a fast verification mode that loads the saved controlled-comparison artifacts and checks:

- target timing and intraday exclusion logic;
- prior-only historical feature construction;
- target/future-return leakage screens;
- walk-forward versus holdout separation;
- stored metric reproducibility;
- clustered uncertainty and permutation diagnostics.

The notebook expects the repository layout already included here: `src/earnings_intelligence/` and `data/artifacts/`. It does not require the 142 MB language-feature cache for the default verification run.

Install the research dependencies with:

```bash
python -m pip install -r research_requirements.txt
```

Then run:

```bash
jupyter notebook notebooks/earnings_call_research_record.ipynb
```

## Optional rebuild

Set `RUN_LOCAL_REFIT = True` in the notebook only when you want to refit the saved cohort. Set `RUN_FULL_REBUILD = True` only when the raw transcript corpus and price inputs are available. These modes need substantially more memory and additional NLP dependencies than the default verification path.

## What was copied

- `notebooks/earnings_call_research_record.ipynb`: single research notebook.
- `scripts/build_research_notebook.py`: notebook generator/source.
- `src/earnings_intelligence/`: reusable event, feature, modeling, artifact, and experiment code.
- `data/artifacts/controlled_comparison/`: same-886-event original-versus-rich model comparison.
- `data/artifacts/`: feature table, feature blocks, predictions, and summary artifacts.
- `data/validation_*.csv`: earlier validation audit outputs.
- `data/lm_dictionary.csv`: optional financial-language dictionary input.
- `RESEARCH_SUMMARY.md`: detailed project research record.

The large `feature_frame_ready.pkl` cache and raw transcript archive were intentionally not copied into the dashboard repo; they are not needed for notebook verification and would unnecessarily increase repository size.

The existing `app.py` and dashboard model files remain unchanged. The dashboard and research record can therefore be developed and run independently in this repository.
