"""Generate the consolidated, reviewer-facing research notebook."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


cells = [
    markdown("""
# Earnings-Call Intelligence: Consolidated Research and Validation Notebook

This is the reviewer-facing research record for the earnings-call project. It consolidates the original baseline, corrected event construction, walk-forward validation, rich language features, complete-case analysis, hyperparameter/model comparisons, and leakage checks.

The default path is a fast artifact verification run. It loads saved out-of-sample predictions under data/artifacts/, independently recomputes the reported metrics, and tests target timing, historical normalization, temporal separation, company-clustered uncertainty, and permutation significance.

Current evidence: on the same 886-event, 61-company complete-language Industrials sample, the sentence-plus-historical XGBoost has walk-forward AUC about 0.631 versus 0.555 for the reconstructed original sentiment Logistic baseline. The exploratory 2023+ holdout AUCs are about 0.627 and 0.635 respectively. This is modest research evidence, not a proven trading strategy.
"""),
    markdown("""
## How to run

Download the whole repository before opening this notebook. It searches the current working directory and its parents for the repository root. If you open it from elsewhere, set `EARNINGS_CALL_PROJECT_ROOT` to the local repository path before running the first code cell. For example:

```bash
export EARNINGS_CALL_PROJECT_ROOT=/path/to/earnings-call-dashboard
```

On Windows PowerShell, use `$env:EARNINGS_CALL_PROJECT_ROOT = 'C:\\path\\to\\earnings-call-dashboard'`. The default verification path does not download the 33,000+ transcript corpus, call yfinance, or run FinBERT.

Expensive refitting and full source rebuilding are guarded by RUN_LOCAL_REFIT and RUN_FULL_REBUILD flags near the end. Enable those only when the source package, raw/cache transcript inputs, financial dictionary, and price inputs are available.

Saved-prediction verification checks the statistical claims from the completed runs. A full rebuild additionally checks data acquisition and feature extraction.
"""),
    code("""
from pathlib import Path
import ast
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from IPython.display import display, Markdown


def _is_repo_root(candidate):
    return (
        (candidate / 'src' / 'earnings_intelligence').is_dir()
        and (candidate / 'data' / 'artifacts').is_dir()
    )


def find_repo_root():
    configured_root = os.environ.get('EARNINGS_CALL_PROJECT_ROOT')
    if configured_root:
        configured = Path(configured_root).expanduser().resolve()
        if _is_repo_root(configured):
            return configured
        raise FileNotFoundError(
            'EARNINGS_CALL_PROJECT_ROOT does not point to this repository: '
            f'{configured}'
        )

    candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    for candidate in candidates:
        if _is_repo_root(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        'Could not locate the earnings-call-dashboard repository. '
        'Run this notebook from the repository or set '
        'EARNINGS_CALL_PROJECT_ROOT to its local path.'
    )


ROOT = find_repo_root()
SRC = ROOT / 'src'
ARTIFACT_ROOT = ROOT / 'data' / 'artifacts'
CONTROLLED_ROOT = ARTIFACT_ROOT / 'controlled_comparison'
sys.path.insert(0, str(SRC))

RANDOM_STATE = 42
WALK_FORWARD_YEARS = (2019, 2020, 2021, 2022)
HOLDOUT_CUTOFF = 2023
TARGET_COLUMN = 'abnormal_return_5d'
BOOTSTRAP_REPETITIONS = 1000
PERMUTATION_REPETITIONS = 1000

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 180)
print('Repository:', ROOT)
print('Artifacts:', ARTIFACT_ROOT)
"""),
    code("""
# Optional fresh-environment installation:
# %pip install -q pandas numpy scipy scikit-learn xgboost matplotlib seaborn

from earnings_intelligence.config import EventConfig, EvaluationConfig, ModelConfig
from earnings_intelligence.events import build_event_dataset
from earnings_intelligence.features import add_expanding_history_features

print(EventConfig())
print(EvaluationConfig())
print(ModelConfig())
"""),
    markdown("""
## Reproducibility contract

The primary protocol is fixed before interpreting the final holdout:

- Unit: one timestamped earnings call.
- Universe: current GICS Industrials mapping, calls from 2015 onward. Sector membership is not point-in-time historical membership.
- Target: sign of the five-session abnormal return versus the S&P 500.
- Timing: pre-open calls use the event day's first tradable session; after-close calls begin the next session; intraday calls are excluded.
- Primary evaluation: expanding walk-forward test years 2019, 2020, 2021, and 2022.
- Holdout: 2023 onward, exploratory because it was inspected during development.
- Preprocessing: imputation, scaling, and correlation pruning are fit within training folds.
- Historical surprise: prior-call statistics only, with sector fallback when company history is insufficient.
- Uncertainty: company-clustered bootstrap intervals.
"""),
    code("""
required_files = [
    CONTROLLED_ROOT / 'metrics.csv',
    CONTROLLED_ROOT / 'predictions.csv',
    CONTROLLED_ROOT / 'auc_differences.csv',
    CONTROLLED_ROOT / 'confusion_matrices.csv',
    CONTROLLED_ROOT / 'feature_schema.csv',
    CONTROLLED_ROOT / 'run_manifest.json',
]
missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError('Missing saved research artifacts:\\n' + '\\n'.join(missing))

controlled_metrics = pd.read_csv(CONTROLLED_ROOT / 'metrics.csv')
controlled_predictions = pd.read_csv(CONTROLLED_ROOT / 'predictions.csv')
auc_differences = pd.read_csv(CONTROLLED_ROOT / 'auc_differences.csv')
confusion_matrices = pd.read_csv(CONTROLLED_ROOT / 'confusion_matrices.csv')
feature_schema = pd.read_csv(CONTROLLED_ROOT / 'feature_schema.csv')
controlled_manifest = json.loads((CONTROLLED_ROOT / 'run_manifest.json').read_text())

controlled_predictions['call_datetime'] = pd.to_datetime(controlled_predictions['call_datetime'], errors='coerce')
controlled_predictions['event_year'] = pd.to_numeric(controlled_predictions['event_year'], errors='coerce').astype('Int64')
controlled_predictions['y'] = pd.to_numeric(controlled_predictions['y'], errors='coerce').astype(int)
controlled_predictions['probability'] = pd.to_numeric(controlled_predictions['probability'], errors='coerce')

display(pd.DataFrame([controlled_manifest]))
print('Stored predictions:', controlled_predictions.shape)
print('Stored metrics:', controlled_metrics.shape)
"""),
    markdown("""
## Research progression

The project moved through these branches:

1. Continuous abnormal-return regression with Linear Regression and XGBoost. Test R squared was approximately zero or negative, so the estimand changed to directional classification.
2. Aggregate presentation/Q&A FinBERT sentiment Logistic Regression and XGBoost.
3. Market-control expansion with momentum, volatility, market momentum, and beta.
4. Validation-standard implementation with expanding walk-forward folds, exploratory holdout, clustered intervals, permutation checks, and probability-spread diagnostics.
5. Rich language features: sentence statistics, presentation/Q&A changes, financial dictionary rates, historical company-relative sentiment, and earnings-language proxies.
6. Complete-case and all-row comparisons, Elastic Net diagnostics, shallow XGBoost comparisons, focused feature unions, and E9 sector-transfer experiments.
7. Corrected same-sample comparison of the original baseline and sentence-plus-historical XGBoost.
"""),
    code("""
experiment_ledger = pd.DataFrame([
    ['Initial continuous-return regression', 'Linear Regression and XGBoost', 'R2 near zero or negative', 'Exploratory / superseded'],
    ['Directional baseline', 'Aggregate presentation/Q&A FinBERT; Logistic and XGBoost', 'More stable but modest ranking', 'Provisional'],
    ['Market-control expansion', 'Momentum, volatility, market momentum, beta', 'No consistent incremental gain', 'Ablation'],
    ['Validation standard', 'Walk-forward, holdout, clustered intervals, permutation', 'Reduced optimism and exposed uncertainty', 'Authoritative protocol'],
    ['Rich feature experiment', 'Sentence, dictionary, historical surprise, earnings proxies; Elastic Net/XGBoost', 'Historical and sentence-plus-historical blocks strongest', 'Provisional'],
    ['Complete-case comparison', '886 events, 61 companies; original versus rich XGBoost', 'Rich walk-forward AUC 0.631 versus original 0.555', 'Current controlled evidence'],
    ['E9 multiverse / sector transfer', 'Per-block tuning and frozen sector transfer', 'Sector composition and completeness matter', 'Generalization diagnostic'],
], columns=['stage', 'models_or_features', 'finding', 'status'])
display(experiment_ledger)
"""),
    markdown("""
### Earlier validation artifact

The older validation export is retained for historical context. Its 183-row holdout and 276-row walk-forward samples are not the same as the later 886-event complete-case comparison, so these numbers are displayed as provisional and are never merged with the controlled results below.
"""),
    code("""
legacy_validation_path = ROOT / 'data' / 'validation_summary.csv'
if legacy_validation_path.exists():
    legacy_validation = pd.read_csv(legacy_validation_path)
    display(legacy_validation[
        legacy_validation['model'].isin(['standardized_logistic', 'train_rate_baseline'])
        & legacy_validation['split'].isin(['walk_forward_2019_2022', 'final_holdout'])
    ][['variant', 'split', 'model', 'n', 'accuracy', 'balanced_accuracy', 'auc', 'average_precision', 'log_loss', 'brier']])
    print('Historical context only: this export predates the corrected complete-case comparison.')
else:
    print('No legacy validation summary found; continuing with controlled artifacts.')
"""),
    markdown("""
## Data and sample audit

The raw source corpus is much larger than the modeling sample. Filtering requires usable timestamps, transcript length, sector mapping, valid tickers, available prices, non-intraday timing, and a complete five-session target. The controlled comparison is the 886-event language-complete sample and must not be silently combined with the broader all-row sample.
"""),
    code("""
event_keys = controlled_predictions[['symbol', 'call_datetime', 'event_year', 'y', 'return']].drop_duplicates(['symbol', 'call_datetime'])
sample_audit = pd.DataFrame([
    ['Full complete-case cohort events', controlled_manifest['rows']],
    ['Full cohort companies', controlled_manifest['companies']],
    ['Out-of-sample prediction events', len(event_keys)],
    ['Out-of-sample prediction companies', event_keys['symbol'].nunique()],
    ['Out-of-sample event-year range', f"{event_keys['event_year'].min()}-{event_keys['event_year'].max()}"],
    ['Out-of-sample positive target count', int(event_keys['y'].sum())],
    ['Out-of-sample negative target count', int((1 - event_keys['y']).sum())],
    ['Out-of-sample positive target rate', event_keys['y'].mean()],
    ['Duplicate symbol/date keys', int(event_keys.duplicated(['symbol', 'call_datetime']).sum())],
    ['Missing timestamps', int(event_keys['call_datetime'].isna().sum())],
], columns=['check', 'value'])
display(sample_audit)

print('Events by evaluation split:')
display(controlled_predictions.groupby(['split', 'model'], as_index=False).size().head(20))
"""),
    code("""
feature_table_path = ARTIFACT_ROOT / 'feature_table.csv'
feature_frame = pd.read_csv(feature_table_path) if feature_table_path.exists() else None
if feature_frame is not None:
    parsed_dates = pd.to_datetime(feature_frame.get('call_datetime'), errors='coerce')
    table_audit = pd.DataFrame([
        ['Saved feature-table rows', len(feature_frame)],
        ['Saved feature-table companies', feature_frame['symbol'].nunique()],
        ['Duplicate feature-table columns', int(feature_frame.columns.duplicated().sum())],
        ['Duplicate symbol/date keys', int(feature_frame.duplicated(['symbol', 'call_datetime']).sum())],
        ['Invalid call timestamps', int(parsed_dates.isna().sum())],
        ['Event year differs from timestamp year', int((pd.to_numeric(feature_frame['event_year'], errors='coerce') != parsed_dates.dt.year).sum())],
    ], columns=['check', 'value'])
    display(table_audit)
    if 'call_phase' in feature_frame:
        display(feature_frame['call_phase'].value_counts(dropna=False).rename_axis('call_phase').reset_index(name='rows'))
else:
    print('No CSV feature table found; controlled predictions remain available.')
"""),
    markdown("""
## Target construction and event-time causality

The target is abnormal return, not raw stock return:

AR equals the company return from the event baseline through five sessions minus the benchmark return over the same window. Company and benchmark prices use the same adjusted basis. Intraday calls are excluded because daily prices cannot identify the post-call information set.

The target is built from post-event prices for offline evaluation only. It is never in the model feature schema.
"""),
    code("""
synthetic_dates = pd.bdate_range('2020-01-01', periods=40)
synthetic_prices = pd.DataFrame({'Adj Close': np.linspace(100.0, 140.0, len(synthetic_dates))}, index=synthetic_dates)
synthetic_benchmark = pd.Series(np.linspace(100.0, 110.0, len(synthetic_dates)), index=synthetic_dates)
synthetic_transcripts = pd.DataFrame({
    'symbol': ['TEST', 'TEST', 'TEST'],
    'date': ['2020-01-02 08:30:00', '2020-01-03 17:00:00', '2020-01-06 12:00:00'],
})
synthetic_result = build_event_dataset(
    synthetic_transcripts,
    {'TEST': synthetic_prices},
    synthetic_benchmark,
    config=EventConfig(price_col='Adj Close'),
)
display(synthetic_result[['call_datetime', 'call_phase', 'target_baseline_date', 'target_end_date_5d', 'abnormal_return_5d']])
display(pd.DataFrame([synthetic_result.attrs['event_audit']]))
assert set(synthetic_result['call_phase']) == {'pre_open', 'after_close'}
assert synthetic_result.attrs['event_audit']['intraday_excluded'] == 1
print('Synthetic event-timing assertions passed.')
"""),
    markdown("""
## Prior-only historical surprise test

Historical surprise is sentiment relative to prior company history, not EPS surprise:

z_t = (x_t - mean(prior observations)) / std(prior observations).

The current observation is excluded from its own statistics. At least four prior observations are required for a company history score; sector history can be used as a fallback.
"""),
    code("""
history_test = pd.DataFrame({
    'symbol': ['AAA'] * 5,
    'gics_sector': ['Industrials'] * 5,
    'call_datetime': pd.date_range('2020-01-01', periods=5, freq='QE'),
    'pres_sent_mean': [0.0, 1.0, 2.0, 3.0, 100.0],
})
history_output = add_expanding_history_features(history_test, ['pres_sent_mean'], min_history=4)
display(history_output)
assert history_output.loc[:3, 'pres_sent_mean_z'].isna().all()
assert history_output.loc[4, 'pres_sent_mean_history_count'] == 4
assert history_output.loc[4, 'historical_score_source'] == 'company'
print('Prior-only historical normalization assertions passed.')
"""),
    markdown("""
## Feature inventory and leakage screen

The controlled comparison uses separate, ordered schemas:

- Original baseline: 11 aggregate presentation/Q&A sentiment, net-tone, mismatch, and evasion features.
- Historical surprise: prior-only z-scores and history counts.
- Sentence plus historical: sentence-level means, dispersion, quantiles, fractions, entropy, beginning/middle/end tone, slope, Q&A-minus-presentation features, and prior-only history features.

Raw text, realized returns, target labels, and future-return fields must never enter a model schema.
"""),
    code("""
feature_schema['features_parsed'] = feature_schema['features'].map(ast.literal_eval)
feature_schema['feature_count'] = feature_schema['features_parsed'].map(len)
display(feature_schema[['model', 'feature_block', 'feature_count']])

all_declared_features = sorted({feature for values in feature_schema['features_parsed'] for feature in values})
forbidden_markers = ('target', 'label', 'abnormal_return', 'future_return', 'stock_return', 'market_return', 'return_5d', 'actual_outcome', 'realized_return')
forbidden_features = [feature for feature in all_declared_features if any(marker in feature.lower() for marker in forbidden_markers)]
print('Declared feature count:', len(all_declared_features))
print('Target-like declared features:', forbidden_features)
assert not forbidden_features

if feature_frame is not None:
    target_like_columns = [column for column in feature_frame.columns if any(marker in column.lower() for marker in forbidden_markers)]
    print('Target-like columns in saved table, allowed only for offline labels:', target_like_columns)
print('Leakage-name screen passed.')
"""),
    code("""
block_sets = {row['feature_block']: set(row['features_parsed']) for _, row in feature_schema.iterrows()}
overlap_rows = []
block_names = list(block_sets)
for i, first in enumerate(block_names):
    for second in block_names[i + 1:]:
        overlap_rows.append({'block_a': first, 'block_b': second, 'overlap_count': len(block_sets[first] & block_sets[second])})
display(pd.DataFrame(overlap_rows))
"""),
    markdown("""
## Walk-forward and holdout separation

Each walk-forward test year is evaluated using earlier observations only. The final holdout is trained on pre-2023 observations and evaluated on 2023 onward. Because the holdout was inspected during development, its role is exploratory.
"""),
    code("""
walk_predictions = controlled_predictions[controlled_predictions['split'].str.startswith('walk_forward')].copy()
holdout_predictions = controlled_predictions[controlled_predictions['split'].eq('final_holdout')].copy()

fold_rows = []
for split, group in controlled_predictions.groupby('split'):
    if split.startswith('walk_forward_'):
        test_year = int(split.rsplit('_', 1)[-1])
        fold_rows.append({'split': split, 'test_year': test_year, 'test_rows': group[['symbol', 'call_datetime']].drop_duplicates().shape[0], 'train_is_earlier': True})
    elif split == 'final_holdout':
        fold_rows.append({'split': split, 'test_year': HOLDOUT_CUTOFF, 'test_rows': group[['symbol', 'call_datetime']].drop_duplicates().shape[0], 'train_is_earlier': True})
display(pd.DataFrame(fold_rows))

walk_keys = set(map(tuple, walk_predictions[['symbol', 'call_datetime']].drop_duplicates().to_numpy()))
holdout_keys = set(map(tuple, holdout_predictions[['symbol', 'call_datetime']].drop_duplicates().to_numpy()))
print('Walk/holdout key overlap:', len(walk_keys & holdout_keys))
assert len(walk_keys & holdout_keys) == 0
assert holdout_predictions['event_year'].ge(HOLDOUT_CUTOFF).all()
print('Temporal separation assertions passed.')
"""),
    markdown("""
## Independent metric recomputation

The next cell recomputes accuracy, balanced accuracy, precision, recall, F1, MCC, ROC AUC, average precision, log loss, Brier score, and confusion-matrix counts directly from saved out-of-sample probabilities.
"""),
    code("""
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)


def recompute_metrics(group):
    y = group['y'].to_numpy(dtype=int)
    p = np.clip(group['probability'].to_numpy(dtype=float), 1e-7, 1 - 1e-7)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'n': len(y),
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'mcc': matthews_corrcoef(y, pred),
        'auc': roc_auc_score(y, p),
        'average_precision': average_precision_score(y, p),
        'log_loss': log_loss(y, p, labels=[0, 1]),
        'brier': brier_score_loss(y, p),
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
    }


recomputed_rows = []
for (model, split), group in controlled_predictions.groupby(['model', 'split']):
    row = recompute_metrics(group)
    row.update({'model': model, 'split': split})
    recomputed_rows.append(row)
recomputed = pd.DataFrame(recomputed_rows)
display(recomputed.query("split in ['walk_forward_aggregate', 'final_holdout']"))
"""),
    code("""
metric_columns = ['n', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'average_precision', 'log_loss', 'brier']
stored_subset = controlled_metrics[controlled_metrics['split'].isin([
    'walk_forward_2019', 'walk_forward_2020', 'walk_forward_2021', 'walk_forward_2022', 'walk_forward_aggregate', 'final_holdout'
])].copy()
check = recomputed.merge(stored_subset, on=['model', 'split'], suffixes=('_recomputed', '_stored'))
comparison_rows = []
for column in metric_columns:
    if f'{column}_recomputed' in check and f'{column}_stored' in check:
        comparison_rows.append({
            'metric': column,
            'max_abs_difference': (check[f'{column}_recomputed'] - check[f'{column}_stored']).abs().max(),
        })
metric_check = pd.DataFrame(comparison_rows)
display(metric_check)
assert metric_check['max_abs_difference'].max() < 1e-9
print('Independent metric recomputation matches stored metrics.')
"""),
    markdown("""
## Results and per-fold stability
"""),
    code("""
model_labels = {
    'original_logistic': 'Original Logistic',
    'historical_xgboost_depth1_trees200': 'Historical-surprise XGBoost',
    'sentence_plus_historical_xgboost_depth1_trees100': 'Sentence + historical XGBoost',
}
result_table = stored_subset.copy()
result_table['model_label'] = result_table['model'].map(model_labels).fillna(result_table['model'])
display(result_table[['model_label', 'split', 'n', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'mcc', 'auc', 'average_precision', 'log_loss', 'brier']].sort_values(['split', 'model_label']))
"""),
    code("""
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print('Plotting dependencies are unavailable; the numeric tables remain valid.')
else:
    fold_auc = stored_subset[stored_subset['split'].isin([f'walk_forward_{year}' for year in WALK_FORWARD_YEARS])].copy()
    fold_auc['model_label'] = fold_auc['model'].map(model_labels).fillna(fold_auc['model'])
    plt.figure(figsize=(11, 5))
    sns.barplot(data=fold_auc, x='split', y='auc', hue='model_label')
    plt.ylim(0.4, 0.75)
    plt.axhline(0.5, color='black', linestyle='--', linewidth=1)
    plt.title('Out-of-sample AUC by walk-forward year')
    plt.ylabel('ROC AUC')
    plt.xlabel('Test fold')
    plt.tight_layout()
    plt.show()
"""),
    code("""
headline = stored_subset[stored_subset['split'].isin(['walk_forward_aggregate', 'final_holdout'])].copy()
headline['model_label'] = headline['model'].map(model_labels).fillna(headline['model'])
display(headline.pivot(index='model_label', columns='split', values=['auc', 'accuracy', 'precision', 'recall', 'f1', 'log_loss', 'brier']).round(4))
display(auc_differences[auc_differences['split'].isin(['walk_forward_aggregate', 'final_holdout'])])
"""),
    markdown("""
## Confusion matrices and class imbalance

A no-information or all-positive rule can have recall equal to one and a respectable F1 without ranking skill. That is why balanced accuracy, MCC, AUC, log loss, and Brier score are reported alongside accuracy and F1.
"""),
    code("""
display(confusion_matrices[confusion_matrices['split'].isin(['walk_forward_aggregate', 'final_holdout'])].assign(
    model_label=lambda frame: frame['model'].map(model_labels).fillna(frame['model'])
))
"""),
    markdown("""
## Company-clustered bootstrap uncertainty

Calls from the same company are correlated. The interval below resamples companies rather than individual calls and includes every call from each sampled company.
"""),
    code("""
def cluster_bootstrap_auc(predictions, repetitions=BOOTSTRAP_REPETITIONS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    group_values = predictions['symbol'].astype(str).to_numpy()
    groups = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in groups}
    y = predictions['y'].to_numpy(dtype=int)
    probability = predictions['probability'].to_numpy(dtype=float)
    values = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_positions = np.concatenate([positions[group] for group in sampled_groups])
        if np.unique(y[sampled_positions]).size < 2:
            continue
        values.append(roc_auc_score(y[sampled_positions], probability[sampled_positions]))
    return tuple(np.quantile(values, [0.025, 0.975]))


bootstrap_rows = []
for model in model_labels:
    group = controlled_predictions[(controlled_predictions['model'] == model) & controlled_predictions['split'].str.startswith('walk_forward')]
    lower, upper = cluster_bootstrap_auc(group)
    bootstrap_rows.append({'model': model_labels[model], 'auc': roc_auc_score(group['y'], group['probability']), 'lower_95': lower, 'upper_95': upper, 'companies': group['symbol'].nunique()})
display(pd.DataFrame(bootstrap_rows))
"""),
    code("""
def paired_cluster_bootstrap_delta(predictions, challenger, baseline, repetitions=BOOTSTRAP_REPETITIONS, seed=RANDOM_STATE):
    wide = predictions[predictions['model'].isin([challenger, baseline])].pivot_table(
        index=['symbol', 'call_datetime', 'split', 'y'], columns='model', values='probability'
    ).reset_index()
    walk = wide[wide['split'].str.startswith('walk_forward')].dropna(subset=[challenger, baseline])
    rng = np.random.default_rng(seed)
    group_values = walk['symbol'].astype(str).to_numpy()
    groups = np.unique(group_values)
    positions = {group: np.flatnonzero(group_values == group) for group in groups}
    y = walk['y'].to_numpy(dtype=int)
    challenger_probability = walk[challenger].to_numpy(dtype=float)
    baseline_probability = walk[baseline].to_numpy(dtype=float)
    deltas = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled_positions = np.concatenate([positions[group] for group in sampled_groups])
        if np.unique(y[sampled_positions]).size < 2:
            continue
        deltas.append(roc_auc_score(y[sampled_positions], challenger_probability[sampled_positions]) - roc_auc_score(y[sampled_positions], baseline_probability[sampled_positions]))
    observed = roc_auc_score(y, challenger_probability) - roc_auc_score(y, baseline_probability)
    return {'observed_delta': observed, 'lower_95': np.quantile(deltas, 0.025), 'upper_95': np.quantile(deltas, 0.975), 'prob_delta_gt_zero': np.mean(np.asarray(deltas) > 0)}


display(pd.DataFrame([
    {'comparison': 'Historical XGBoost - Original Logistic', **paired_cluster_bootstrap_delta(controlled_predictions, 'historical_xgboost_depth1_trees200', 'original_logistic')},
    {'comparison': 'Sentence + historical XGBoost - Original Logistic', **paired_cluster_bootstrap_delta(controlled_predictions, 'sentence_plus_historical_xgboost_depth1_trees100', 'original_logistic')},
]))
"""),
    markdown("""
## Permutation test for chance association

This test keeps saved out-of-sample probabilities fixed and repeatedly permutes labels. It asks whether an AUC at least as large as observed is common under a no-association null. It does not repair earlier data-snooping and is not a substitute for an untouched future sample.
"""),
    code("""
def permutation_auc_pvalue(group, repetitions=PERMUTATION_REPETITIONS, seed=RANDOM_STATE):
    y = group['y'].to_numpy(dtype=int)
    p = group['probability'].to_numpy(dtype=float)
    observed = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    null = np.asarray([roc_auc_score(rng.permutation(y), p) for _ in range(repetitions)])
    return {'observed_auc': observed, 'null_mean': null.mean(), 'p_value_one_sided': (np.sum(null >= observed) + 1) / (len(null) + 1)}


permutation_rows = []
for model in model_labels:
    group = controlled_predictions[(controlled_predictions['model'] == model) & controlled_predictions['split'].str.startswith('walk_forward')]
    permutation_rows.append({'model': model_labels[model], **permutation_auc_pvalue(group)})
display(pd.DataFrame(permutation_rows))
"""),
    markdown("""
## What was tested

Feature families included:

- Original aggregate FinBERT presentation/Q&A probabilities, net sentiment, mismatches, and evasion.
- Pre-event market controls: 5/20-session momentum, volatility, market momentum, and beta.
- Sentence-level means, standard deviations, quantiles, positive/negative fractions, entropy, beginning/middle/end tone, slopes, sentence counts, and Q&A-minus-presentation differences.
- Financial dictionary positive/negative, uncertainty, litigious, modal, constraining, complexity, and token-rate features.
- Prior-only company-relative sentiment z-scores and history counts with sector fallback.
- Earnings-language proxies for EPS/per-share mentions, beat/miss wording, expectation wording, and guidance direction. These are not true EPS surprises.
- Standardized L2 Logistic Regression, Elastic Net Logistic Regression, and shallow XGBoost.
"""),
    code("""
for _, row in feature_schema.iterrows():
    print(f"\\n{row['feature_block']} | {row['model']} | {len(row['features_parsed'])} features")
    print(', '.join(row['features_parsed']))
"""),
    markdown("""
## Optional: refit from the saved feature table

This is disabled by default because the locally saved feature table can differ slightly from the exact 886-event controlled complete-case sample. Enable it to verify that the reusable package performs fold-safe refitting on the available table.
"""),
    code("""
RUN_LOCAL_REFIT = False

if RUN_LOCAL_REFIT:
    from earnings_intelligence.features import feature_block_columns
    from earnings_intelligence.modeling import evaluate_holdout, evaluate_walk_forward

    if feature_frame is None:
        raise FileNotFoundError('data/artifacts/feature_table.csv is required for local refitting.')
    refit_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()].copy()
    refit_frame['label'] = (pd.to_numeric(refit_frame[TARGET_COLUMN], errors='coerce') > 0).astype(int)
    declared_blocks = json.loads((ARTIFACT_ROOT / 'feature_blocks.json').read_text())
    available_blocks = feature_block_columns(refit_frame, declared_blocks)
    walk_refit, walk_refit_predictions = evaluate_walk_forward(
        refit_frame, available_blocks, target_col='label', model_names=('logistic', 'xgboost'),
        evaluation_config=EvaluationConfig(bootstrap_repetitions=250),
    )
    holdout_refit, holdout_refit_predictions = evaluate_holdout(
        refit_frame, available_blocks, target_col='label', model_names=('logistic', 'xgboost'),
        cutoff_year=HOLDOUT_CUTOFF, evaluation_config=EvaluationConfig(bootstrap_repetitions=250),
    )
    display(walk_refit.sort_values('auc', ascending=False).head(20))
    display(holdout_refit.sort_values('auc', ascending=False).head(20))
else:
    print('RUN_LOCAL_REFIT=False; using saved out-of-sample predictions.')
"""),
    markdown("""
## Optional: full source rebuild

The original Colab branches downloaded the Hugging Face transcript corpus, mapped current sectors, downloaded adjusted prices, extracted FinBERT sentence features, added dictionary and earnings-language features, created prior-only historical scores, and evaluated the model ladder. That path is expensive and requires network access plus substantial RAM/GPU time.

Set RUN_FULL_REBUILD=True only after supplying transcript and price inputs. The reusable pipeline rebuilds labels from current adjusted prices rather than trusting labels inside a language cache.
"""),
    code("""
RUN_FULL_REBUILD = False

if RUN_FULL_REBUILD:
    from earnings_intelligence.final_pipeline import run_final_experiment
    required_inputs = {'df_bose', 'price_data', 'benchmark'}
    missing_inputs = sorted(required_inputs - set(globals()))
    if missing_inputs:
        raise RuntimeError('Full rebuild requires these namespace inputs: ' + ', '.join(missing_inputs))
    full_results = run_final_experiment(
        globals(),
        lm_dictionary_path=str(ROOT / 'data' / 'lm_dictionary.csv'),
        output_dir=str(ROOT / 'data' / 'artifacts' / 'reproduced_run'),
    )
    display(full_results['walk_summary'])
else:
    print('RUN_FULL_REBUILD=False; no network download or FinBERT extraction was started.')
"""),
    markdown("""
## Bias and limitation checklist

Implemented controls:

1. Lookahead: event labels use post-event prices; model schemas are screened for target/future-return names; historical z-scores are tested as prior-only.
2. Temporal leakage: walk-forward training precedes every test year; the holdout is separate and exploratory.
3. Preprocessing leakage: the reusable evaluator fits imputation, scaling, and correlation pruning inside training folds.
4. Dependence: intervals resample companies rather than treating repeated company calls as independent.
5. Selection bias: exploratory branches are distinguished from the controlled complete-case comparison; holdout metrics are not used for tuning.
6. Missing language: complete-case and all-row samples are not silently combined.

Residual limitations: current sector mapping is not point-in-time, timestamps are source wall-clock times, the holdout was inspected during development, four walk-forward years provide limited regime coverage, and no point-in-time consensus EPS data is available. A significant permutation result means measurable association in these saved out-of-sample predictions, not a proven deployable strategy.
"""),
    markdown("""
## Final interpretation

On the same 886-event, 61-company Industrials sample, sentence-level sentiment combined with prior-only company-history features produced modestly better temporal ranking than the original aggregate-sentiment Logistic baseline across 2019–2022 walk-forward folds. The latest exploratory holdout did not establish a decisive winner.

The defensible conclusion is measurable but modest predictive information, not a validated trading strategy. A reviewer should report the sample definition, target timing, fold design, complete-case distinction, company-clustered intervals, and holdout status beside every headline metric.
"""),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parents[1] / "notebooks" / "earnings_call_research_record.ipynb"
output.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"Wrote {output} with {len(cells)} cells")
