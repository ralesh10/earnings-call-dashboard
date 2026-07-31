# Earnings-Call Intelligence Project Summary

**Last updated:** 2026-07-25  
**Status:** Research prototype; final modeling decision not yet frozen

## 1. Executive summary

This project tests whether earnings-call language contains information about a company’s subsequent stock performance after controlling for broad market movement.

The primary target is the direction of the company’s five-session abnormal return after an earnings call:

> abnormal return = company return − S&P 500 return

The current strongest result comes from the complete-case experiment:

- 886 events with successfully extracted presentation and Q&A language features
- 61 companies
- Historical company-relative sentiment features
- Shallow XGBoost
- Mean 2019–2022 walk-forward AUC: **0.611**
- Pooled walk-forward AUC: **0.622**
- Exploratory 2023–2025 holdout AUC: **0.596**

This is evidence of modest predictive information, not proof of a reliable trading strategy. The holdout is lower than the pooled walk-forward result, and the sample is small once company clustering and complete language availability are considered.

The remaining recommended work is one small, predeclared hyperparameter comparison. If it does not produce a material and consistent improvement, the model should be frozen and the project should move to interpretation, dashboard presentation, and documentation.

## 2. Project progression

### Stage 1: Initial continuous-return regression

The first approach attempted to predict the magnitude of the five-session abnormal return directly.

- Linear Regression: train R² approximately `0.0016`; test R² approximately `-0.0115`.
- XGBoost regression: train R² approximately `0.0000`; test R² approximately `-0.0110`.
- XGBoost assigned effectively zero feature importance.

The continuous target was too noisy to model reliably. This motivated a change to directional classification.

### Stage 2: Directional classification

The target was converted to:

```text
1 if five-session abnormal return > 0
0 otherwise
```

An early exploratory chronological split used pre-2023 events for training and 2023+ events for testing. The cached exploratory results were:

| Model | Accuracy | AUC | Baseline accuracy |
|---|---:|---:|---:|
| Logistic Regression | 55.7% | 0.627 | 51.3% |
| XGBoost Classifier | 56.0% | 0.584 | 51.3% |

These results were not treated as final because the same 2023+ period was repeatedly inspected during experimentation.

### Stage 3: Baseline sentiment model

The first serious feature set used transcript-level FinBERT probabilities:

- Presentation positive, negative, and neutral probabilities.
- Q&A positive, negative, and neutral probabilities.
- Presentation/Q&A positive mismatch.
- Presentation/Q&A negative mismatch.
- Presentation net sentiment.
- Q&A net sentiment.
- Evasion index.

The baseline models were Logistic Regression and XGBoost. The baseline was useful because it established whether broad presentation/Q&A sentiment had any measurable value before adding richer features.

### Stage 4: Market-control expansion

Pre-call market variables were added:

- Five-session momentum.
- Twenty-session momentum.
- Twenty-session volatility.

The final corrected implementation also includes twenty-session market momentum and a 120-session pre-event beta.

The purpose was to test whether transcript language added information beyond recent price behavior and market conditions.

### Stage 5: Validation standard

The evaluation process was strengthened to include:

- Expanding walk-forward validation for 2019, 2020, 2021, and 2022.
- A 2023–2025 final holdout reported separately.
- Train-only imputation, scaling, and correlation pruning.
- Majority and train-rate baselines.
- ROC AUC, average precision, balanced accuracy, MCC, log loss, and Brier score.
- Calibration error.
- Top-versus-bottom probability-quintile return spreads.
- After-cost spreads.
- Company-clustered bootstrap intervals.
- Permutation diagnostics in the earlier validation phase.

Walk-forward validation measures temporal stability across multiple historical periods. The latest holdout is a single later regime: it is noisier, but it is essential for checking whether a historically selected model generalizes to newer data. Neither metric should be used alone.

### Stage 6: Rich feature experiment

The rich experiment introduced:

- Sentence-level FinBERT statistics.
- Financial dictionary features.
- Prior-only historical sentiment normalization.
- Earnings-language and guidance proxies.
- Elastic Net Logistic Regression.
- Shallow XGBoost.
- Beta-adjusted abnormal-return robustness targets.

Earlier rich results were provisional. Some early block definitions were contaminated by broad prefix matching, and the target and price-basis rules were subsequently corrected.

### Stage 7: Corrected final pipeline

The final pipeline corrected the main methodological issues:

- Event year comes from the actual call timestamp.
- Intraday calls are excluded because daily prices cannot isolate their post-call movement.
- Pre-open and after-close windows use different baselines.
- Company and benchmark returns use the same adjusted-price basis.
- Historical z-scores use prior observations only.
- Cached language features cannot supply trusted target labels.
- Stale market and historical features are removed from older caches and rebuilt.
- Feature blocks are disjoint and target-like columns are rejected.
- Artifacts include model, schema, metrics, predictions, feature blocks, and target audits.

### Stage 8: Complete-case experiment

The all-row corrected run contained 1,516 target events, but language features were present for only 886. The other 630 rows had missing presentation/Q&A language features and were median-imputed by the model pipeline.

The complete-case experiment removed those incomplete language rows and recomputed historical features on the remaining 886 events. This produced a more comparable sample to the earlier approximately 883-event baseline and materially improved the results.

## 3. Data and target construction

### Source data

The project uses the Hugging Face earnings-transcript dataset:

```text
Bose345/sp500_earnings_transcripts
```

The source corpus contains more than 33,000 transcripts. The final comparable experiment is intentionally restricted to:

- Current Industrials membership.
- Calls from 2015 onward.
- Calls with usable daily market-price history.

Sector membership is based on current or fallback sector mappings rather than a complete point-in-time historical membership database. This is documented as a limitation.

### Event timing

- Pre-open calls use the call date’s first available trading session as the post-event start.
- After-close calls use the next available trading session.
- Intraday calls between 09:30 and 16:00 are excluded.
- Timezone-naive source timestamps are interpreted according to their source wall-clock time, with market-hour thresholds treated as America/New_York.

### Primary target

For each valid event:

1. Identify the first eligible post-call trading session.
2. Use the preceding available session as the company and benchmark baseline.
3. Calculate the five-session company return.
4. Calculate the five-session S&P 500 return.
5. Subtract the benchmark return from the company return.
6. Set the directional label to one when the abnormal return is positive.

Both company and benchmark prices use adjusted close in the corrected pipeline.

### Beta-adjusted robustness target

A second target estimates a pre-event market model using historical company and benchmark returns. The post-event abnormal return is then measured relative to the estimated alpha and beta relationship.

This is a robustness check, not the primary target used to select the final model.

### Target audit counts

| Run | Target events | Companies | Positive rate | Notes |
|---|---:|---:|---:|---|
| Corrected all-row run | 1,516 | 67 | 50.86% | 1,258 intraday calls excluded; no missing-price exclusions |
| Complete-case run | 886 | 61 | 51.69% | Both presentation and Q&A sentiment available |

The complete-case run retained 731 pre-open and 155 after-close events.

## 4. Feature inventory

Raw transcript text, future returns, target labels, and post-event information are not model features.

### Baseline sentiment

- Presentation positive, negative, and neutral probabilities.
- Q&A positive, negative, and neutral probabilities.
- Presentation net sentiment.
- Q&A net sentiment.
- Presentation/Q&A positive mismatch.
- Presentation/Q&A negative mismatch.
- Evasion index.

### Market-only controls

- Five-session momentum.
- Twenty-session momentum.
- Twenty-session volatility.
- Twenty-session market momentum.
- 120-session pre-event beta.

All market controls are calculated using prices available before the call’s post-event window.

### Sentence sentiment

For presentation and Q&A separately:

- Mean net sentiment.
- Sentiment standard deviation.
- Lower and upper sentiment quantiles.
- Mean positive, negative, and neutral probabilities.
- Positive and negative sentence fractions.
- Entropy.
- Beginning, middle, and end sentiment.
- Sentiment slope through the call.
- Sentence count.

Additional features measure Q&A-minus-presentation differences for sentiment, dispersion, entropy, sentence fractions, and slope.

### Financial dictionary

Presentation and Q&A rates are calculated using Loughran–McDonald-style categories:

- Positive language.
- Negative language.
- Uncertainty.
- Litigious language.
- Strong modal language.
- Weak modal language.
- Constraining language.
- Complexity.
- Token counts.
- Q&A-minus-presentation differences.

### Historical surprise

Historical features normalize the current call against earlier communication from the same company:

\[
z_t = \frac{x_t - \mu_{\text{prior company}}}{\sigma_{\text{prior company}}}
\]

The current observation is never included in its own mean or standard deviation. At least four prior company observations are required. When company history is insufficient, a prior sector-level fallback may be used.

The normalized variables include:

- Presentation sentiment.
- Q&A sentiment.
- Presentation entropy.
- Q&A entropy.
- Presentation negative fraction.
- Q&A negative fraction.
- Presentation slope.
- Q&A slope.
- Prior-history counts.

These features measure sentiment surprise relative to a company’s communication history. They are not EPS surprises or true earnings surprises.

### Earnings-language proxies

- EPS and per-share mention counts.
- Beat-language counts.
- Miss-language counts.
- Above-expectations language.
- Below-expectations language.
- Guidance-up indicators.
- Guidance-down indicators.
- Guidance-maintained indicators.
- Forward-looking language counts.

These are transcript-derived language proxies only. No point-in-time analyst consensus EPS or revenue-surprise dataset is currently available.

## 5. Models and evaluation methods

### Models tested

- Majority-class baseline.
- Train-rate probability baseline.
- Initial Linear Regression.
- Initial XGBoost regression.
- Standardized L2 Logistic Regression.
- Elastic Net Logistic Regression.
- Shallow XGBoost classification.

The initial regression models failed to explain return magnitude. The project therefore focuses on directional classification.

### Elastic Net

The rich pipeline tested:

```text
C: 0.01, 0.1, 1, 10
l1_ratio: 0.1, 0.5, 0.9
```

The complete-case experiment selected `C=1.0` and `l1_ratio=0.9` during tuning, but the final complete-case winner was still XGBoost on the historical-surprise block.

### Evaluation metrics

- ROC AUC: ranking quality independent of a single classification threshold.
- Average precision: ranking quality relative to the positive-class prevalence.
- Balanced accuracy: threshold-based performance adjusted for class imbalance.
- MCC: correlation between predictions and labels, useful near balanced classes.
- Log loss: probability quality and confidence calibration.
- Brier score: squared probability error.
- Calibration error: agreement between predicted probabilities and observed frequencies.
- Top-versus-bottom spread: abnormal-return difference between high- and low-probability groups.
- After-cost spread: spread after an assumed two-leg transaction-cost deduction.
- Cluster bootstrap interval: uncertainty interval resampling companies rather than treating every call as independent.

## 6. Results

### Initial exploratory results

The early 2023+ exploratory split produced:

| Model | Accuracy | AUC | Baseline accuracy |
|---|---:|---:|---:|
| Logistic Regression | 55.7% | 0.627 | 51.3% |
| XGBoost Classifier | 56.0% | 0.584 | 51.3% |

These values are not authoritative because the same holdout period was repeatedly inspected during development.

### Earlier provisional rich-feature results

The earlier rich experiment reported approximately:

| Feature block/model | Walk-forward AUC | Final holdout AUC |
|---|---:|---:|
| Market-only Logistic | 0.543 | 0.473 |
| Dictionary-only Logistic | 0.533 | 0.612 |
| Historical features Logistic | 0.583 | 0.614 |
| Historical features XGBoost | 0.622 | 0.575 |
| All features Logistic | 0.621 | 0.612 |
| All features XGBoost | 0.608 | 0.577 |

These results were retained as historical context, not final evidence, because the target basis, sample composition, feature-block definitions, and holdout usage were subsequently corrected.

### Corrected all-row run

The corrected all-row run used 1,516 target events and 67 companies. Its selected model was sentence-sentiment XGBoost:

- Mean walk-forward AUC: `0.588`.
- Pooled walk-forward AUC: `0.555`.
- Cluster bootstrap interval: approximately `0.519–0.593`.
- Exploratory holdout AUC: `0.547`.
- Holdout after-cost spread: approximately `1.26` percentage points.

The all-row run retained 630 events with missing language features and median-imputed them. This made the language-model comparison less clean.

### Complete-case run

The complete-case run used 886 events and 61 companies. The selected model was historical-surprise XGBoost:

- Mean walk-forward AUC: `0.611`.
- Pooled walk-forward AUC: `0.622`.
- Cluster bootstrap interval: approximately `0.558–0.685`.
- Walk-forward fold AUCs:
  - 2019: `0.621`.
  - 2020: `0.624`.
  - 2021: `0.621`.
  - 2022: `0.580`.
- Exploratory holdout AUC: `0.596`.
- Holdout after-cost spread: approximately `3.69` percentage points.

### Key comparison

| Run | Events | Winner | Mean walk AUC | Pooled walk AUC | Holdout AUC |
|---|---:|---|---:|---:|---:|
| Original baseline | approximately 883 | Logistic baseline | 0.585 | Not consistently available | 0.611 |
| Corrected all-row run | 1,516 | Sentence XGBoost | 0.588 | 0.555 | 0.547 |
| Complete-case run | 886 | Historical-surprise XGBoost | 0.611 | 0.622 | 0.596 |

These runs are not perfectly apples-to-apples. They use different target corrections, event counts, company counts, feature availability rules, and evaluation implementations.

### Beta-adjusted robustness

In the complete-case run, the historical-surprise XGBoost model achieved approximately:

- Beta-target pooled walk-forward AUC: `0.566`.
- Beta-target mean fold AUC: `0.557`.
- Beta-target holdout AUC: `0.592`.

The beta-target results are directionally positive but less consistent across folds. They support the primary result only partially and do not replace the primary target.

## 7. Interpretation and current conclusion

The complete-case result is materially better than the corrected all-row result. This indicates that median-imputing rows with no usable language features was likely diluting the language signal.

Historical company-relative sentiment is currently the strongest feature family. It performs better than raw sentiment, dictionary-only features, earnings-language proxies, and market-only controls in the complete-case evaluation.

Adding every available feature does not reliably improve generalization. The all-feature model is not automatically superior to the historical-surprise block.

Elastic Net is useful as a diagnostic and may remove some redundant variables, but it has not demonstrated a decisive advantage over Logistic Regression or XGBoost.

XGBoost improves ranking in some feature blocks, but the model-choice advantage is modest and should not be interpreted as proof that a more complex model created genuine economic signal.

The defensible project conclusion is:

> Earnings-call language contains modest, measurable predictive information, particularly when sentiment is normalized relative to a company’s own communication history. The signal survives corrected walk-forward testing and remains above random on the exploratory holdout, but it is not strong or stable enough to establish a proven trading strategy.

## 8. Known limitations

- Current-sector membership is not point-in-time historical sector membership.
- Timezone-naive timestamps rely on source wall-clock assumptions.
- Four walk-forward years provide limited regime coverage.
- Companies contribute multiple calls, so the effective independent sample is smaller than the event count.
- Complete-case results have fewer events and therefore higher variance.
- Historical z-scores are unavailable for early company calls with insufficient prior history.
- The final holdout has already been inspected and should remain exploratory rather than confirmatory.
- No reliable point-in-time consensus EPS or revenue-surprise data is available.
- Transcript earnings-language features must not be called true EPS surprises.
- Transaction-cost spreads are simple diagnostics, not a complete portfolio backtest.
- AUC confidence intervals do not fully account for selecting the winner from multiple feature/model candidates.

## 9. Reproducibility and artifacts

The main consolidated notebook is:

```text
notebooks/ai4all_final_model.ipynb
```

The lightweight complete-case notebook is:

```text
notebooks/complete_case_experiment.ipynb
```

The reusable package is uploaded to Colab as:

```text
earnings_intelligence_src.zip
```

The final artifact bundle contains:

- `model.joblib`.
- `feature_schema.json`.
- `feature_table.csv`.
- `predictions.csv`.
- `metrics.csv`.
- `feature_blocks.json`.
- `target_audit.csv` and `target_audit.json`.
- `run_manifest.json`.

The dashboard should use the standalone `artifacts/` directory generated by the selected final model. Old `earnings_model.pkl` and `dashboard_data.csv` files should not be treated as final artifacts because they were generated before the corrected target and complete-case pipeline.

## 10. Current next step

The E9 runner has now been implemented in `src/earnings_intelligence/experiments.py` and `notebooks/e9_multiverse_experiment.ipynb`. It performs the predeclared per-block Elastic Net/XGBoost comparison, writes tuning and selected-model artifacts, and can apply the frozen Industrials configurations to larger sector universes. No new E9 or multi-universe performance numbers should be considered authoritative until the Colab notebook has been run.

If no candidate improves the complete-case result materially and consistently across years, freeze the historical-surprise XGBoost model and move to the dashboard, feature explanations, and final presentation.

---

# Research appendix: complete experiment record

The sections below expand the summary into a research log. They distinguish experiments that were actually run from ideas that were discussed but not incorporated into the authoritative result. This distinction matters because many candidate feature blocks, models, targets, and evaluation views were examined; a high result among many candidates is not automatically evidence of a genuine signal.

## A. Research question, estimand, and unit of analysis

The unit of analysis is one earnings-call event for one publicly traded company. The intended prediction time is the moment at which the call becomes available to the market. The intended outcome is the sign of the company’s subsequent five-trading-session return after removing broad S&P 500 movement.

The primary estimand is:

```text
P(company has positive five-session abnormal return | information available at the call)
```

This is not a prediction of exact stock return, a price target, or long-run excess return. It is a ranking/classification problem. A model with AUC 0.60 can have modest ranking value even if threshold accuracy remains near 50%; conversely, high accuracy in one small holdout can occur without stable ranking skill.

The research questions are:

1. Does transcript language contain information beyond pre-event market controls?
2. Is that information stable across time, companies, and alternative abnormal-return definitions?

The project is not designed to establish a production trading strategy. It does not yet model portfolio construction, overlapping positions, liquidity, short-sale constraints, execution timing, tax, or a full transaction-cost/slippage process.

## B. Experiment ledger

| ID | Data/sample | Target | Features | Models | Validation | Status and purpose |
|---|---|---|---|---|---|---|
| E1 | Initial transcript/price sample | Continuous five-session abnormal return | Early sentiment/market variables | Linear Regression; XGBoost regression | Chronological train/test | Actually run; failed to explain return magnitude |
| E2 | Early filtered transcript sample | Sign of five-session abnormal return | Initial transcript sentiment | Logistic Regression; XGBoost classifier | Pre-2023 train, 2023+ test | Actually run; discovery only because the holdout was repeatedly inspected |
| E3 | Industrials, 2015+ baseline | Directional abnormal-return label | 11 transcript-level FinBERT/divergence features | Logistic Regression; XGBoost | Chronological holdout | Actually run; original baseline reference |
| E4 | Same baseline sample | Directional abnormal-return label | E3 plus momentum and volatility controls | Logistic Regression | Chronological holdout | Actually run; market-control diagnostic |
| E5 | Validation-audit snapshot | Directional five-session target | Sentiment and sentiment-plus-market variants | Standardized Logistic Regression | Holdout; bootstrap/permutation/economic diagnostics | Actually run; established the stricter audit standard |
| E6 | Industrials, 2015+, corrected all-row sample, 1,516 events | Adjusted-close market-subtracted direction | Market, baseline, sentence, dictionary, historical, earnings proxy, all-feature blocks | L2 Logistic; tuned Elastic Net; shallow XGBoost | 2019–2022 expanding walk-forward; 2023+ exploratory holdout | Actually run; corrected target and disjoint blocks |
| E7 | Same corrected sample | Beta-adjusted abnormal-return direction | Same E6 blocks | L2 Logistic; tuned Elastic Net; shallow XGBoost | Walk-forward and exploratory holdout | Actually run; robustness target |
| E8 | E6 complete-language subset, 886 events | Same corrected primary target | Same disjoint blocks, historical features recomputed after row removal | L2 Logistic; tuned Elastic Net; shallow XGBoost | 2019–2022 expanding walk-forward; 2023+ exploratory holdout | Actually run; current strongest and most comparable result |
| E9 | Implemented per-block hyperparameter comparison | Same primary target | Historical-surprise, sentence-sentiment, and all-feature blocks plus controls | Predeclared Elastic Net/XGBoost variations | Walk-forward only for selection | Implemented; awaiting Colab execution, no new authoritative metrics yet |
| E10 | Product prototype | No new target | Saved feature and prediction artifacts | Saved selected model | Offline artifact loading | Proposed deployment work; not evidence of additional signal |

### What was not treated as evidence

- An LLM-generated replacement for FinBERT was considered and tested informally, but it did not produce a reliable improvement and is not part of the final feature set.
- True EPS surprise, revenue surprise, analyst-consensus, market-cap, and point-in-time sector controls were not available with sufficiently reliable historical coverage. Transcript proxies were tested instead and are explicitly labeled proxies.
- Broad random hyperparameter search was not performed. The final code uses a small Elastic Net grid and conservative fixed XGBoost settings to reduce selection noise.
- The 2023–2025 holdout was inspected during development. It is retained as an exploratory generalization check, not as a clean confirmatory test.
- No claim is based on selecting the best single fold, best single holdout, or best metric among all reported metrics.

## C. Data lineage and filtering in detail

### C.1 Original corpus and why the final event count is smaller

The source corpus contains more than 33,000 transcript records across a much larger universe than the final experiment. The final count is smaller because filters are applied sequentially:

1. The transcript must have usable symbol, date, and content fields.
2. Very short transcripts are removed (`word_count > 500` in final preparation).
3. Extremely long or malformed records are removed (`word_count < 20,000`).
4. The symbol must map to the selected current sector.
5. The event must be in 2015 or later.
6. The call timestamp must be interpretable for event timing.
7. Intraday calls are excluded because daily OHLC data cannot isolate the exact post-call interval.
8. The stock and benchmark must have sufficient price history for the event window.
9. The complete-case diagnostic additionally requires both presentation and Q&A sentence-level sentiment.

The final experiment is not a claim about the entire 33,000-transcript corpus. It is a controlled Industrials event study. The sector restriction was retained for comparability with earlier work and to avoid expanding the data/compute problem late in the project.

### C.2 Sector scope

The final comparable universe is Industrials, 2015 onward. Sector labels come from current S&P 500 sector information with ticker fallbacks. This is a practical mapping, not a point-in-time historical classification. A company can therefore be assigned its present-day sector even when its historical classification or index membership differed.

This creates a possible measurement issue, but it is applied consistently within the corrected runs. Expanding to all sectors would be a separate experiment, not a silent improvement to the current result.

### C.3 Event audit history

Several audit snapshots exist because target code and sample rules were corrected over time:

| Audit snapshot | Rows after target construction | Companies | Positive rate | Interpretation |
|---|---:|---:|---:|---|
| Earlier validation audit | 1,598 | 77 | 52.88% | Historical snapshot before final reconciliation |
| Corrected all-row run | 1,516 | 67 | 50.86% | Authoritative corrected all-row sample |
| Complete-case run | 886 | 61 | 51.69% | Authoritative language-complete diagnostic sample |

The earlier 1,598-row audit must not be merged with the 1,516-row or 886-row results. The difference reflects changing target construction, price coverage, sector/ticker handling, and feature availability.

For the corrected all-row run, the event audit reported 2,774 candidate rows before phase filtering, 1,258 intraday exclusions, and 1,516 retained events. No missing-price or invalid-ticker exclusions were reported. The retained phase mix was 1,238 pre-open and 278 after-close events.

The complete-case run began with the corrected 1,516-row frame and retained 886 rows where both `pres_sent_mean` and `qa_sent_mean` were non-missing. It removed 630 rows for language incompleteness, not because their target was invalid. The retained phase mix was 731 pre-open and 155 after-close events.

### C.4 Timestamp and price rules

Event year is derived from the parsed call timestamp, not a separate transcript year field. Timezone-naive timestamps are interpreted according to the source wall-clock convention and New York market-hour thresholds. This is a stated assumption because the source records do not consistently provide an explicit timezone.

The corrected price rule uses adjusted close for both company and S&P 500 benchmark. Earlier code used adjusted close for the company but could use unadjusted close for the benchmark; that inconsistency was corrected before the authoritative runs.

For a pre-open call, the first eligible trading day is the call date. For an after-close call, the first eligible trading day is the next trading day. The five-session return uses the first eligible session through the fifth eligible session, with the preceding available session as baseline. This avoids treating a pre-call overnight gap as post-call information and avoids unobservable intraday windows.

## D. Transcript processing and FinBERT details

### D.1 Structural parsing

Structured transcript turns are scanned for the operator/question-and-answer marker. Text before that marker is treated as presentation; text after it is treated as Q&A. This allows separate modeling of prepared remarks and analyst questioning, but it depends on the source’s section markers. Missing or malformed markers can create incomplete language rows.

### D.2 Original baseline scoring

The original baseline passed each complete presentation and Q&A text through FinBERT using tokenizer overflow chunks with a maximum sequence length of 512 tokens. Chunk probabilities were averaged to obtain one presentation-level and one Q&A-level probability vector. This was computationally simple but coarse: it discarded within-call position, dispersion, and sentence-level changes in tone.

### D.3 Rich sentence-level scoring

The rich pipeline splits text using punctuation/newline boundaries and ignores sentences shorter than 20 characters. Each sentence is truncated to a maximum of 128 tokens and scored by FinBERT. Sentence inference is batched; the implementation flattens all presentation and Q&A sentences into a continuous stream of batches, then restores scores to their original call and section positions.

This is sentence-based, not random fixed-size text chunking. Batch size changes GPU throughput and memory behavior only; it does not change the statistical unit or randomly partition the transcript. Very long individual sentences are truncated at the tokenizer limit, but sentence ordering is preserved.

### D.4 Missing sections and warnings

If a section has no valid sentences, sentence-level features are stored as unavailable rather than silently converted to zero. The earlier implementation emitted “mean of empty slice” warnings for short or empty sections; this was corrected so unavailable beginning/middle/end groups are represented as missing values. The complete-case experiment then explicitly removes rows with missing presentation or Q&A sentiment for its diagnostic comparison.

## E. Complete feature inventory and block design

The final blocks are mutually exclusive. A feature belongs to one primary block only; `all_features` is the union of those blocks. This prevents, for example, a market variable from appearing in both `market_only` and a nominally transcript-only block.

### E.1 `market_only`

| Feature | Definition and intended role |
|---|---|
| `momentum_5d` | Recent five-session company price movement before the event |
| `momentum_20d` | Recent twenty-session company price movement before the event |
| `volatility_20d` | Recent twenty-session company return volatility |
| `market_momentum_20d` | Recent twenty-session S&P 500 movement |
| `beta_120d` | Pre-event company sensitivity to the S&P 500 over a 120-session window, subject to a minimum-observation rule |

This block answers whether the model can rank outcomes using market state alone. It is the control against which transcript blocks should be judged.

### E.2 `baseline_sentiment`

| Feature family | Features |
|---|---|
| Presentation probabilities | `pres_pos`, `pres_neg`, `pres_neu` |
| Q&A probabilities | `qa_pos`, `qa_neg`, `qa_neu` |
| Net tone | `pres_net_sentiment`, `qa_net_sentiment` |
| Divergence | `sentiment_mismatch_pos`, `sentiment_mismatch_neg` |
| Evasion | `evasion_index` |

The probability features are section-level averages. Net sentiment is positive probability minus negative probability. Mismatch features compare presentation tone with Q&A tone. The evasion feature is a hand-designed interaction intended to represent positive presentation language combined with neutral Q&A language; it is a heuristic, not a validated psychological construct.

### E.3 `sentence_sentiment`

For both `pres` and `qa`, the sentence-level block contains:

- `sent_mean`, `sent_std`, `sent_p10`, and `sent_p90` for net sentiment.
- `pos_mean`, `neg_mean`, and `neutral_mean`.
- `pos_frac` and `neg_frac`, where a sentence is assigned a directional fraction when that class probability exceeds the other directional class.
- `entropy`, measuring uncertainty in the three-class FinBERT probability vector.
- `begin_mean`, `middle_mean`, and `end_mean`, based on ordered thirds of the section.
- `slope`, a linear trend in net sentiment over sentence position.
- `n_sentences`, the number of retained sentences.

The block also includes Q&A-minus-presentation differences for mean sentiment, standard deviation, entropy, negative fraction, positive fraction, and slope. These positional features are intended to capture whether tone deteriorates, improves, or becomes more uncertain through a call.

### E.4 `financial_dictionary`

The local Loughran–McDonald-style dictionary is read from `lm_dictionary.csv`. The loader ignores metadata columns such as sequence identifiers and accepts recognized category columns. Rates are normalized by token count, with separate presentation, Q&A, and difference features. The tested categories are:

- Positive language.
- Negative language.
- Uncertainty.
- Litigious language.
- Strong modal language.
- Weak modal language.
- Constraining language.
- Complexity.
- Token counts.

The dictionary block is separate from FinBERT. It tests whether domain-specific lexical categories add information that a three-class sentiment model misses.

### E.5 `historical_surprise`

Historical features are expanding, prior-only transformations. For a raw call feature (x_t), the company-relative score is:

\[
z_t = \frac{x_t - \mu(x_1,\ldots,x_{t-1})}{\sigma(x_1,\ldots,x_{t-1})}
\]

The current call is excluded from the mean and standard deviation. At least four prior company observations are required. If company history is insufficient, a prior sector-level fallback can be used; early observations remain identifiable through history-count/source fields. The features include z-scores and prior-history counts for:

- Presentation sentiment mean.
- Q&A sentiment mean.
- Presentation entropy.
- Q&A entropy.
- Presentation negative fraction.
- Q&A negative fraction.
- Presentation sentiment slope.
- Q&A sentiment slope.

This block measures surprise/novelty in communication style. A company that is unusually negative relative to its own past may differ from a company that is always moderately negative. It does not use realized returns and is not an EPS surprise.

### E.6 `earnings_language_proxy`

The transcript proxy block counts or flags language related to earnings and guidance:

- EPS/per-share mentions and mention rates.
- Beat and miss language.
- Above-expectations and below-expectations language.
- Guidance-up, guidance-down, and guidance-maintained indicators.
- Forward-looking language.

These features are closer to earnings-event semantics than generic sentiment, but they are only text mentions. A sentence saying “we beat expectations” is not a verified surprise without contemporaneous analyst consensus, the reporting period, and exact realized values.

### E.7 `all_features`

`all_features` is the union of the six blocks above after duplicate-name removal. Raw transcript text, structured transcript fields, target labels, abnormal-return columns, future-return aliases, and post-event variables are excluded. The final pipeline checks for target-like names before modeling and fails if such a column enters the feature union.

## F. Model specifications and what each model tests

### F.1 Baselines

Two non-predictive baselines are reported:

- Majority-class classification: always predicts the most common class.
- Train-rate probability: predicts the training-set positive rate for every test event.

Because labels are close to 50/50, majority accuracy is only slightly above 50%, and a random probability near 0.5 has log loss near 0.693 and Brier score near 0.25. These baselines distinguish a small ranking effect from a model that merely exploits prevalence.

### F.2 Linear Regression and XGBoost regression

The initial continuous-return models were Ordinary Linear Regression and XGBoost regression. Both produced approximately zero or negative out-of-sample R². This showed that the exact magnitude of a five-session abnormal return was dominated by noise at the available sample size and feature resolution, motivating directional classification.

### F.3 Standardized L2 Logistic Regression

The transparent baseline is a pipeline with:

1. Training-fold median imputation.
2. Training-fold correlation pruning at a 0.95 absolute-correlation threshold.
3. Training-fold standardization.
4. Logistic Regression with L2 penalty, `C=1.0`, and `max_iter=2000`.

It produces stable probabilities and is easy to interpret. Correlation pruning is performed separately inside each training fold so the test fold cannot influence feature selection.

### F.4 Elastic Net Logistic Regression

Elastic Net tests whether a sparse or partially sparse linear model can suppress redundant/noisy language variables. The fixed grid is:

```text
C = {0.01, 0.1, 1.0, 10.0}
l1_ratio = {0.1, 0.5, 0.9}
```

The search is performed using primary-target walk-forward results. The selected setting is then evaluated against L2 Logistic and XGBoost. L1 regularization can remove weak coefficients, but it cannot manufacture information absent from the features; with correlated features, it may select one arbitrary representative of a group. A better Elastic Net result therefore needs to be stable across folds and blocks, not merely sparse.

### F.5 Shallow XGBoost

The conservative comparison uses approximately:

```text
n_estimators = 150
max_depth = 2
learning_rate = 0.03
min_child_weight = 10
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

The model is intentionally shallow and regularized because each training fold contains only hundreds of events from a limited number of companies. It can capture interactions and nonlinear thresholds that Logistic Regression cannot, but its flexibility increases selection risk.

### F.6 Hyperparameter and model-selection policy

The policy is:

1. Evaluate the model ladder on predefined blocks.
2. Select using mean walk-forward AUC across 2019–2022.
3. Use mean walk-forward log loss as the tie-breaker.
4. Run the exploratory holdout only after the primary selection rule is fixed.
5. Do not choose a model because it has the highest holdout AUC.

The code uses reduced bootstrap repetition during search and a larger clustered bootstrap for the selected primary model. This saves compute while preserving a more complete uncertainty estimate for the reported winner.

## G. Detailed results by feature block

The following tables summarize the corrected runs. “Mean walk” averages the four yearly walk-forward AUCs, while “pooled walk” combines out-of-fold predictions before computing one AUC. The highest holdout value in a row is not a selection criterion.

### G.1 Corrected all-row experiment: best observed model by block

| Block | Comparison model | Mean walk AUC | Pooled walk AUC | Representative holdout AUC |
|---|---|---:|---:|---:|
| Market-only | Logistic | 0.508 | 0.504 | 0.498 |
| Baseline sentiment | Logistic | 0.564 | 0.545 | 0.530 |
| Sentence sentiment | XGBoost | 0.588 | 0.555 | 0.547 |
| Financial dictionary | Logistic | 0.550 | 0.518 | 0.518 |
| Historical surprise | XGBoost | 0.576 | 0.568 | 0.557 |
| Earnings-language proxy | Logistic | 0.520 | 0.500 | approximately 0.549 |
| All features | XGBoost | 0.566 | 0.556 | 0.548 |

The selected all-row winner was sentence-sentiment XGBoost because selection used mean walk-forward AUC. Historical-surprise XGBoost had a higher pooled AUC than its mean fold score, illustrating why both aggregate views are reported.

Selected sentence-XGBoost primary results:

- Fold AUCs: 2019 `0.633`, 2020 `0.541`, 2021 `0.603`, 2022 `0.575`.
- Mean fold AUC: `0.588`.
- Pooled walk AUC: `0.555`.
- Cluster bootstrap interval: approximately `0.519–0.593`.
- Pooled balanced accuracy: `0.567`.
- Pooled MCC: `0.160`.
- Average precision: `0.568`.
- Log loss: `0.694`.
- Brier score: `0.250`.
- Top-minus-bottom spread: `0.0228`.
- After-cost spread: `0.0208` under the pipeline’s simple assumed cost.
- Exploratory holdout AUC: `0.547`.

These values show weak but somewhat repeatable walk-forward ranking in some folds, followed by weaker later holdout performance. They do not show that every sentence feature is useful.

### G.2 Complete-case experiment: best observed model by block

| Block | Strongest walk-forward comparison | Mean walk AUC | Pooled walk AUC | Representative holdout AUC |
|---|---|---:|---:|---:|
| Market-only | Logistic | 0.552 | 0.543 | 0.473 |
| Baseline sentiment | Logistic | 0.576 | 0.572 | 0.633 |
| Sentence sentiment | XGBoost | 0.604 | 0.600 | 0.631 |
| Financial dictionary | Logistic | 0.546 | 0.548 | 0.614 |
| Historical surprise | XGBoost | 0.611 | 0.622 | 0.608* |
| Earnings-language proxy | Elastic Net | 0.552 | 0.536 | 0.643 |
| All features | Elastic Net | 0.597 | 0.605 | 0.632 |

\*The selected complete-case historical-surprise XGBoost holdout AUC was `0.596`; the approximately `0.608` value in the compact block comparison refers to a different model entry. It is retained to show the breadth of the evaluated table. The selected winner is determined by walk-forward performance, not holdout performance.

The complete-case historical-surprise XGBoost winner had:

| Metric | Value |
|---|---:|
| Events | 886 |
| Companies | 61 |
| Positive rate | 51.69% |
| Mean fold AUC | 0.611 |
| Pooled walk AUC | 0.622 |
| Clustered 95% AUC interval | 0.558–0.685 |
| Pooled average precision | 0.642 |
| Pooled balanced accuracy | 0.584 |
| Pooled MCC | 0.167 |
| Pooled log loss | 0.676 |
| Pooled Brier score | 0.241 |
| Top-minus-bottom spread | 0.0345 |
| After-cost spread | 0.0325 |
| Exploratory holdout AUC | 0.596 |
| Holdout average precision | 0.555 |
| Holdout balanced accuracy | 0.559 |
| Holdout MCC | 0.119 |
| Holdout log loss | 0.690 |
| Holdout Brier score | 0.247 |
| Holdout top-minus-bottom spread | 0.0389 |
| Holdout after-cost spread | 0.0369 |

The selected walk-forward feature schema was:

```text
pres_sent_mean_z
pres_sent_mean_history_count
qa_sent_mean_z
pres_entropy_z
qa_entropy_z
pres_neg_frac_z
qa_neg_frac_z
pres_slope_z
qa_slope_z
```

The selected schema is concentrated in company-relative sentiment statistics. This is consistent with the broader block result: historical normalization is more promising than simply adding every raw language measurement.

### G.3 Primary-target fold table for the selected complete-case winner

| Test year | Events in test fold | AUC | Interpretation |
|---:|---:|---:|---|
| 2019 | 53 | 0.621 | Above-random ranking |
| 2020 | 74 | 0.624 | Above-random ranking |
| 2021 | 66 | 0.621 | Above-random ranking |
| 2022 | 83 | 0.580 | Weaker but still above random |

No selected primary fold is below 0.50. This is encouraging, but four small temporal folds do not provide broad regime coverage and the observations remain clustered by company.

### G.4 Beta-adjusted target results

For the complete-case historical-surprise XGBoost comparison:

- Fold AUCs were approximately `0.574`, `0.588`, `0.598`, and `0.467` for 2019–2022.
- Mean fold AUC was approximately `0.557`.
- Pooled walk AUC was approximately `0.566`.
- Clustered interval was approximately `0.465–0.618`.
- Exploratory holdout AUC was approximately `0.592`.

This robustness target is directionally positive in aggregate but less stable, particularly in 2022. It partially supports the primary result but does not replace it.

## H. Original baseline versus corrected/richer experiments

The original baseline was approximately 883 events, with walk-forward AUC `0.585` and exploratory holdout AUC `0.611`. The corrected all-row experiment had a similar mean walk AUC (`0.588`) but lower pooled walk AUC (`0.555`) and holdout AUC (`0.547`). The complete-case experiment improved to mean walk AUC `0.611`, pooled walk AUC `0.622`, and holdout AUC `0.596`.

The correct interpretation is not simply “more features improved the model.” Several things changed simultaneously:

- The target was reconstructed with consistent adjusted prices.
- Intraday events were excluded explicitly.
- Historical transformations were recomputed using prior observations only.
- The corrected all-row sample changed from the earlier approximately 883-event sample to 1,516 events.
- The all-row run median-imputed 630 missing-language rows.
- The complete-case run removed those rows, leaving 886 events.
- The feature blocks and model-selection rule became stricter.
- The final winner was selected by mean walk-forward AUC rather than by whichever holdout or aggregate number looked best.

The complete-case improvement is promising but not a clean causal estimate of the value of every new feature. It may reflect both better feature availability and sample-selection effects. Complete rows may correspond to companies, years, or transcripts with cleaner structure and longer sections. A professional follow-up should compare complete and incomplete groups’ year/company distributions and treat the complete-case result as a diagnostic, not an unbiased estimate for all calls.

## I. Leakage, causality, and statistical-rigor audit

### I.1 Controls implemented

- Event year is derived from call timestamp.
- Intraday calls are excluded.
- Pre-open and after-close windows use distinct first tradable sessions.
- Company and benchmark returns use the same adjusted-price basis.
- Pre-call market features use only data before the event window.
- Historical z-scores use expanding prior observations only.
- Correlation pruning, imputation, and scaling are fit inside each training fold.
- The language cache is untrusted for target construction; targets are rebuilt from fresh price data.
- Target-like and future-return column names are screened from feature blocks.
- The model winner is selected from primary-target walk-forward results, not holdout results.

### I.2 Remaining threats

- Current sector labels are not point-in-time.
- Source timestamps may be timezone-naive or inconsistently formatted.
- Multiple calls from one company are not independent observations.
- Multiple candidate blocks/models and repeated exploratory analysis create researcher degrees of freedom.
- Complete-case inclusion may be informative rather than random.
- Bootstrap intervals cluster by company but do not fully account for model-selection uncertainty, temporal dependence, or all prior experimentation.
- Five-session windows can overlap for calls close together in time.
- The simple top/bottom spread is not a full implementable portfolio simulation.
- The 2023+ holdout has already influenced project decisions and is no longer a clean final test.

### I.3 How to read the uncertainty interval

The clustered interval for the selected complete-case winner is approximately `0.558–0.685`, which excludes 0.50 in the reported bootstrap calculation. This is encouraging, but it should not be described as a definitive hypothesis test because:

1. The model was selected after comparing multiple blocks and models.
2. The interval conditions on the selected winner rather than accounting for selection.
3. Company clustering addresses repeated-company dependence but not every time-series or selection issue.
4. Only four walk-forward years are available.

The appropriate conclusion is “compatible with a modest, temporally repeated ranking signal under this research design,” not “the strategy is statistically proven.”

## J. Interpretation of the metrics

The selected complete-case walk-forward AUC of approximately 0.622 is materially above random ranking but still modest. It does not mean that the model is correct 62.2% of the time. At a threshold chosen to maximize accuracy, classification accuracy could remain near the mid-50s.

The pooled log loss of approximately 0.676 is better than the 0.693 random-probability reference, and the Brier score of approximately 0.241 is better than the approximately 0.25 reference. The improvement is positive but small in absolute probability quality. The MCC of approximately 0.167 also indicates a weak relationship rather than a strong classifier.

The top-versus-bottom spread is more economically interpretable than AUC, but the reported 3.45 percentage-point walk-forward spread is a probability-group diagnostic, not a complete strategy return. It does not yet include realistic turnover, bid-ask spreads, market impact, overlapping positions, or a predeclared portfolio rule.

The complete-case holdout AUC of approximately 0.596 is lower than its pooled walk-forward AUC. This is consistent with ordinary sampling noise, regime change, sample differences, or development overfit. It remains above random in this run but should temper the strength of the claim.

## K. Reproducibility map

### K.1 Main files

| File | Role |
|---|---|
| `notebooks/ai4all_baseline_model.ipynb` | Original exploratory pipeline |
| `notebooks/ai4all_final_model.ipynb` | Corrected one-pass final experiment |
| `notebooks/complete_case_experiment.ipynb` | Complete-language diagnostic and artifact builder |
| `src/earnings_intelligence/events.py` | Event timing, price windows, targets, market features |
| `src/earnings_intelligence/text_features.py` | Transcript parsing, sentence FinBERT, dictionary, language proxies |
| `src/earnings_intelligence/features.py` | Prior-only historical feature construction |
| `src/earnings_intelligence/modeling.py` | Preprocessing, Logistic, Elastic Net, XGBoost, metrics |
| `src/earnings_intelligence/final_pipeline.py` | One-pass orchestration and artifact generation |
| `src/earnings_intelligence/artifacts.py` | Standalone model/dashboard artifact serialization |
| `earnings_intelligence_src.zip` | Colab-uploadable package archive |
| `lm_dictionary.csv` | Local financial-language lexicon |

### K.2 Artifact interpretation

The experiment output directory contains:

- `feature_table.csv`: event-level features and offline target/return fields.
- `predictions.csv`: fold and holdout probabilities with event identifiers.
- `metrics.csv`: per-fold, aggregate, holdout, baseline, and beta-target results.
- `feature_blocks.json`: exact column membership of every block.
- `target_audit.csv` and `target_audit.json`: event exclusions, target basis, counts, and phase information.
- `run_manifest.json`: model version, target/cache versions, selection rule, and sample metadata.
- `model.joblib`, `feature_schema.json`, and associated files under standalone `artifacts/`: saved model and dashboard-facing schema.

The old `earnings_model.pkl` and `dashboard_data.csv` are historical artifacts from the pre-correction workflow. They should not be mixed with the final artifact directory.

### K.3 Reproduction principle

A reproducible rerun should use a fixed random seed, the same source data or language cache, the same price-download date range, the same package version, and the same model configuration. It should verify that:

- feature block names and column lists are unchanged;
- no target-like feature is present;
- event counts and phase counts match;
- selected model probabilities are identical or numerically equivalent;
- the dashboard loads from the artifact directory without the notebook namespace or network access.

## L. Final research judgment and decision rule

The evidence supports a narrow claim:

> Within the restricted Industrials sample and under the corrected event-study design, company-relative earnings-call language produced a modest above-random ranking of five-session abnormal-return direction. The effect was most visible in the complete-language sample and was weaker under the beta-adjusted target and exploratory later holdout.

The evidence does not support stronger claims that:

- the model predicts exact returns;
- all richer language features are useful;
- LLM-generated features are necessary or superior;
- EPS surprise is being measured;
- the strategy is profitable after realistic costs;
- the holdout result is confirmatory;
- the current-sector sample generalizes to all listed companies;
- the selected AUC interval fully accounts for searching across models and features.

The appropriate stopping rule is practical and statistical: run one final predeclared comparison of the historical-surprise, sentence-sentiment, and all-feature blocks; select only by 2019–2022 mean walk-forward AUC with log-loss tie-break; inspect the holdout once for reporting; then freeze. If the result remains around 0.60–0.62 walk-forward AUC without stable improvement, the contribution is the careful measurement and interpretation of a modest signal, not the pursuit of an increasingly complex model.

## M. Historical branches and non-final experiments

This section records exploratory paths that are easy to lose when focusing only on the current winner.

### M.1 Original data preparation

The initial notebook loaded the Hugging Face dataset into memory, filtered transcripts using word count, mapped symbols to sectors using an online S&P 500 table plus ticker fallbacks, and selected Industrials. This approach was convenient but expensive in Colab. Attempts to materialize the full corpus contributed to RAM problems, which led to a streaming loader that scans for candidate symbols and retains only the selected sector.

The original preparation also used a transcript `year` field in parts of the workflow. The corrected pipeline instead derives `event_year` from `call_datetime`, because the call timestamp is the relevant event-time field.

### M.2 Initial FinBERT feature set

The original 11-feature sentiment vector was:

```text
pres_pos
pres_neg
pres_neu
qa_pos
qa_neg
qa_neu
sentiment_mismatch_pos
sentiment_mismatch_neg
evasion_index
pres_net_sentiment
qa_net_sentiment
```

This was the baseline against which market controls and richer language blocks were compared. The probability triplets are mathematically constrained because each section’s positive, negative, and neutral probabilities sum approximately to one. The later feature audit checked rank and redundant identities; the baseline matrix was full rank after the selected feature construction, but several variables remained conceptually redundant.

### M.3 Market-plus-sentiment comparison

The earlier validation audit compared:

- `sentiment`: the 11 transcript sentiment/divergence variables.
- `sentiment_plus_market`: the sentiment variables plus recent momentum, volatility, and related controls.

In the earlier final-holdout snapshot, sentiment-only standardized Logistic Regression had accuracy `0.567`, balanced accuracy `0.570`, AUC `0.577`, average precision `0.563`, log loss `0.691`, and Brier `0.249`. The train-rate baseline had AUC `0.500`. The sentiment-plus-market version had accuracy `0.550`, balanced accuracy `0.553`, AUC `0.581`, average precision `0.561`, log loss `0.690`, and Brier `0.248`. The market controls did not create a large incremental improvement in that snapshot.

The same audit reported a permutation p-value of approximately `0.012` for sentiment-only AUC and `0.009` for sentiment-plus-market AUC. These should be treated as diagnostics from one inspected holdout, not as final confirmatory p-values. The earlier audit also reported a top/bottom economic spread of about `0.036` for sentiment-only and `0.029` for sentiment-plus-market after its assumed cost calculation.

### M.4 Exploratory volatility classification

One branch changed the outcome from direction to magnitude. It defined:

```text
movement_30d = abs(abnormal_return_30d)
```

Training-only 33rd and 67th percentiles were used to retain low- and high-movement cases while discarding the middle third. This tested whether transcript language could distinguish calm from volatile post-event outcomes. It was not carried into the final research contract because it changes the question, reduces sample size, and is less directly aligned with the project’s main directional objective.

### M.5 Early classifier settings

The original classification notebook used a simple chronological split and compared Logistic Regression with an XGBoost classifier. The early XGBoost configuration was approximately 100 trees, depth 3, learning rate `0.03`, subsampling `0.8`, column subsampling `0.7`, minimum child weight `10`, and a small gamma regularizer. Later experiments moved to a fixed, shallower 150-tree configuration with depth 2 and explicit fold-safe preprocessing.

### M.6 Rich-feature implementation problems that were corrected

During the rich experiment, several engineering issues affected execution and comparability:

- Financial dictionary metadata fields such as values like `12of12inf` were initially passed into numeric conversion; the loader was restricted to recognized category columns.
- Duplicate feature names caused pandas correlations to return DataFrames rather than scalars, producing an “ambiguous truth value” error; requested feature names are now deduplicated before correlation pruning.
- Older caches contained target aliases and stale market/history features; the final pipeline removes these and rebuilds targets/derived features.
- Sentence-level sections could be empty, producing invalid mean warnings; empty groups are now represented as missing.
- The rich experiment could be computationally expensive because FinBERT inference was repeated over many sentences; cached language features and continuous batched inference were added.

These were implementation defects, not evidence that the model itself was failing. They are documented because a reproducible researcher needs to know which outputs came from corrected code.

### M.7 Why more features did not automatically improve results

The rich blocks differ in statistical character:

- Market variables are low-dimensional but can be regime-dependent.
- Raw FinBERT probabilities are broad and correlated.
- Sentence statistics create many noisy summaries from the same transcript.
- Dictionary rates are sparse and depend on word-tokenization choices.
- Historical z-scores are more company-specific but unavailable or unstable early in a company’s history.
- Earnings-language proxies are semantically targeted but do not measure actual surprise.

Adding these families to `all_features` increases the number of possible splits and coefficient tradeoffs. It can improve in-sample fit or one holdout while reducing temporal stability. The observed results are therefore consistent with a low signal-to-noise setting in which representation and sample quality matter more than broad model complexity.

## N. Suggested researcher-facing reporting table

For the final presentation or paper-style report, the following compact table should be treated as the headline evidence. It keeps the samples separate and prevents holdout tuning from being implied.

| Evidence layer | Sample | Selection rule | Result | Interpretation |
|---|---|---|---:|---|
| Original baseline | ~883 events | Earlier baseline workflow | Walk AUC 0.585; holdout AUC 0.611 | Historical reference; not fully comparable |
| Corrected all-row primary | 1,516 events, 67 companies | Mean 2019–2022 walk AUC | Mean 0.588; pooled 0.555; holdout 0.547 | Weak signal with imputed language rows |
| Complete-case primary | 886 events, 61 companies | Mean 2019–2022 walk AUC | Mean 0.611; pooled 0.622; holdout 0.596 | Strongest current evidence; possible selection effect |
| Complete-case robustness | 886-ish beta-valid events | Same selection rule | Mean beta walk 0.557; pooled 0.566; holdout 0.592 | Partial support under alternate target |

The headline should be accompanied by per-year folds, company count, positive rate, clustered interval, and the complete-case caveat. A single AUC without those details would overstate the evidence.
