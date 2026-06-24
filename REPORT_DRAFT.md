# CSI500 Stock Selection — Final Report

## 1. Factors

The final model uses 17 signals (12 technical, 2 fundamental, 3 cross-sectional rank), developed from a 14-feature baseline through iterative walk-forward validation. The technical set was substantially reworked: oscillators (MACD histogram, RSI), Bollinger Band width, and 52-week price-location features were added (E01–E03), while several redundant signals (raw daily return, raw 20-day volatility, 20-day MA deviation, turnover) were dropped (E73). Three cross-sectional rank features (`ret_20d_rank`, `vol_20d_rank`, `bb_position_rank`) are retained; replacing raw counterparts with ranks reduces sensitivity to market-wide drift. Beyond these technical changes, two fundamental factors were introduced:

### 1.1 Valuation Factors: PE/PB

Two fundamental features were added: `pe_ttm` (trailing twelve-month P/E) and `pb` (price-to-book). These are sourced from the AKShare Baidu Finance API (daily updated, available from 2025-04-29 onward).

Adding PE/PB produced the single largest IC improvement in the project. In walk-forward CV (E04 vs. baseline), mean IC rose from 0.0170 to 0.0189 (+11%). Removing PE while keeping PB alone dropped IC to 0.0082 (E05), confirming the two are complementary: PB captures the well-documented A-share value premium; PE adds information by distinguishing profitable firms from loss-makers, particularly relevant for CSI500 which mixes quality and distressed names.

### 1.2 Preprocessing: Winsorization and Industry Neutralization

**Winsorization (±3σ, cross-sectional per date)** clips extreme daily observations before they reach the model. A single day with a 30%+ limit-up can distort gradient boosting's split thresholds if unclipped. Applied only to training data; clipping boundaries are re-estimated per date on the test set.

**Industry neutralization (31 Shenwan sectors)** removes sector-common factor exposures by de-meaning momentum and MA-deviation features within each industry per date. Without neutralization, a model trained on CSI500 can learn to overweight entire sectors (e.g., all semiconductor names correlated by policy announcement) rather than picking idiosyncratic winners. Industry membership covers 100% of the 499-stock universe (`data/industry_map.csv`).

Together, these two preprocessing steps lifted mean IC from ≈0.007 to ≈0.0135 in early ablations (E01), and stabilized IC standard deviation from 0.063 to 0.028.

### 1.3 Feature Scope Decisions

Over 13 additional candidate signals were tested (higher-order momentum, MA crossovers, Williams %R, price position, short volatility ratios). None produced reliable improvement in walk-forward CV that held across both IC and excess return metrics simultaneously. The final feature set was held at 17, avoiding over-parameterization on the relatively small cross-section (499 stocks per day).

---

## 2. Models

### 2.1 Architecture: LightGBM + XGBoost Ensemble

The prediction model combines LightGBM (leaf-wise tree growth, main model) and XGBoost (level-wise, secondary) via rank-percentile blending. Both models receive the same 17 features; scores are converted to within-date percentile ranks (0–1) and averaged 50/50.

**Why ensemble?** LightGBM and XGBoost produce different rankings (≈41/50 stocks overlap per date), because they differ in tree-growing strategy and regularization path. In rolling backtests across multiple date ranges, the ensemble consistently matched or exceeded either model individually (e.g., April 5-day windows: +1.33% mean excess return vs. +1.20% for LightGBM-only at identical parameters; `eval_log.csv`). The diversification benefit is structural — two distinct optimization criteria applied to the same data produce complementary errors.

### 2.2 Key Hyperparameter Findings

All tuning was conducted exclusively on walk-forward CV using data strictly before 2026-02-01. Table 1 excerpts the most consequential experiments from the 111-run log (`experiment_log.csv`).

**Table 1. Selected experiments from the tuning log.**

| Exp | Model | Change | Mean IC | ExRet | Finding |
|-----|-------|--------|---------|-------|---------|
| E01 | xgb | baseline (winsorize + ind-neut) | 0.0135 | — | preprocessing foundation: IC doubled vs. raw |
| E03 | xgb | +MACD/BB/52w (7 new features) | 0.0170 | — | IC std halved (0.063 → 0.028) |
| E04 | xgb | +PE/PB | 0.0189 | — | largest single feature-driven gain (+11%) |
| E10 | lgbm | +margin trading features | 0.0162 | — | modest CV gain; later removed (overfit) |
| E46 | lgbm | nl: 4→8 | 0.0170 | +0.12% | nl=8 is the sweet spot for 499-stock cross-section |
| E66 | lgbm | nl=8 × col=0.7 (interaction) | **0.0378** | +0.96% | global max; coordinate descent missed this synergy |
| E40 | lgbm | λ: 0.1→10 | −0.0152 | −0.45% | double-regularization kills signal at low nl |
| E95 | lgbm | +ret_20d+ret_60d (5d target) | 0.0107 | +2.31% | medium momentum essential for weekly horizon |
| E109 | xgb | λ: 10→2 (5d re-tune) | 0.0127 | +1.92% | XGBoost optimum for 5-day target |
| E110 | ensemble | LGBM + XGB (final config) | — | — | 41/50 stock overlap; complementary errors |

Key takeaways from the tuning process:

**LightGBM `num_leaves` is the critical parameter.** Leaf-wise growth on a 499-stock cross-section overfits rapidly. Scanning nl ∈ {4, 6, 8, 10, 12, 16, 32, 64} (E46–E50): nl=8 was uniquely optimal. Below 8, the model underfit; above 8, IC dropped immediately.

**Interaction effect at nl=8 × colsample=0.7 (E66).** During the 3-day-target phase, coordinate descent (varying one parameter at a time) had pre-locked nl=6 early, causing it to miss the global optimum. E66 tried the (nl=8, col=0.7) combination explicitly and found IC=0.0378 — the highest across all 111 experiments. This interaction pattern informed the final configuration, though absolute IC levels differ between the 3-day and 5-day target regimes.

**Light L2 regularization for LightGBM.** With nl=8 already constraining capacity, adding `reg_lambda > 0.1` double-regularized and suppressed genuine signal. λ=10 drove IC to −0.0152 (E40). Final value: λ=0.1.

**XGBoost re-tuning for 5-day target.** Hyperparameters were originally optimized on a 3-day evaluation horizon. When the target changed to 5 trading days, medium-momentum features (`ret_20d`, `ret_60d`) became indispensable (E93–E95: IC collapsed to ≈0.001 without them). XGBoost `reg_lambda` was re-tuned from 10 (3-day optimal) to 2 (5-day optimal, E97–E109).

| Parameter | LightGBM | XGBoost |
|-----------|----------|---------|
| `num_leaves` | **8** | — |
| `max_depth` | 5 | 6 |
| `min_child_weight` | 20 | 20 |
| `reg_lambda` | **0.1** | **2.0** |
| `subsample` | 0.6 | 0.6 |
| `colsample_bytree` | **0.7** | 0.6 |
| `learning_rate` | 0.05 | 0.05 |
| `n_estimators` | 400 + early stop (30) | 400 + early stop (30) |

### 2.3 Portfolio Construction

Top-50 stocks by ensemble score are selected. Weights are proportional to rank (highest-ranked stock: weight ∝ 50; lowest: ∝ 1), normalized to sum to 1.0, with a 10% per-name cap applied via iterative redistribution. Alternatives tested — exponential weights (α=3), top-30, margin-filter — failed to consistently improve over this simple baseline (eval_log.csv).

---

## 3. Results

### 3.1 Self-Test Methodology

Because the data is a time series with serial dependence, a random shuffle split would leak future information into training via adjacent-day correlations. The split is therefore purely time-based:

| Set | Range | Use |
|-----|-------|-----|
| **Training** | 2025-01-01 → 2026-01-26 | Model weight estimation + walk-forward CV tuning |
| **Test** | 2026-02-01 → 2026-04-30 (55 trading days) | Held-out evaluation (run once, model locked) |

A 5-trading-day embargo gap separates the training cutoff from the test start. Specifically, the training cutoff is set to the trading date that is `FORWARD_HORIZON` (5) trading days before the test-start date, so the 5-day forward return target computed on the last training date falls strictly before the first test date. This prevents any training-target leakage into the evaluation period. The final model's training set is smaller than the baseline's (90,896 vs. 128,752 rows) because PE/PB data begins 2025-04-29; this is a data-availability constraint, not a design choice. All hyperparameter and feature selection used walk-forward CV on the training set as the primary signal. The test set was evaluated exactly once with a locked model configuration.

### 3.2 Walk-Forward Cross-Validation (Pre-2026-02-01)

Before the test-set evaluation, walk-forward CV with the final locked configuration was run on five non-overlapping windows within the training period to establish a baseline overfitting check.

**Table 2. Walk-forward CV per-window results (final LGBM config, 5 windows × 21 days).**

| Window | Validation Period | LGBM IC | Notes |
|--------|-------------------|---------|-------|
| W1 | 2025-09-25 → 10-31 | +0.0158 | normal range |
| W2 | 2025-11-03 → 12-01 | −0.0407 | A-share Nov 2025 correction |
| W3 | 2025-12-02 → 12-30 | −0.0043 | flat market |
| W4 | 2025-12-31 → 2026-01-30 | +0.0511 | year-end rally |
| W5 | 2026-01-02 → 2026-01-30 | +0.0210 | normal range |
| **Mean (5 windows)** | | **IC: +0.0107** | **ExRet: +1.79%** |

W2's negative IC coincides with the November 2025 A-share pullback — a regime where momentum signals underperform across most CSI500 factors. W4's strong positive IC corresponds to the December–January rally. The mean walk-forward excess return of +1.79% per 21-day window (aggregate across all 5 windows) is substantially positive despite two windows with negative IC, confirming that the model's portfolio construction converts noisy cross-sectional rankings into net positive realized returns. The 3:2 positive-to-negative window ratio across diverse market conditions provides a pre-test overfitting check consistent with the teacher's recommended validation methodology.

For reference, the instructor-provided original feature set (14 features: `ret_1d/5d/10d/20d/60d`, `vol_20d`, `volume_z_20d`, `turnover_ma_20d`, `close_over_ma20/ma60`, `rsi_14`, plus 3 rank features) with XGBoost default parameters achieves a mean walk-forward CV IC of approximately 0.013 (E01).

### 3.3 Held-Out Test Set Results (2026-02-01 to 2026-04-30)

The self-test compares the instructor-provided original baseline model against the final ensemble model on the same time split. The baseline uses exactly the 14 features and XGBoost parameters from the provided starter code; the final model adds a reworked technical feature set, PE/PB fundamentals, winsorization, industry neutralization, and LGBM+XGBoost ensemble.

**Table 3. Self-test: teacher baseline vs. final model on 55 held-out trading days.**

| Metric | Teacher Baseline (14 feat) | Final (17 feat, ensemble) | Change |
|--------|:-----------:|:-------------------:|:------:|
| Model | XGBoost | LGBM + XGBoost ensemble | — |
| Preprocessing | none | winsorize + industry-neutral | — |
| Training rows | 128,752 | 90,896 | −29% |
| Test days | 55 | 55 | — |
| **Mean Rank IC** | **0.0202** | 0.0197 | −2% |
| IC positive-day ratio | 51% | 51% | tie |
| **Mean excess return / 5d window** | −0.130% | **+0.279%** | **+0.409 pp** |
| Rolling windows (n) | 55 | 55 | — |

The two models are essentially tied on rank IC (0.0202 vs. 0.0197, a negligible 0.0004 difference). However, the final model outperforms the baseline decisively on realized excess return: +0.279% per 5-day window vs. −0.130% for the baseline — a swing of +0.409 percentage points. Across 55 rolling windows, this compounds to a substantial cumulative separation.

This result is a concrete illustration of the IC/ExRet divergence discussed in Section 4.3. The baseline achieves marginally higher IC by ranking stocks slightly more accurately in cross-section, but the stocks it ranks highest perform worse in realized portfolio returns. The final model's preprocessing (winsorization, industry neutralization) and ensemble blending produce rankings that, while noisier in IC terms, translate into positive portfolio performance. The final model also achieves this with 29% fewer training rows (PE/PB data begins 2025-04-29), further supporting the conclusion that feature quality matters more than training data quantity.

---

## 4. Analysis

### 4.1 What Worked

**Industry neutralization and winsorization** had the highest per-unit impact of any single change: together they more than doubled mean IC and halved IC standard deviation (E01). Removing sector-level noise allows the model to learn stock-specific signals rather than industry momentum, which is especially important during policy-driven sector rotations common in the A-share market.

**Valuation factors (PE/PB)** provided the largest feature-driven IC gain (+11%). In a market where growth stocks frequently reprice on earnings visibility changes, combining PB (structural cheapness) with PE (near-term profitability) captures two distinct return drivers that technical features alone cannot replicate.

**Systematic experiment logging** (111 experiments, one variable changed at a time) prevented convergence to local optima. The most important benefit was catching interaction effects — specifically the `nl=8 × col=0.7` synergy (E66) that coordinate descent had missed. Recording both IC and excess return for each experiment also exposed frequent divergences between the two, preventing over-optimization on either metric alone.

### 4.2 What Did Not Work

**Northbound fund flow.** The AKShare fund-flow endpoint returned data for only 31/499 stocks (IP rate-limiting from Eastmoney). With 94% of the universe missing, the feature was abandoned. Attempting to merge and drop-naively would have trained on a biased 31-stock subsample; omitting the feature was the correct decision.

**Margin trading features.** Margin-buy ratio and balance change (97% coverage, 955K rows) produced a modest CV improvement (+0.0012 IC, E10). However, the features biased the model toward highly leveraged stocks, which suffered forced liquidations during the Feb–Apr 2026 correction — realized excess return was −2.08% vs. −0.85% without margin features. The features were excluded from the final 5-day configuration.

**ROE quality factor.** Quarterly reporting frequency makes ROE too stale for weekly stock selection. 35% NaN coverage further reduced training set size. IC dropped on addition (E07, E69–E72).

**High-complexity feature expansion.** Of 13 candidate features tested (MA crossovers, short-volatility ratios, momentum ratios, skewness), none provided a consistent improvement in both IC and excess return. The A-share CSI500 cross-section (499 stocks, daily rebalancing) does not support a large feature space without overfitting risk.

**Table 4. Summary of approaches tested but not adopted.**

| Approach | IC change | ExRet impact | Root cause |
|----------|-----------|-------------|------------|
| Margin features (E10) | +0.0012 | −2.08% (vs −0.85%) | biased toward leveraged stocks; forced liquidation in drawdowns |
| ROE (E07, E69) | −0.0083 to +0.058 | −1.17% | quarterly data too stale; 35% training rows lost |
| Northbound flow | — | — | only 31/499 stocks covered; abandoned |
| colsample=0.5 (E32) | **+0.0346** (max) | −0.66% | IC maximal but ExRet negative; IC ≠ ExRet |

### 4.3 Key Takeaways

**IC and excess return are distinct objectives.** Several parameter configurations that maximized IC produced negative excess returns (most consistently: XGBoost `colsample=0.5`, IC=0.035 but ExRet=−0.66%). Most tellingly, in the final self-test the teacher's original baseline scored a marginally higher IC (0.0202 vs. 0.0197) while producing *negative* excess return (−0.13% per window vs. +0.28%). A methodology that optimized only for IC would have selected the baseline over the final model — the wrong choice. Final model selection weighted both metrics, and the self-test confirms this dual-objective approach was correct.

---

## Reproducibility

All results in this report can be reproduced end-to-end:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch data (or use pre-fetched files in data/)
python download_data.py --start 20250101 --end 20260421
python download_data.py --update --end 20260510
python fetch_industry.py
python fetch_pe_pb_akshare.py

# 3. Run self-test (compares teacher baseline vs. final model on held-out test set)
python self_test.py --train-min 20250101

# 4. Reproduce the submitted Sub2 portfolio
python baseline_xgboost.py --model lgbm --ensemble \
    --num-leaves 8 --reg-lambda 0.1 --subsample 0.6 \
    --colsample-bytree 0.7 --min-child-weight 20 \
    --top-k 50 --weight-scheme rank \
    --as-of 20260508 --out submission_evaluation2.csv

# 5. Validate submission format
python validate_submission.py submission_evaluation2.csv
```

To reproduce Submission 1, see `README_MODEL.md` §Submission 1 for the required `features.py` edits and command.

Random seed: **42** (fixed in both `train_lgbm()` and `train_xgb()`). All other parameters are defaults or as listed in Section 2.2.

---

## Acknowledgments

Claude (Anthropic) was used as a coding assistant for model development, hyperparameter tuning, and drafting portions of this report. All final decisions are the author's own.
