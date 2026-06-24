# CSI500 Stock Selection — Model Documentation

## Model

LightGBM + XGBoost ensemble with a 50/50 rank-percentile blend.

- **LightGBM**: `num_leaves=8, max_depth=5, learning_rate=0.05, subsample=0.6, colsample_bytree=0.7, min_child_samples=20, reg_lambda=0.1`
- **XGBoost**: `max_depth=6, learning_rate=0.05, subsample=0.6, colsample_bytree=0.6, min_child_weight=20, reg_lambda=2.0`

Target: 5-day forward return. Portfolio: top-50 stocks by predicted score, linear rank-weighted, 10% per-stock cap.

### Features (17)

Technical: `ret_5d, ret_10d, ret_20d, ret_60d, volume_z_20d, vol_ratio_5_20, close_over_ma60, rsi_14, macd_hist, bb_width, dist_52w_high, dist_52w_low`  
Fundamental: `pb, pe_ttm`  
Cross-sectional ranks: `ret_20d_rank, vol_20d_rank, bb_position_rank`

### Preprocessing

1. Winsorization at ±3σ (cross-sectional, per date).
2. Industry neutralization (31 Shenwan sectors): subtract daily sector mean from momentum, MA-deviation, and fund-flow features.
3. Missing values: dropped during training, cross-sectional median fill at prediction time.

### Validation

- Walk-forward CV (5 windows × 21 days, pre-2026-02-01 for tuning).
- 26-window rolling evaluation (Mar–Apr 2026).
- 11-window robustness check (Feb–May 2026) with shifted window boundaries.

---

## File Overview

### Pipeline

| File | Purpose |
|------|---------|
| `baseline_xgboost.py` | Main pipeline: data loading, feature building, model training, prediction, portfolio construction. Supports both LightGBM and XGBoost with optional ensemble mode. |
| `features.py` | Feature engineering module. Computes all technical, fundamental, and cross-sectional features; applies winsorization and industry neutralization. |
| `download_data.py` | Fetches CSI500 constituents, OHLCV prices, and index data from AKShare. Supports `--update` for incremental refresh. |
| `validate_submission.py` | Validates submission CSV against competition constraints (stock codes, weights, min names, caps). |
| `score_submission.py` | Scores a submission CSV against realized returns over a specified window. |

### Data Fetching

| File | Output |
|------|--------|
| `fetch_industry.py` | `data/industry_map.csv` — Shenwan industry classifications |
| `fetch_pe_pb_akshare.py` | `data/fundamentals.parquet` — daily PE(TTM) and PB |
| `fetch_margin.py` | `data/margin.parquet` — margin trading history |
| `fetch_roe_akshare.py` | `data/roe.parquet` — quarterly ROE (not used in final model) |
| `fetch_northbound.py` | `data/northbound.parquet` — fund flow (insufficient coverage, not used) |

### Validation Scripts

| File | Purpose |
|------|---------|
| `walk_forward_cv.py` | Walk-forward CV for hyperparameter/feature tuning. Logs to `cv_log.csv`. |
| `eval.py` | Dual-metric evaluator: April backtest + walk-forward CV overfit check. Logs to `eval_log.csv`. |
| `robust_check.py` | Multi-window robustness evaluation. Logs to `robust_log.csv`. |
| `april_rolling.py` | 26-window rolling 5-day evaluation. |

### Experiment Logs

| File | Contents |
|------|----------|
| `experiment_log.csv` | 111 experiments documenting the full development history. |
| `cv_log.csv` | 204 walk-forward CV runs with per-window IC and parameters. |
| `eval_log.csv` | Final model comparisons (April backtest + walk-forward CV). |
| `robust_log.csv` | Robustness evaluations across 11 weekly windows. |

---

## Reproducing Submissions

### Submission 2 (May 11–15 window) — current codebase

```bash
pip install -r requirements.txt

# Download data
python download_data.py --start 20250101 --end 20260421
python download_data.py --update --end 20260510
python fetch_industry.py
python fetch_pe_pb_akshare.py
python fetch_margin.py

# Generate portfolio
python baseline_xgboost.py --ensemble --as-of 20260508 --out submission_evaluation2.csv

# Validate
python validate_submission.py submission_evaluation2.csv
```

The pipeline skips any optional data file that is missing, so `fetch_roe_akshare.py` and `fetch_northbound.py` are not required.

### Submission 1 (May 6–8 window)

Submission 1 differs from Submission 2 in three ways:

| | Submission 1 | Submission 2 |
|---|---|---|
| **Target horizon** | 3-day (`FORWARD_HORIZON = 3`) | 5-day (`FORWARD_HORIZON = 5`) |
| **Target column** | `target_3d` | `target_5d` |
| **Momentum features** | Short only: `ret_5d`, `ret_10d` | Full: `ret_5d`, `ret_10d`, `ret_20d`, `ret_60d` |
| **Total features** | 15 (10 tech + 2 PE/PB + 3 rank) | 17 (12 tech + 2 PE/PB + 3 rank) |
| **Model** | LightGBM only (no ensemble) | LightGBM + XGBoost ensemble (50/50 rank blend) |
| **Submission as-of** | 2026-04-30 | 2026-05-08 |

**Why 3-day target with short momentum?** For a 3-day holding period, short-term price drift dominates — `ret_20d` and `ret_60d` capture medium-term trends that are too slow for a 3-day window and may even mean-revert intra-week. Submission 1 therefore drops these two raw features (but keeps the cross-sectional rank `ret_20d_rank`, which provides relative-positioning signal without raw-momentum noise).

**Why LightGBM-only?** The ensemble (LightGBM + XGBoost) was introduced on April 30 (see `eval_log.csv` experiments B0–F10) after Submission 1 had already been finalized and uploaded. The Sub1 model was pure LightGBM with `num_leaves=8`.

#### Quick reproduction (drop-in replacement)

```bash
# 1. Swap in the Sub1 feature module
copy features_sub1.py features.py

# 2. Generate portfolio (note: NO --ensemble)
python baseline_xgboost.py --model lgbm ^
    --num-leaves 8 --reg-lambda 0.1 --subsample 0.6 ^
    --colsample-bytree 0.7 --min-child-weight 20 ^
    --top-k 50 --weight-scheme rank ^
    --as-of 20260430 --out submission_evaluation1.csv

# 3. Validate
python validate_submission.py submission_evaluation1.csv

# 4. Restore the Sub2 feature module
git checkout features.py
```

The reproduced portfolio matches `submission_evaluation1.csv` exactly (50/50 stocks, weight rank correlation = 1.000).

#### Manual reproduction (edit features.py directly)

If you prefer not to use `features_sub1.py`, make these two edits in `features.py`:

**Edit 1** — target horizon (lines 62–63):
```python
# Change:
TARGET_COLUMN    = "target_5d"
FORWARD_HORIZON  = 5
# To:
TARGET_COLUMN    = "target_3d"
FORWARD_HORIZON  = 3
```

**Edit 2** — feature set (lines 14–20), drop `ret_20d` and `ret_60d`:
```python
# Change:
_TECH_FEATURES_RAW = [
    "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    ...
]
# To:
_TECH_FEATURES_RAW = [
    "ret_5d", "ret_10d",
    ...
]
```

Then run the same `baseline_xgboost.py` command as above (without `--ensemble`). Revert both edits afterward.

### Notes

- Default hyperparameters in `baseline_xgboost.py` match the Submission 2 config. Running with `--ensemble` and `--as-of 20260508` reproduces the submitted portfolio.
- To reproduce Submission 1, apply the two edits to `features.py` documented above, then run the LGBM-only command. Revert the edits afterward to restore Sub2 compatibility.
- Random seed is fixed (`random_state=42`) in both `train_lgbm()` and `train_xgb()`.
- `notes/说明.md` contains a detailed progress log in Chinese.
