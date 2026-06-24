"""
Self-test: train on pre-TEST_START, evaluate on held-out test period.
Compares the instructor-provided 14-feature XGBoost baseline against the
final LightGBM+XGBoost ensemble model.

Usage:
  python self_test.py --train-min 20250101
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

# ── Teacher's original baseline modules ────────────────────────────────────
import features_baseline
# ── Final model module ─────────────────────────────────────────────────────
from features import (
    TARGET_COLUMN as TARGET_COLUMN_FINAL,
    FORWARD_HORIZON,
    build_features, load_industry_map, load_fundamentals, load_margin,
)

DATA_DIR   = Path(__file__).parent / "data"
TEST_START = pd.Timestamp("2026-02-01")
TEST_END   = pd.Timestamp("2026-04-30")
MAX_WEIGHT = 0.10


# ── helpers ──────────────────────────────────────────────────────────────────

def _rank_ic(y_true, y_pred, dates):
    ics = []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[m], y_pred[m])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def _get_test_pool(panel, feats):
    """Filter panel to test period, fill missing with cross-sectional median."""
    df = panel[(panel["date"] >= TEST_START) & (panel["date"] <= TEST_END)].copy()
    fcols = [c for c in feats if c in df.columns]
    for col in fcols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df.groupby("date")[col].transform("median"))
    return df.dropna(subset=fcols + [TARGET_COLUMN_FINAL])


def _build_portfolio(scores, top_k=50):
    chosen = scores.nlargest(top_k)
    n = len(chosen)
    ranks = np.arange(n, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def _simulate_excess_returns(panel, feats, model, xgb_model, xgb_feats,
                              prices_raw, index_df, top_k=50):
    all_dates = sorted(prices_raw["date"].unique())
    test_dates = [d for d in all_dates
                  if TEST_START <= pd.Timestamp(d) <= TEST_END]
    ex_rets = []
    for as_of in test_dates:
        try:
            idx = all_dates.index(as_of)
        except ValueError:
            continue
        if idx + 5 >= len(all_dates):
            continue
        start, end = all_dates[idx + 1], all_dates[idx + 5]
        pred_rows = panel[panel["date"] == pd.Timestamp(as_of)]
        fcols = [c for c in feats if c in pred_rows.columns]
        pred_rows = pred_rows.dropna(subset=fcols)
        if len(pred_rows) < 50:
            continue

        if xgb_model is not None:
            preds = model.predict(pred_rows[fcols].values)
            preds_x = xgb_model.predict(pred_rows[xgb_feats].values)
            r_main = pd.Series(preds).rank(pct=True)
            r_xgb  = pd.Series(preds_x).rank(pct=True)
            scores_arr = (0.5 * r_main + 0.5 * r_xgb).values
        else:
            scores_arr = model.predict(pred_rows[fcols].values)

        scores = pd.Series(scores_arr, index=pred_rows["stock_code"].values)
        weights = _build_portfolio(scores, top_k)
        p = prices_raw[prices_raw["stock_code"].isin(weights.index)]
        p = p[p["date"].isin([start, end])]
        pv = p.pivot(index="date", columns="stock_code", values="close")
        if start not in pv.index or end not in pv.index:
            continue
        ret = pv.loc[end] / pv.loc[start] - 1
        ret = ret.reindex(weights.index).dropna()
        if len(ret) < 20:
            continue
        w = weights.reindex(ret.index)
        w = w / w.sum()
        port_ret = float((w * ret).sum())
        idx_row = index_df[index_df["date"].isin([start, end])].set_index("date")["close"]
        if start not in idx_row.index or end not in idx_row.index:
            continue
        bench_ret = float(idx_row.loc[end] / idx_row.loc[start] - 1)
        ex_rets.append(port_ret - bench_ret)
    if not ex_rets:
        return float("nan"), 0
    return float(np.mean(ex_rets)) * 100, len(ex_rets)


# ── baseline evaluation (teacher's 14-feature XGBoost) ───────────────────────

def eval_baseline(prices, prices_raw, index_df, train_min_date=None):
    """Teacher's original: 14 tech features, XGBoost default params, no preprocessing."""
    print("\n>> Building features [BASELINE: teacher's original 14-feature set]")
    panel = features_baseline.build_features(prices)
    feats = list(features_baseline.FEATURE_COLUMNS)
    print(f"   {len(feats)} features: {feats}")

    trading_dates = np.sort(panel["date"].unique())
    test_start_idx = int(np.searchsorted(trading_dates, np.datetime64(TEST_START)))
    train_cutoff_idx = max(0, test_start_idx - features_baseline.FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[train_cutoff_idx])

    print(f"\n{'='*60}")
    print(f"  BASELINE: Teacher's Original XGBoost (14 features)")
    print(f"  train: {train_min_date or 'all'} ~ {train_cutoff.date()}"
          f"  |  test: {TEST_START.date()} ~ {TEST_END.date()}")
    print(f"{'='*60}")

    train_pool = features_baseline.training_frame(panel, max_date=train_cutoff, min_date=train_min_date)
    if len(train_pool) < 5000:
        print(f"  ERROR: only {len(train_pool)} training rows")
        return {}
    train_dates = np.sort(train_pool["date"].unique())
    inner_cut = pd.Timestamp(train_dates[-10])
    inner_train = train_pool[train_pool["date"] <  inner_cut]
    inner_val   = train_pool[train_pool["date"] >= inner_cut]
    print(f"  train rows: {len(train_pool):,} ({train_pool['date'].nunique()} dates)"
          f"  |  inner val: {len(inner_val):,} rows")

    model = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30,
    )
    model.fit(
        inner_train[feats], inner_train[features_baseline.TARGET_COLUMN],
        eval_set=[(inner_val[feats], inner_val[features_baseline.TARGET_COLUMN])],
        verbose=False,
    )

    # Test IC
    test_pool = _get_test_pool(panel, feats)
    test_dates = np.sort(test_pool["date"].unique())
    preds = model.predict(test_pool[feats].values)
    ic = _rank_ic(test_pool[features_baseline.TARGET_COLUMN].to_numpy(), preds, test_pool["date"].to_numpy())

    daily_ics = {}
    for d in test_dates:
        m = test_pool["date"] == d
        if m.sum() < 20:
            continue
        rho, _ = spearmanr(test_pool.loc[m, features_baseline.TARGET_COLUMN], preds[m.values])
        if not np.isnan(rho):
            daily_ics[str(pd.Timestamp(d).date())] = rho

    ic_std = float(np.std(list(daily_ics.values())))
    ic_pos = sum(1 for v in daily_ics.values() if v > 0)
    n_days = len(daily_ics)
    print(f"  test IC:  {ic:.4f}  (std={ic_std:.4f},  positive={ic_pos}/{n_days})")

    exret_mean, exret_n = _simulate_excess_returns(
        panel, feats, model, None, None, prices_raw, index_df)
    print(f"  test ExRet: {exret_mean:+.3f}%  ({exret_n} rolling 5d windows)")

    return {
        "label": "BASELINE (teacher original, 14 feat)",
        "n_features": len(feats),
        "n_train_rows": len(train_pool),
        "n_train_dates": train_pool["date"].nunique(),
        "n_test_days": n_days,
        "mean_ic": ic,
        "std_ic": ic_std,
        "ic_pos_ratio": ic_pos / n_days if n_days else 0,
        "exret_mean": exret_mean,
        "exret_n": exret_n,
    }


# ── final model evaluation ───────────────────────────────────────────────────

def eval_final(prices, prices_raw, index_df, train_min_date=None):
    """Final model: LGBM+XGBoost ensemble, 17 features, full preprocessing."""
    print("\n>> Building features [FINAL: +PE/PB, +industry, +winsorize]")
    ind_map = load_industry_map()
    fund = load_fundamentals()
    panel = build_features(prices, industry_map=ind_map, fundamentals=fund,
                           margin=None, winsorize_sigma=3.0)
    from features import FEATURE_COLUMNS as fc_fin
    feats = list(fc_fin)
    print(f"   {len(feats)} features: {feats}")

    trading_dates = np.sort(panel["date"].unique())
    test_start_idx = int(np.searchsorted(trading_dates, np.datetime64(TEST_START)))
    train_cutoff_idx = max(0, test_start_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[train_cutoff_idx])

    print(f"\n{'='*60}")
    print(f"  FINAL: LGBM+XGBoost Ensemble (17 features, full preprocessing)")
    print(f"  train: {train_min_date or 'all'} ~ {train_cutoff.date()}"
          f"  |  test: {TEST_START.date()} ~ {TEST_END.date()}")
    print(f"{'='*60}")

    from features import training_frame
    train_pool = training_frame(panel, max_date=train_cutoff, min_date=train_min_date)
    if len(train_pool) < 5000:
        print(f"  ERROR: only {len(train_pool)} training rows")
        return {}
    train_dates = np.sort(train_pool["date"].unique())
    inner_cut = pd.Timestamp(train_dates[-10])
    inner_train = train_pool[train_pool["date"] <  inner_cut]
    inner_val   = train_pool[train_pool["date"] >= inner_cut]
    print(f"  train rows: {len(train_pool):,} ({train_pool['date'].nunique()} dates)"
          f"  |  inner val: {len(inner_val):,} rows")

    # LGBM main model
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, max_depth=5, num_leaves=8,
        min_child_samples=20, reg_lambda=0.1,
        subsample=0.6, subsample_freq=1, colsample_bytree=0.7,
        learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1,
    )
    lgb_model.fit(
        inner_train[feats], inner_train[TARGET_COLUMN_FINAL],
        eval_set=[(inner_val[feats], inner_val[TARGET_COLUMN_FINAL])],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )

    # XGBoost ensemble partner
    xgb_feats = [c for c in feats if "margin" not in c]
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, min_child_weight=20,
        reg_lambda=2.0, subsample=0.6, colsample_bytree=0.6,
        learning_rate=0.05, tree_method="hist", n_jobs=-1,
        early_stopping_rounds=30, random_state=42,
    )
    xgb_model.fit(
        inner_train[xgb_feats], inner_train[TARGET_COLUMN_FINAL],
        eval_set=[(inner_val[xgb_feats], inner_val[TARGET_COLUMN_FINAL])],
        verbose=False,
    )

    # Test IC
    test_pool = _get_test_pool(panel, feats)
    test_dates = np.sort(test_pool["date"].unique())

    preds_lgb = lgb_model.predict(test_pool[feats].values)
    preds_xgb = xgb_model.predict(test_pool[xgb_feats].values)
    r_main = pd.Series(preds_lgb).rank(pct=True)
    r_xgb  = pd.Series(preds_xgb).rank(pct=True)
    preds = (0.5 * r_main + 0.5 * r_xgb).values

    ic = _rank_ic(test_pool[TARGET_COLUMN_FINAL].to_numpy(), preds, test_pool["date"].to_numpy())

    daily_ics = {}
    for d in test_dates:
        m = test_pool["date"] == d
        if m.sum() < 20:
            continue
        rho, _ = spearmanr(test_pool.loc[m, TARGET_COLUMN_FINAL], preds[m.values])
        if not np.isnan(rho):
            daily_ics[str(pd.Timestamp(d).date())] = rho

    ic_std = float(np.std(list(daily_ics.values())))
    ic_pos = sum(1 for v in daily_ics.values() if v > 0)
    n_days = len(daily_ics)
    print(f"  test IC:  {ic:.4f}  (std={ic_std:.4f},  positive={ic_pos}/{n_days})")

    exret_mean, exret_n = _simulate_excess_returns(
        panel, feats, lgb_model, xgb_model, xgb_feats, prices_raw, index_df)
    print(f"  test ExRet: {exret_mean:+.3f}%  ({exret_n} rolling 5d windows)")

    return {
        "label": "FINAL (LGBM+XGBoost ensemble, 17 feat)",
        "n_features": len(feats),
        "n_train_rows": len(train_pool),
        "n_train_dates": train_pool["date"].nunique(),
        "n_test_days": n_days,
        "mean_ic": ic,
        "std_ic": ic_std,
        "ic_pos_ratio": ic_pos / n_days if n_days else 0,
        "exret_mean": exret_mean,
        "exret_n": exret_n,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-min", default=None, help="Earliest training date YYYYMMDD")
    args = p.parse_args()

    print(">> Loading prices + index")
    prices = pd.read_parquet(DATA_DIR / "prices.parquet")
    prices["date"] = pd.to_datetime(prices["date"])
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks")

    results = []
    train_min = args.train_min

    # 1. Teacher baseline
    results.append(eval_baseline(prices, prices, index_df, train_min_date=train_min))

    # 2. Final model
    results.append(eval_final(prices, prices, index_df, train_min_date=train_min))

    # Summary
    print(f"\n{'='*70}")
    print(f"  SELF-TEST SUMMARY")
    print(f"  split: train < {TEST_START.date()}  |  test: {TEST_START.date()} ~ {TEST_END.date()}")
    if train_min:
        print(f"  train_min_date: {train_min}")
    print(f"{'='*70}")
    hdr = (f"  {'Model':<42} {'Feat':>4} {'Train':>7} {'Days':>5} "
           f"{'IC':>8} {'IC>0':>6} {'ExRet':>8}")
    print(hdr)
    print(f"  {'-'*68}")
    for r in results:
        if not r:
            continue
        ex_str = f"{r['exret_mean']:+.2f}%" if not np.isnan(r['exret_mean']) else "N/A"
        print(f"  {r['label']:<42} {r['n_features']:>4} {r['n_train_rows']:>7,} "
              f"{r['n_test_days']:>5} {r['mean_ic']:>8.4f} {r['ic_pos_ratio']:>5.0%} {ex_str:>8}")

    if len(results) >= 2 and results[0] and results[1]:
        d_ic = results[1]["mean_ic"] - results[0]["mean_ic"]
        print(f"\n  IC delta (Final - Baseline): {d_ic:+.4f}")
        if not np.isnan(results[0].get("exret_mean", float("nan"))) and \
           not np.isnan(results[1].get("exret_mean", float("nan"))):
            d_er = results[1]["exret_mean"] - results[0]["exret_mean"]
            print(f"  ExRet delta (Final - Baseline): {d_er:+.3f}%")

    # Log to CSV
    log_file = Path(__file__).parent / "self_test_log.csv"
    is_new = not log_file.exists()
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "model", "n_features", "n_train_rows",
                        "n_train_dates", "n_test_days", "mean_ic", "std_ic",
                        "ic_pos_ratio", "exret_mean", "exret_n_windows",
                        "train_min", "test_start", "test_end"])
        for r in results:
            if not r:
                continue
            w.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                r["label"], r["n_features"], r["n_train_rows"],
                r["n_train_dates"], r["n_test_days"],
                f"{r['mean_ic']:.4f}", f"{r['std_ic']:.4f}",
                f"{r['ic_pos_ratio']:.3f}",
                f"{r['exret_mean']:+.4f}" if not np.isnan(r['exret_mean']) else "",
                r.get("exret_n", ""),
                train_min or "", str(TEST_START.date()), str(TEST_END.date()),
            ])
    print(f"\n  Logged to {log_file.name}")


if __name__ == "__main__":
    main()
