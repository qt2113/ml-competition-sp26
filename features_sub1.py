"""
Feature engineering for the CSI500 stock-selection baseline.
=== SUBMISSION 1 VERSION ===

Differences from features.py (Submission 2):
  - TARGET: 3-day forward return (Sub2 uses 5-day)
  - MOMENTUM: ret_5d / ret_10d only — no ret_20d / ret_60d
  - Result: 15 features vs Sub2's 17

To reproduce Submission 1:
  1. Copy this file over features.py:
       copy features_sub1.py features.py
  2. Run:
       python baseline_xgboost.py --model lgbm \
           --num-leaves 8 --reg-lambda 0.1 --subsample 0.6 \
           --colsample-bytree 0.7 --min-child-weight 20 \
           --top-k 50 --weight-scheme rank \
           --as-of 20260430 --out submission_evaluation1.csv
  3. Restore features.py:
       git checkout features.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── 技术特征 — Sub1: 短动量（无 ret_20d / ret_60d）─────────────────────────
#   Sub1 使用 3 天 forward target，中期动量信号太慢，只保留短动量。
_TECH_FEATURES_RAW = [
    "ret_5d", "ret_10d",
    "volume_z_20d", "vol_ratio_5_20",
    "close_over_ma60",
    "rsi_14", "macd_hist", "bb_width",
    "dist_52w_high", "dist_52w_low",
]

# ── 基本面特征（有 fundamentals.parquet 时才加入）────────────────────────────
_FUND_FEATURES_RAW = [
    "pb",
    "pe_ttm",
]

# ── 主力资金净流入特征 ──────────────────────────────────────────────────────
_FLOW_FEATURES_RAW = [
    "main_pct",
    "main_pct_ma5",
]

# ── 融资融券特征 ────────────────────────────────────────────────────────────
_MARGIN_FEATURES_RAW = [
    "margin_buy_ratio",
    "margin_bal_chg",
]

# ── ROE 质量因子 ─────────────────────────────────────────────────────────────
_ROE_FEATURES_RAW = [
    "roe",
]

# ── 运行时动态确定（见 build_features()）─────────────────────────────────────
FEATURE_COLUMNS_RAW: list[str] = []
FEATURE_COLUMNS:     list[str] = []

# ── 行业中性化目标 ──────────────────────────────────────────────────────────
NEUTRALIZE_COLS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_120d",
    "ret_5d_over_20d",
    "close_over_ma20", "close_over_ma60",
    "ma5_over_ma20", "ma20_over_ma60",
    "price_pos_20", "price_pos_60",
    "main_pct", "main_pct_ma5",
    "margin_buy_ratio", "margin_bal_chg",
]

# ═══════════════════════════════════════════════════════════════════════════════
# Sub1: 3-day target
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_COLUMN    = "target_3d"
FORWARD_HORIZON  = 3


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """单只股票的时间序列特征。"""
    df = df.sort_values("date").copy()
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # ── 动量 ────────────────────────────────────────────────────────────────
    df["ret_1d"]   = close.pct_change(1)
    df["ret_5d"]   = close.pct_change(5)
    df["ret_10d"]  = close.pct_change(10)
    df["ret_20d"]  = close.pct_change(20)
    df["ret_60d"]  = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)

    # ── 波动率 & 成交量 ──────────────────────────────────────────────────────
    df["vol_5d"]  = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_60d"] = df["ret_1d"].rolling(60).std()
    df["vol_ratio_5_60"] = df["vol_5d"] / df["vol_60d"].replace(0, np.nan)

    df["ret_5d_over_20d"] = df["ret_5d"] / df["ret_20d"].replace(0, np.nan)
    df["up_days_20"] = (df["ret_1d"] > 0).rolling(20).mean()
    df["skew_20"] = df["ret_1d"].rolling(20).skew()

    vol      = df["volume"].astype(float)
    vol_ma20 = vol.rolling(20).mean()
    vol_std  = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"]   = (vol - vol_ma20) / vol_std
    df["vol_ratio_5_20"] = vol.rolling(5).mean() / vol_ma20.replace(0, np.nan)

    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    # ── 均线偏差 ─────────────────────────────────────────────────────────────
    ma5  = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    df["close_over_ma20"] = close / ma20 - 1.0
    df["close_over_ma60"] = close / ma60 - 1.0
    df["ma5_over_ma20"]  = ma5  / ma20.replace(0, np.nan) - 1.0
    df["ma20_over_ma60"] = ma20 / ma60.replace(0, np.nan) - 1.0

    high_20  = high.rolling(20).max()
    low_20   = low.rolling(20).min()
    rng_20   = (high_20 - low_20).replace(0, np.nan)
    df["price_pos_20"] = (close - low_20) / rng_20

    high_60  = high.rolling(60).max()
    low_60   = low.rolling(60).min()
    rng_60   = (high_60 - low_60).replace(0, np.nan)
    df["price_pos_60"] = (close - low_60) / rng_60

    df["high_low_range_20"] = ((high - low) / close.replace(0, np.nan)).rolling(20).mean()

    # ── RSI(14) ──────────────────────────────────────────────────────────────
    delta = close.diff()
    up    = delta.clip(lower=0).rolling(14).mean()
    down  = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + up / down)

    up5   = delta.clip(lower=0).rolling(5).mean()
    down5 = (-delta.clip(upper=0)).rolling(5).mean().replace(0, np.nan)
    df["rsi_5"] = 100 - 100 / (1 + up5 / down5)

    high_14 = high.rolling(14).max()
    low_14  = low.rolling(14).min()
    rng_14  = (high_14 - low_14).replace(0, np.nan)
    df["williams_r_14"] = -100 * (high_14 - close) / rng_14

    # ── MACD (12/26/9) ───────────────────────────────────────────────────────
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    signal     = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - signal

    # ── 布林带 (20日, ±2σ) ──────────────────────────────────────────────────
    bb_mid   = close.rolling(20).mean()
    bb_std   = close.rolling(20).std().replace(0, np.nan)
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    df["bb_position"] = (close - bb_lower) / bb_range
    df["bb_width"]    = bb_range / bb_mid

    # ── 52周高低距离 ─────────────────────────────────────────────────────────
    high_52w = high.rolling(120).max().replace(0, np.nan)
    low_52w  = low.rolling(120).min().replace(0, np.nan)
    df["dist_52w_high"] = close / high_52w - 1.0
    df["dist_52w_low"]  = close / low_52w  - 1.0

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _winsorize_cross_sectional(
    panel: pd.DataFrame,
    cols: list[str],
    n_sigma: float = 3.0,
) -> pd.DataFrame:
    """每天在截面内将极值截断到 ±n_sigma 倍标准差。"""
    for col in cols:
        grp  = panel.groupby("date")[col]
        mean = grp.transform("mean")
        std  = grp.transform("std").replace(0, np.nan)
        panel[col] = panel[col].clip(lower=mean - n_sigma * std,
                                     upper=mean + n_sigma * std)
    return panel


def _neutralize_industry(
    panel: pd.DataFrame,
    cols: list[str],
    industry_map: pd.DataFrame,
) -> pd.DataFrame:
    """行业中性化：每个特征减去当日所在行业的均值。"""
    panel = panel.merge(
        industry_map[["stock_code", "industry"]],
        on="stock_code",
        how="left",
    )
    for col in cols:
        indu_mean = panel.groupby(["date", "industry"])[col].transform("mean")
        mkt_mean  = panel.groupby("date")[col].transform("mean")
        panel[col] = panel[col] - indu_mean.fillna(mkt_mean)
    panel = panel.drop(columns=["industry"])
    return panel


def _cross_sectional_ranks(panel: pd.DataFrame, rank_cols: list[str]) -> pd.DataFrame:
    """截面排名（值域 [0,1]）。rank_cols 决定哪些列参与排名。"""
    for base in rank_cols:
        panel[f"{base}_rank"] = (
            panel.groupby("date")[base].rank(method="average", pct=True)
        )
    return panel


def build_features(
    prices: pd.DataFrame,
    industry_map: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    fund_flow: pd.DataFrame | None = None,
    margin: pd.DataFrame | None = None,
    roe: pd.DataFrame | None = None,
    extra_features: list[str] | None = None,
    winsorize_sigma: float = 3.0,
) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target (Sub1: 3-day)."""
    global FEATURE_COLUMNS_RAW, FEATURE_COLUMNS

    required = {"date", "stock_code", "close", "high", "low", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = pd.concat(
        [_per_stock_features(g) for _, g in prices.groupby("stock_code", sort=False)],
        ignore_index=True,
    )

    # ── 合并基本面数据（可选）────────────────────────────────────────────────
    if fundamentals is not None:
        fund = fundamentals[["date", "stock_code"] + _FUND_FEATURES_RAW].copy()
        fund["date"] = pd.to_datetime(fund["date"])
        panel = panel.merge(fund, on=["date", "stock_code"], how="left")
        panel = panel.sort_values(["stock_code", "date"])
        for col in _FUND_FEATURES_RAW:
            panel[col] = panel.groupby("stock_code")[col].transform(lambda x: x.ffill())
        panel = panel.sort_values(["date", "stock_code"])
        raw_cols  = _TECH_FEATURES_RAW + _FUND_FEATURES_RAW
        rank_cols = ["ret_20d", "vol_20d", "bb_position"]
    else:
        raw_cols  = _TECH_FEATURES_RAW
        rank_cols = ["ret_20d", "vol_20d", "bb_position"]

    # ── 合并主力资金流向数据（可选）──────────────────────────────────────────
    if fund_flow is not None:
        ff = fund_flow[["date", "stock_code", "main_pct"]].copy()
        ff["date"] = pd.to_datetime(ff["date"])
        panel = panel.merge(ff, on=["date", "stock_code"], how="left")
        panel = panel.sort_values(["stock_code", "date"])
        panel["main_pct_ma5"] = (
            panel.groupby("stock_code")["main_pct"]
            .transform(lambda x: x.rolling(5, min_periods=1).mean())
        )
        panel = panel.sort_values(["date", "stock_code"])
        raw_cols  = raw_cols  + _FLOW_FEATURES_RAW
        rank_cols = rank_cols + _FLOW_FEATURES_RAW

    # ── 合并融资融券数据（可选）──────────────────────────────────────────────
    if margin is not None:
        mg = margin[["date", "stock_code", "margin_bal", "margin_buy"]].copy()
        mg["date"] = pd.to_datetime(mg["date"])
        panel = panel.merge(mg, on=["date", "stock_code"], how="left")
        panel = panel.sort_values(["stock_code", "date"])
        mg_buy_ma5 = (
            panel.groupby("stock_code")["margin_buy"]
            .transform(lambda x: x.rolling(5, min_periods=2).mean())
            .replace(0, np.nan)
        )
        panel["margin_buy_ratio"] = panel["margin_buy"] / mg_buy_ma5
        mg_bal_lag = (
            panel.groupby("stock_code")["margin_bal"]
            .transform(lambda x: x.shift(1))
            .replace(0, np.nan)
        )
        panel["margin_bal_chg"] = (
            panel["margin_bal"] / mg_bal_lag - 1.0
        )
        panel = panel.sort_values(["date", "stock_code"])

    # ── 合并 ROE 质量因子（可选）──────────────────────────────────────────────
    if roe is not None:
        roe_df = roe[["date", "stock_code", "roe"]].copy()
        roe_df["date"] = pd.to_datetime(roe_df["date"])
        panel = panel.merge(roe_df, on=["date", "stock_code"], how="left")
        panel = panel.sort_values(["stock_code", "date"])
        panel["roe"] = panel.groupby("stock_code")["roe"].transform(lambda x: x.ffill())
        panel = panel.sort_values(["date", "stock_code"])
        raw_cols  = raw_cols  + _ROE_FEATURES_RAW
        rank_cols = rank_cols + _ROE_FEATURES_RAW
        print(f"   ROE 已合并：{panel['roe'].notna().mean():.1%} 非空覆盖率")

    if extra_features:
        valid_extra = [f for f in extra_features if f not in raw_cols and f in panel.columns]
        if valid_extra:
            raw_cols = raw_cols + valid_extra

    neutralize_active = [c for c in NEUTRALIZE_COLS if c in raw_cols]

    FEATURE_COLUMNS_RAW.clear()
    FEATURE_COLUMNS_RAW.extend(raw_cols)
    FEATURE_COLUMNS.clear()
    FEATURE_COLUMNS.extend(raw_cols + [f"{b}_rank" for b in rank_cols])

    if winsorize_sigma is not None:
        panel = _winsorize_cross_sectional(panel, raw_cols, n_sigma=winsorize_sigma)

    if industry_map is not None:
        panel = _neutralize_industry(panel, neutralize_active, industry_map)

    panel = _cross_sectional_ranks(panel, rank_cols)
    return panel


def load_margin(path: str | Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = Path(__file__).parent / "data" / "margin.parquet"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_fund_flow(path: str | Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = Path(__file__).parent / "data" / "northbound.parquet"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_roe(path: str | Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = Path(__file__).parent / "data" / "roe.parquet"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_fundamentals(path: str | Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = Path(__file__).parent / "data" / "fundamentals.parquet"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def load_industry_map(path: str | Path | None = None) -> pd.DataFrame | None:
    if path is None:
        path = Path(__file__).parent / "data" / "industry_map.csv"
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"stock_code": str})


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].copy()
    for col in FEATURE_COLUMNS:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    df = df.dropna(subset=FEATURE_COLUMNS)
    return df
