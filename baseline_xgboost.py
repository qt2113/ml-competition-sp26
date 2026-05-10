"""
XGBoost baseline for the CSI500 stock-selection competition.

Pipeline
--------
1. Load data/prices.parquet A #加载数据
2. Build features + 5-day forward target (features.py) #构建特征和目标
3. Train XGBoost on all but the last `EMBARGO_DAYS` training rows #时间切分，最后几天作为验证集，前面部分训练模型
4. Validate on those held-out rows (reports rank IC as sanity check) #验证模型，计算rank IC作为指标
5. Predict on the most recent date #预测最新日期的股票得分
6. Build a portfolio: top-K names, score-weighted with the 10% cap #构建投资组合，选top-K的股票，权重根据得分分配，但单只股票权重不超过10%

Usage
-----
  python baseline_xgboost.py                         # predict from latest data
  python baseline_xgboost.py --as-of 20260503        # predict as of a given date
  python baseline_xgboost.py --top-k 30 --out submissions/week1.csv
  python baseline_xgboost.py --weight-scheme exp     # 指数权重（放大高分股）
  python baseline_xgboost.py --margin-filter 0.1     # 过滤融资杠杆最高的10%
  python baseline_xgboost.py --ensemble              # LightGBM+XGBoost双模型融合
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import spearmanr #计算spearman相关系数，作为rank IC的指标

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, load_industry_map, load_fundamentals, load_fund_flow, load_margin,
    training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score


def train_lgbm(train_df: pd.DataFrame, val_df: pd.DataFrame,
               features: list[str], **kwargs) -> lgb.LGBMRegressor:
    params = dict(
        n_estimators=400,          # 树的数量；配合 early_stopping 实际用量会更少
        max_depth=5,               # 树深度；3太浅，>6易过拟合，建议 4~6
        num_leaves=8,              # leaf-wise生长的关键参数；扫描4~31后8是甜点（IC最高）
        learning_rate=0.05,        # 步长；调小需同步增大 n_estimators
        subsample=0.6,             # 行采样比例；0.6比0.8更防过拟合
        subsample_freq=1,          # LightGBM 必须设此项才启用行采样
        colsample_bytree=0.7,      # 列采样比例；0.7在nl=8时最优（E66验证）
        min_child_samples=20,      # 叶节点最少样本数；越大越保守，建议 10~30
        reg_lambda=0.1,            # L2正则；LGB用0.1（num_leaves已限制复杂度，不需要大λ）
        n_jobs=-1,
        random_state=42,           # 固定随机种子，保证结果可复现（课件要求）
        verbose=-1,
    )
    params.update(kwargs)          # CLI 传入的参数覆盖默认值
    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[features], train_df[TARGET_COLUMN],
        eval_set=[(val_df[features], val_df[TARGET_COLUMN])],
        callbacks=[
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),
        ],
    )
    return model


def train_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame,
              features: list[str], **kwargs) -> xgb.XGBRegressor:
    """XGBoost 副模型（用于 ensemble）；参数与 LightGBM 对齐。"""
    params = dict(
        n_estimators=400,
        max_depth=6,               # submission 1的甜点是6，之前测的是8
        min_child_weight=20,       # 对应 LightGBM 的 min_child_samples；mcw扫描后20最优
        reg_lambda=2.0,            # 5d target实测：λ=2 ExRet+1.92% > λ=5 +1.08% > λ=10 -0.07%
        subsample=0.6,
        colsample_bytree=0.6,
        learning_rate=0.05,
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,
        random_state=42,
    )
    params.update(kwargs)          # CLI 传入的参数覆盖默认值
    model = xgb.XGBRegressor(**params)
    model.fit(
        train_df[features], train_df[TARGET_COLUMN],
        eval_set=[(val_df[features], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


# 向后兼容旧调用名
def train_model(train_df, val_df):
    return train_lgbm(train_df, val_df, FEATURE_COLUMNS)


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(
    scores: pd.Series,
    top_k: int = DEFAULT_TOP_K,
    weight_scheme: str = "rank",   # "rank"=线性排名权重（原始），"exp"=指数权重（放大高分差距）
    exp_alpha: float = 3.0,        # 指数权重的集中度；越大越集中于 top 股，建议范围 1~5
) -> pd.Series:
    """Top-K names, weighted by scheme then capped at MAX_WEIGHT.

    weight_scheme='rank': w ∝ rank（原始方案，top-1最高，线性递减）
    weight_scheme='exp':  w ∝ exp(α·score_norm)（放大高分股和低分股的差距）
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    if weight_scheme == "rank":
        # 原始方案：top-1 得 top_k 分，top-2 得 top_k-1 分...线性递减
        ranks = np.arange(top_k, 0, -1, dtype=float)
        w = pd.Series(ranks / ranks.sum(), index=chosen.index)
    elif weight_scheme == "exp":
        # 指数方案：先把 score 归一化到 [0,1]，再取 exp(α·s)
        # 效果：分数差距大的股票权重差距被放大；α=3时 top 股和末位股权重比约 e^3≈20倍
        s = chosen.values.astype(float)
        s_norm = (s - s.min()) / max(s.max() - s.min(), 1e-8)
        raw = np.exp(exp_alpha * s_norm)
        w = pd.Series(raw / raw.sum(), index=chosen.index)
    else:
        raise ValueError(f"unknown weight_scheme: {weight_scheme}")

    # Iteratively cap at MAX_WEIGHT and redistribute to uncapped names.
    # 迭代处理10%上限：超出的部分均匀分给其他股票
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

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def _blend_scores(scores_a: pd.Series, scores_b: pd.Series,
                  alpha: float = 0.5) -> pd.Series:
    """把两个模型的分数转成截面排名百分位后加权融合。

    用排名而不是原始分数融合，避免两个模型的分数尺度不同导致一方主导。
    alpha=0.5 表示各占一半；alpha 越大越偏向 scores_a（LightGBM）。
    """
    stocks = scores_a.index.union(scores_b.index)
    ra = scores_a.reindex(stocks).rank(pct=True).fillna(0.5)  # 缺失的按中位数处理
    rb = scores_b.reindex(stocks).rank(pct=True).fillna(0.5)
    return alpha * ra + (1 - alpha) * rb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")

    # ── strategy layer 参数 ──────────────────────────────────────────────────
    p.add_argument("--weight-scheme", default="rank", choices=["rank", "exp"],
                   help="权重方案：rank=线性排名（默认），exp=指数权重（放大高分差距）")
    p.add_argument("--exp-alpha", type=float, default=3.0,
                   help="指数权重的集中度；仅 --weight-scheme exp 时生效，建议 1~5（默认3）")
    p.add_argument("--margin-filter", type=float, default=0.0,
                   help="过滤融资杠杆最高的前 X 比例；0=不过滤，0.1=过滤最高10%%（默认0）")
    p.add_argument("--ensemble", action="store_true",
                   help="同时训练 XGBoost（不含 margin 特征）并与 LightGBM 融合分数（各50%%）")

    # ── 模型超参数（覆盖 train_lgbm/train_xgb 默认值）────────────────────────
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb"],
                   help="主模型类型：lgbm=LightGBM（默认），xgb=XGBoost")
    p.add_argument("--reg-lambda", type=float, default=None,
                   help="L2正则化系数；LGB默认0.1，XGB默认10.0")
    p.add_argument("--num-leaves", type=int, default=None,
                   help="LightGBM叶子数；默认8（E66最优）；XGB忽略此参数")
    p.add_argument("--max-depth", type=int, default=None,
                   help="树深度；LGB默认5，XGB默认8")
    p.add_argument("--min-child-weight", type=int, default=None,
                   help="叶节点最小样本权重/数量；LGB=min_child_samples默认20，XGB默认10")
    p.add_argument("--subsample", type=float, default=None,
                   help="行采样比例；默认0.6")
    p.add_argument("--colsample-bytree", type=float, default=None,
                   help="列采样比例；LGB默认0.7，XGB默认0.6")
    p.add_argument("--train-min-date", default=None,
                   help="训练集最早日期（YYYYMMDD）；CV发现2025-06-01起点IC最高，建议设此值")
    p.add_argument("--extra-features", default="",
                   help="额外加入的特征名，逗号分隔，如 ret_120d,close_over_ma20")
    p.add_argument("--no-fundamentals", action="store_true",
                   help="跳过 PE/PB（覆盖率不足时训练行被 dropna 删光）")
    p.add_argument("--no-fund-flow", action="store_true",
                   help="跳过 northbound 主力资金（只覆盖31只股票，会删掉94%训练行）")
    p.add_argument("--no-margin", action="store_true",
                   help="跳过融资融券特征")

    args = p.parse_args()

    # 收集非 None 的超参数覆盖
    model_kwargs: dict = {}
    if args.reg_lambda is not None:
        model_kwargs["reg_lambda"] = args.reg_lambda
    if args.num_leaves is not None:
        model_kwargs["num_leaves"] = args.num_leaves
    if args.max_depth is not None:
        model_kwargs["max_depth"] = args.max_depth
    if args.min_child_weight is not None:
        if args.model == "lgbm":
            model_kwargs["min_child_samples"] = args.min_child_weight  # LGB用min_child_samples
        else:
            model_kwargs["min_child_weight"] = args.min_child_weight
    if args.subsample is not None:
        model_kwargs["subsample"] = args.subsample
    if args.colsample_bytree is not None:
        model_kwargs["colsample_bytree"] = args.colsample_bytree

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    industry_map = load_industry_map()     # 若 data/industry_map.csv 不存在则跳过行业中性化
    fundamentals = None if args.no_fundamentals else load_fundamentals()
    fund_flow    = None if args.no_fund_flow    else load_fund_flow()
    margin       = None if args.no_margin       else load_margin()
    roe          = None                    # ROE 暂时关闭（E69实验：对XGB有害，待统一测试）
    extra_features = [f.strip() for f in args.extra_features.split(",") if f.strip()] or None
    panel = build_features(prices, industry_map=industry_map,
                           fundamentals=fundamentals, fund_flow=fund_flow,
                           margin=margin, roe=roe, extra_features=extra_features)

    as_of_ts = pd.Timestamp(args.as_of) if args.as_of else panel["date"].max()
    trading_dates = np.sort(panel["date"].unique())
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff_idx = max(0, as_of_idx - FORWARD_HORIZON)
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx]) # 防泄露：训练目标(t+5)不超过 as_of
    train_pool = training_frame(panel, min_date=args.train_min_date, max_date=train_cutoff)

    #   [ ... train ... | embargo (discarded) | val (last VAL_DAYS) ]
    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train; download more history.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    if args.model == "lgbm":
        print(f">> Training LightGBM (main model, kwargs={model_kwargs})")
        main_model = train_lgbm(train_df, val_df, FEATURE_COLUMNS, **model_kwargs)
        main_features = FEATURE_COLUMNS
    else:
        print(f">> Training XGBoost (main model, kwargs={model_kwargs})")
        main_model = train_xgb(train_df, val_df, FEATURE_COLUMNS, **model_kwargs)
        main_features = FEATURE_COLUMNS

    val_pred = main_model.predict(val_df[main_features])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")

    # ── ensemble：同时训练 XGBoost（不含 margin 特征）───────────────────────
    xgb_model = None
    xgb_features = None
    if args.ensemble:
        # margin 特征名中含 "margin"，过滤掉即可得到 M2 风格特征集
        xgb_features = [c for c in FEATURE_COLUMNS if "margin" not in c]
        print(f">> Training XGBoost (ensemble, {len(xgb_features)} features, no margin)")
        xgb_model = train_xgb(train_df, val_df, xgb_features)
        val_pred_xgb = xgb_model.predict(val_df[xgb_features])
        ic_xgb = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred_xgb, val_df["date"].to_numpy())
        print(f"   XGBoost validation rank IC: {ic_xgb:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=args.as_of)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={args.as_of}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=main_model.predict(pred_df[main_features]))

    # ── 融资杠杆过滤：排除 margin_buy_ratio 最高的一部分股票 ─────────────────
    if args.margin_filter > 0 and "margin_buy_ratio" in pred_df.columns:
        threshold = pred_df["margin_buy_ratio"].quantile(1 - args.margin_filter)
        mask = pred_df["margin_buy_ratio"] < threshold
        n_removed = (~mask).sum()
        pred_df = pred_df[mask | pred_df["margin_buy_ratio"].isna()]
        print(f"   margin filter: removed {n_removed} stocks "
              f"(margin_buy_ratio >= {threshold:.3f}, top {args.margin_filter:.0%})")

    scores = pred_df.set_index("stock_code")["score"]

    # ── ensemble 融合：LightGBM 排名 + XGBoost 排名各 50% ───────────────────
    if xgb_model is not None:
        scores_xgb = pred_df.set_index("stock_code").apply(
            lambda row: xgb_model.predict(
                pd.DataFrame([row[xgb_features]]))[0], axis=1
        )
        # 向量化预测（比逐行快）
        scores_xgb = pd.Series(
            xgb_model.predict(pred_df[xgb_features].values),
            index=pred_df["stock_code"].values,
        )
        scores = _blend_scores(scores, scores_xgb, alpha=0.5)
        print(f"   ensemble: blended LightGBM + XGBoost (50/50 rank blend)")

    weights = build_portfolio(scores, top_k=args.top_k,
                              weight_scheme=args.weight_scheme,
                              exp_alpha=args.exp_alpha)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")


if __name__ == "__main__":
    main()
