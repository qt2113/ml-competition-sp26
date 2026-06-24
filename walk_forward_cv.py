"""
Walk-forward cross-validation for the CSI500 stock-selection model.

每次运行自动追加一行到 cv_log.csv，方便对比实验。

用法示例：
  python walk_forward_cv.py                                      # XGBoost 默认参数
  python walk_forward_cv.py --model lgbm                        # 换成 LightGBM
  python walk_forward_cv.py --max-depth 4 --reg-lambda 5.0      # 调超参
  python walk_forward_cv.py --note "加了MACD特征"               # 附上实验备注
  python walk_forward_cv.py --n-windows 6 --val-days 42         # 更多/更长的窗口
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

# LightGBM 是可选依赖，没装也能用 XGBoost
try:
    import lightgbm as lgb
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

from features import (
    FEATURE_COLUMNS, TARGET_COLUMN,
    build_features, load_industry_map, load_fundamentals, load_fund_flow, load_margin,
    training_frame,
)

DATA_DIR     = Path(__file__).parent / "data"
LOG_FILE     = Path(__file__).parent / "cv_log.csv"   # 实验日志保存位置
EMBARGO_DAYS = 5

# ── 测试集起点 ──────────────────────────────────────────────────────────────
# 这条线之后的数据是锁住的 test set，在调参/选特征时绝对不能碰。
# 只在最后写报告时运行一次，把 IC 结果填进去。
#
# 怎么选这个日期？
#   - 越晚 → test set 越短（IC 估计噪音越大），但更接近5月的真实评估期
#   - 越早 → test set 越长（IC 更稳定），但离5月评估期更远
#   - 经验规则：test set 至少保留 40~60 个交易日（约 2~3 个月）
#
# 默认：2026-02-01，给 test set 留约 55 个交易日（2月到4月底）
TEST_START = pd.Timestamp("2026-02-01")


# ---------------------------------------------------------------------------
# 模型工厂：支持 XGBoost 和 LightGBM
# ---------------------------------------------------------------------------

def _make_xgb(**kwargs) -> xgb.XGBRegressor:
    defaults = dict(
        # ── 树的结构 ──────────────────────────────────────────────────────
        n_estimators=400,       # 树的数量。early stopping 会自动截断，设大一点没关系
                                # 调整范围：200~800
        max_depth=5,            # 每棵树最多问几层。越深越复杂越容易过拟合
                                # 金融数据建议 3~6
        min_child_weight=10,    # 叶节点最少需要多少只股票。越大越保守
                                # 建议范围：10~50

        # ── 正则化 ────────────────────────────────────────────────────────
        reg_lambda=1.0,         # L2 正则强度。越大模型越保守
                                # 建议范围：0.1~10

        # ── 随机采样 ──────────────────────────────────────────────────────
        subsample=0.8,          # 每棵树随机使用多少比例的样本（行）
                                # 建议范围：0.5~1.0
        colsample_bytree=0.8,   # 每棵树随机使用多少比例的特征（列）
                                # 建议范围：0.5~1.0

        # ── 学习速度 ──────────────────────────────────────────────────────
        learning_rate=0.05,     # 每棵树的贡献权重。越小越稳，需要更多树
                                # 建议范围：0.01~0.1

        # ── 一般不需要改 ──────────────────────────────────────────────────
        tree_method="hist",
        n_jobs=-1,
        early_stopping_rounds=30,  # val loss 连续30轮不改善就停止
        random_state=42,           # 固定随机种子，保证结果可复现（课件要求）
    )
    defaults.update(kwargs)
    return xgb.XGBRegressor(**defaults)


def _make_lgbm(**kwargs) -> "lgb.LGBMRegressor":
    if not _LGBM_AVAILABLE:
        raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
    defaults = dict(
        # ── 树的结构（参数含义和 XGBoost 基本一致）────────────────────────
        n_estimators=400,
        max_depth=5,
        num_leaves=8,           # leaf-wise生长的关键参数；扫描后8是甜点，建议范围：4~16
        min_child_samples=20,   # 对应 XGBoost 的 min_child_weight
                                # 建议范围：10~50

        # ── 正则化 ────────────────────────────────────────────────────────
        reg_lambda=5.0,         # L2正则；扫描后5.0最优，过小易过拟合

        # ── 随机采样 ──────────────────────────────────────────────────────
        subsample=0.6,          # 行采样0.6比0.8更防过拟合（扫描结论）
        subsample_freq=1,       # LightGBM 必须设这个才会启用行采样
        colsample_bytree=0.6,   # 列采样0.6同上

        # ── 学习速度 ──────────────────────────────────────────────────────
        learning_rate=0.05,

        # ── 一般不需要改 ──────────────────────────────────────────────────
        n_jobs=-1,
        random_state=42,        # 固定随机种子，保证结果可复现（课件要求）
        verbose=-1,             # 关掉训练时的输出
    )
    defaults.update(kwargs)
    return lgb.LGBMRegressor(**defaults)


def _fit_model(model, X_train, y_train, X_val, y_val, model_type: str):
    """统一的 fit 接口，屏蔽 XGBoost 和 LightGBM 的 API 差异。"""
    if model_type == "xgb":
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:  # lgbm
        callbacks = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(-1),   # 彻底关掉输出
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    return model


# ---------------------------------------------------------------------------
# 评估指标
# ---------------------------------------------------------------------------

def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """每天做一次截面 Spearman 相关，然后取均值。"""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:   # 当天股票太少，跳过（避免噪音）
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def _window_excess_return(
    scores: pd.Series,
    prices_raw: pd.DataFrame,
    index_df: pd.DataFrame,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    top_k: int = 50,
) -> float | None:
    """val_start 建仓，持有到 val_end，返回相对 CSI500 的超额收益。

    用 rank 权重（和 baseline_xgboost.py 保持一致）。
    数据不足时返回 None。
    """
    chosen = scores.nlargest(top_k)
    if len(chosen) < 30:
        return None

    # rank 权重：top-1 权重最大，线性递减
    n = len(chosen)
    ranks = np.arange(n, 0, -1, dtype=float)
    weights = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # 取首尾两天收盘价
    p = prices_raw[prices_raw["stock_code"].isin(chosen.index)]
    p = p[p["date"].isin([val_start, val_end])]
    p_pivot = p.pivot(index="date", columns="stock_code", values="close")
    if val_start not in p_pivot.index or val_end not in p_pivot.index:
        return None

    ret = p_pivot.loc[val_end] / p_pivot.loc[val_start] - 1
    valid = weights.index[weights.index.isin(ret.index) & ret.notna()]
    if len(valid) < 20:
        return None

    w = weights[valid] / weights[valid].sum()
    port_ret = float((w * ret[valid]).sum())

    # CSI500 基准收益
    idx = index_df[index_df["date"].isin([val_start, val_end])].set_index("date")["close"]
    if val_start not in idx.index or val_end not in idx.index:
        return None
    bench_ret = float(idx.loc[val_end] / idx.loc[val_start] - 1)

    return port_ret - bench_ret


# ---------------------------------------------------------------------------
# 窗口生成
# ---------------------------------------------------------------------------

def _make_windows(
    trading_dates: np.ndarray,
    n_windows: int,
    val_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    从 TEST_START 往前倒着切 n_windows 个不重叠的 val 窗口。
    返回 [(train_end, val_start, val_end), ...] 按时间正序排列。
    """
    pre_test = trading_dates[trading_dates < np.datetime64(TEST_START)]
    n = len(pre_test)

    windows = []
    end_idx = n

    for _ in range(n_windows):
        val_end_idx   = end_idx - 1
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - EMBARGO_DAYS - 1

        # 训练集前面至少需要 65 天历史才能算出 60 日特征
        if train_end_idx < 65:
            break

        windows.append((
            pd.Timestamp(pre_test[train_end_idx]),
            pd.Timestamp(pre_test[val_start_idx]),
            pd.Timestamp(pre_test[val_end_idx]),
        ))
        end_idx = val_start_idx

    windows.reverse()
    return windows


# ---------------------------------------------------------------------------
# CV 主循环
# ---------------------------------------------------------------------------

def run_cv(
    panel: pd.DataFrame,
    model_type: str = "xgb",
    n_windows: int = 5,
    val_days: int = 21,
    prices_raw: pd.DataFrame | None = None,   # 原始价格表，用于计算组合收益
    index_df: pd.DataFrame | None = None,      # CSI500 指数，用于计算超额收益
    top_k: int = 50,
    train_min_date=None,                       # 训练集最早日期；None = 不限制
    **model_kwargs,
) -> tuple[float, list[float], list[float]]:
    """
    跑完所有窗口，打印结果表，返回 (mean_ic, [每窗口ic], [每窗口超额收益])。
    prices_raw 和 index_df 都传入时才计算超额收益，否则跳过。
    """
    trading_dates = np.sort(panel["date"].unique())
    windows = _make_windows(trading_dates, n_windows, val_days)

    if not windows:
        raise RuntimeError("没有找到有效窗口，请减少 --n-windows 或 --val-days")

    show_ret = prices_raw is not None and index_df is not None
    header = (f"{'Win':<5} {'Train end':<13} {'Val':<25} {'IC':>8} {'Days':>5}"
              + (f" {'ExRet':>7}" if show_ret else ""))
    print(f"\n{header}")
    print("-" * len(header))

    ics: list[float] = []
    excess_rets: list[float] = []

    for i, (train_end, val_start, val_end) in enumerate(windows, 1):
        train_pool = training_frame(panel, min_date=train_min_date, max_date=train_end)
        val_df     = training_frame(panel, min_date=val_start, max_date=val_end)

        if len(train_pool) < 5_000 or val_df.empty:
            print(f"  W{i:<4} 跳过（数据不足）")
            continue

        inner_dates = np.sort(train_pool["date"].unique())
        inner_cut   = pd.Timestamp(inner_dates[-10])
        inner_train = train_pool[train_pool["date"] <  inner_cut]
        inner_val   = train_pool[train_pool["date"] >= inner_cut]

        model = _make_xgb(**model_kwargs) if model_type == "xgb" else _make_lgbm(**model_kwargs)
        _fit_model(
            model,
            inner_train[FEATURE_COLUMNS], inner_train[TARGET_COLUMN],
            inner_val[FEATURE_COLUMNS],   inner_val[TARGET_COLUMN],
            model_type,
        )

        pred   = model.predict(val_df[FEATURE_COLUMNS])
        ic     = _rank_ic(val_df[TARGET_COLUMN].to_numpy(), pred, val_df["date"].to_numpy())
        n_days = val_df["date"].nunique()

        # 超额收益：在 val_start 建仓，持有整个窗口
        exret_str = ""
        if show_ret:
            pred_at_start = panel[panel["date"] == val_start].dropna(subset=FEATURE_COLUMNS)
            if not pred_at_start.empty:
                scores = pd.Series(
                    model.predict(pred_at_start[FEATURE_COLUMNS]),
                    index=pred_at_start["stock_code"].values,
                )
                er = _window_excess_return(scores, prices_raw, index_df,
                                           val_start, val_end, top_k)
                if er is not None:
                    excess_rets.append(er)
                    exret_str = f" {er:>+7.2%}"

        val_str = f"{val_start.date()} → {val_end.date()}"
        print(f"  W{i:<4} {str(train_end.date()):<13} {val_str:<25} {ic:>8.4f} {n_days:>5}{exret_str}")
        ics.append(ic)

    print("-" * len(header))
    mean_ic  = float(np.mean(ics))        if ics        else float("nan")
    std_ic   = float(np.std(ics))         if ics        else float("nan")
    mean_er  = float(np.mean(excess_rets)) if excess_rets else float("nan")
    print(f"  {'Mean IC':<39} {mean_ic:>8.4f}")
    print(f"  {'Std  IC':<39} {std_ic:>8.4f}")
    if show_ret:
        print(f"  {'Mean ExRet':<39} {mean_er:>+8.2%}")
    return mean_ic, ics, excess_rets


# ---------------------------------------------------------------------------
# 实验日志
# ---------------------------------------------------------------------------

def _log(
    model_type: str,
    n_windows: int,
    val_days: int,
    model_kwargs: dict,
    mean_ic: float,
    std_ic: float,
    window_ics: list[float],
    note: str,
    excess_rets: list[float] | None = None,
) -> None:
    """把这次实验的结果追加到 cv_log.csv。"""
    is_new = not LOG_FILE.exists()

    mean_exret = (float(np.mean(excess_rets)) if excess_rets else float("nan"))

    # 最多记录10个窗口的 IC
    w_cols = {f"w{i+1}_ic": (f"{window_ics[i]:.4f}" if i < len(window_ics) else "")
              for i in range(10)}

    row = {
        "exp_id"    : datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model"     : model_type,
        "features"  : "|".join(FEATURE_COLUMNS),
        "n_features": len(FEATURE_COLUMNS),
        "n_windows" : n_windows,
        "val_days"  : val_days,
        "mean_ic"   : f"{mean_ic:.4f}",
        "std_ic"    : f"{std_ic:.4f}",
        **w_cols,
        "params"    : str(model_kwargs),
        "note"      : note,
        "mean_exret": ("" if np.isnan(mean_exret) else f"{mean_exret:+.4f}"),
    }

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new:
            writer.writeheader()
        writer.writerow(row)

    print(f"\n>> 已记录到 {LOG_FILE.name}  (exp_id={row['exp_id']})")


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward CV for CSI500 model")
    p.add_argument("--prices", default=str(DATA_DIR / "prices.parquet"))

    # ── 模型选择 ─────────────────────────────────────────────────────────────
    p.add_argument("--model", default="xgb", choices=["xgb", "lgbm"],
                   help="使用哪个模型：xgb（XGBoost）或 lgbm（LightGBM）。默认 xgb")

    # ── CV 结构参数 ───────────────────────────────────────────────────────────
    p.add_argument("--n-windows", type=int, default=5,
                   help="CV 窗口数量，越多评估越稳定但运行越慢（默认5）")
    p.add_argument("--val-days",  type=int, default=21,
                   help="每个 val 窗口的交易日数（默认21 ≈ 1个月）")
    p.add_argument("--train-min-date", default=None,
                   help="训练集起始日（YYYYMMDD）；设此值可排除过早的历史数据"
                        "（例如2年数据时用 20250601 只保留2025年6月以后的训练行）")

    # ── 超参数（命令行传入，方便对比）────────────────────────────────────────
    p.add_argument("--max-depth",        type=int,   default=5)
    p.add_argument("--n-estimators",     type=int,   default=400)
    p.add_argument("--min-child-weight", type=int,   default=10,
                   help="XGBoost 用；LightGBM 对应 min-child-samples")
    p.add_argument("--num-leaves",       type=int,   default=8,
                   help="LightGBM 专用；leaf-wise 生长的最大叶子数（默认8，XGBoost 忽略此参数）")
    p.add_argument("--reg-lambda",       type=float, default=1.0)
    p.add_argument("--learning-rate",    type=float, default=0.05)
    p.add_argument("--subsample",        type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)

    # ── 额外特征 ─────────────────────────────────────────────────────────────
    p.add_argument("--extra-features", default="",
                   help="额外加入的特征名，逗号分隔，如 ret_120d,close_over_ma20")

    # ── 可选数据源开关 ───────────────────────────────────────────────────────
    p.add_argument("--no-fundamentals", action="store_true",
                   help="跳过 PE/PB 基本面特征（覆盖率不足时训练行会被 dropna 删光）")
    p.add_argument("--no-fund-flow", action="store_true",
                   help="跳过主力资金流向特征（northbound 只覆盖少量股票时会删光训练行）")
    p.add_argument("--no-margin", action="store_true",
                   help="跳过融资融券特征")

    # ── 实验备注（写进日志）───────────────────────────────────────────────────
    p.add_argument("--note", default="",
                   help='本次实验的说明，例如 --note "加了MACD特征"')

    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    industry_map = load_industry_map()
    fundamentals = None if args.no_fundamentals else load_fundamentals()
    fund_flow    = None if args.no_fund_flow    else load_fund_flow()
    margin       = None if args.no_margin       else load_margin()
    if industry_map is not None:
        print(f"   行业中性化：已加载 {len(industry_map)} 只股票的行业标签")
    else:
        print("   行业中性化：跳过（先运行 python fetch_industry.py）")
    if fundamentals is not None:
        print(f"   基本面特征：已加载 PE/PB（{len(fundamentals):,} 行）")
    else:
        print("   基本面特征：跳过（先运行 python fetch_fundamentals.py）")
    if fund_flow is not None:
        print(f"   资金流向特征：已加载主力净流入（{len(fund_flow):,} 行）")
    else:
        print("   资金流向特征：跳过（先运行 python fetch_northbound.py）")
    if margin is not None:
        print(f"   融资融券特征：已加载（{len(margin):,} 行）")
    else:
        print("   融资融券特征：跳过（先运行 python fetch_margin.py）")
    extra_features = [f.strip() for f in args.extra_features.split(",") if f.strip()] or None
    panel = build_features(prices, industry_map=industry_map,
                           fundamentals=fundamentals, fund_flow=fund_flow,
                           margin=margin, extra_features=extra_features)
    print(f"   features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # LightGBM 用 min_child_samples，XGBoost 用 min_child_weight，名字不同但含义一样
    weight_key = "min_child_samples" if args.model == "lgbm" else "min_child_weight"
    model_kwargs = {
        "max_depth"       : args.max_depth,
        "n_estimators"    : args.n_estimators,
        weight_key        : args.min_child_weight,
        "reg_lambda"      : args.reg_lambda,
        "learning_rate"   : args.learning_rate,
        "subsample"       : args.subsample,
        "colsample_bytree": args.colsample_bytree,
    }
    if args.model == "lgbm":
        model_kwargs["num_leaves"] = args.num_leaves

    # 加载原始价格和指数（用于计算每窗口超额收益）
    prices_raw = pd.read_parquet(args.prices)
    prices_raw["date"] = pd.to_datetime(prices_raw["date"])
    index_path = DATA_DIR / "index.parquet"
    index_df = None
    if index_path.exists():
        index_df = pd.read_parquet(index_path)
        index_df["date"] = pd.to_datetime(index_df["date"])
    else:
        print("   警告：data/index.parquet 不存在，跳过超额收益计算")

    print(f"\n>> Walk-forward CV  "
          f"(model={args.model}, {args.n_windows} windows × {args.val_days} val days, "
          f"test locked from {TEST_START.date()})")
    print(f"   params: {model_kwargs}")

    mean_ic, ics, excess_rets = run_cv(
        panel,
        model_type     = args.model,
        n_windows      = args.n_windows,
        val_days       = args.val_days,
        prices_raw     = prices_raw,
        index_df       = index_df,
        train_min_date = args.train_min_date,
        **model_kwargs,
    )

    std_ic  = float(np.std(ics))         if ics         else float("nan")
    mean_er = float(np.mean(excess_rets)) if excess_rets else float("nan")
    print(f"\n>> Result: mean IC = {mean_ic:.4f}  std={std_ic:.4f}"
          + (f"  mean ExRet = {mean_er:+.2%}" if excess_rets else ""))

    _log(args.model, args.n_windows, args.val_days,
         model_kwargs, mean_ic, std_ic, ics, args.note, excess_rets)


if __name__ == "__main__":
    main()
