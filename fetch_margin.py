"""
从 AKShare 下载A股融资融券历史数据（深交所 + 上交所）。

接口（按日期批量拉取，无需逐股请求）：
  - ak.stock_margin_detail_szse(date)  深市，每次返回约2000条记录
  - ak.stock_margin_detail_sse(date)   沪市，每次返回约1700条记录（6开头过滤）

核心特征（衍生）：
  - margin_buy：当日融资买入额（绝对量）
  - margin_bal：融资余额（存量杠杆），越高越拥挤
  - margin_bal_chg：融资余额日变化额（正=杠杆加速，负=去杠杆）

约320个交易日 × 2次请求/天 = 640次请求，约5~8分钟。

支持断点续传：已下载的日期自动跳过。

运行：
    python fetch_margin.py          # 全量（自动跳过已下载日期）
    python fetch_margin.py --reset  # 清空重来
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TEMP_CSV = DATA_DIR / "_margin_tmp.csv"
OUT_PARQ = DATA_DIR / "margin.parquet"

SLEEP_SEC   = 1.0   # 每个日期的两次请求之间总等待；分摊到深/沪各0.5s
PRINT_EVERY = 20    # 每下载多少个日期打印进度


def _fetch_one_date(ak, date_str: str) -> pd.DataFrame | None:
    """
    拉取 date_str（格式 YYYYMMDD）的深市+沪市融资融券数据，合并返回。
    失败返回 None。
    """
    frames = []

    # ── 深市 ─────────────────────────────────────────────────────────────────
    try:
        df_sz = ak.stock_margin_detail_szse(date=date_str)
        df_sz.columns = [
            "stock_code", "name",
            "margin_bal", "margin_buy",
            "short_bal_vol", "short_sell_vol", "short_bal", "total_bal",
        ]
        df_sz["stock_code"] = df_sz["stock_code"].astype(str).str.zfill(6)
        # 只保留 A 股代码（000/001/002/003/300 开头）
        df_sz = df_sz[df_sz["stock_code"].str.match(r"^(000|001|002|003|300)")]
        df_sz["date"] = pd.Timestamp(date_str)
        frames.append(df_sz[["date", "stock_code", "margin_bal", "margin_buy"]])
    except Exception:
        pass

    time.sleep(SLEEP_SEC / 2)

    # ── 沪市 ─────────────────────────────────────────────────────────────────
    try:
        df_sh = ak.stock_margin_detail_sse(date=date_str)
        df_sh.columns = [
            "date_col", "stock_code", "name",
            "margin_bal", "margin_buy", "margin_repay",
            "short_sell_vol", "short_bal_vol", "short_bal",
        ]
        df_sh["stock_code"] = df_sh["stock_code"].astype(str).str.zfill(6)
        # 只保留 A 股代码（6 开头）；过滤 ETF（510/513 等）
        df_sh = df_sh[df_sh["stock_code"].str.startswith("6")]
        df_sh["date"] = pd.Timestamp(date_str)
        frames.append(df_sh[["date", "stock_code", "margin_bal", "margin_buy"]])
    except Exception:
        pass

    time.sleep(SLEEP_SEC / 2)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    for col in ["margin_bal", "margin_buy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df if not df.empty else None


def fetch_margin(reset: bool = False) -> None:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装：pip install akshare")

    # ── 清空模式 ─────────────────────────────────────────────────────────────
    if reset and TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print("   已清空临时文件，重新下载")

    # ── 断点续传：读取已完成的日期 ────────────────────────────────────────────
    done_dates: set[str] = set()
    if TEMP_CSV.exists():
        try:
            done_dates = set(
                pd.read_csv(TEMP_CSV, usecols=["date"])["date"].astype(str)
            )
            print(f"   断点续传：跳过已完成的 {len(done_dates)} 个日期")
        except Exception:
            print("   临时文件可能损坏，从头开始")

    # ── 读取需要下载的日期列表（来自 prices.parquet）──────────────────────────
    prices = pd.read_parquet(DATA_DIR / "prices.parquet", columns=["date"])
    all_dates = sorted(prices["date"].unique())
    date_strs = [pd.Timestamp(d).strftime("%Y%m%d") for d in all_dates]
    to_fetch = [d for d in date_strs if d not in done_dates]
    print(f"   需要下载 {len(to_fetch)} 个交易日（共 {len(date_strs)} 个，每日2次请求）")

    if not to_fetch:
        print("   所有日期已下载完毕，直接生成 parquet")
        _finalize()
        return

    # ── 主循环 ────────────────────────────────────────────────────────────────
    failed = []
    for i, date_str in enumerate(to_fetch):
        df = _fetch_one_date(ak, date_str)

        if df is None or df.empty:
            failed.append(date_str)
        else:
            df.to_csv(TEMP_CSV, mode="a", header=not TEMP_CSV.exists(), index=False)

        if (i + 1) % PRINT_EVERY == 0:
            print(f"   进度：{i+1}/{len(to_fetch)}，失败 {len(failed)} 天")

    if failed:
        print(f"\n   [警告] 拉取失败的日期 {len(failed)} 天：{failed[:10]}")

    _finalize()


def _finalize() -> None:
    """把临时 CSV 汇总成 parquet。"""
    if not TEMP_CSV.exists():
        print("   临时文件不存在，无数据可保存")
        return

    df = pd.read_csv(TEMP_CSV, dtype={"stock_code": str})
    df["stock_code"]  = df["stock_code"].str.zfill(6)
    df["date"]        = pd.to_datetime(df["date"], errors="coerce")
    df["margin_bal"]  = pd.to_numeric(df["margin_bal"],  errors="coerce")
    df["margin_buy"]  = pd.to_numeric(df["margin_buy"],  errors="coerce")

    df = df.dropna(subset=["date"])
    df = df.drop_duplicates(subset=["date", "stock_code"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    df.to_parquet(OUT_PARQ, index=False)

    print(f"\n>> 已保存至 {OUT_PARQ}，共 {len(df):,} 行")
    print(f"   日期：{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"   股票：{df['stock_code'].nunique()} 只（每日均值 {len(df)/df['date'].nunique():.0f} 只）")
    print(f"   margin_buy 非空率：{df['margin_buy'].notna().mean():.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="清空已有缓存，从头开始")
    args = p.parse_args()
    print(">> 从 AKShare 拉取融资融券历史数据（深交所+上交所，支持断点续传）")
    print("   预计耗时：约5~8分钟（每日2次API请求）")
    fetch_margin(reset=args.reset)


if __name__ == "__main__":
    main()
