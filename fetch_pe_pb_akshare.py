"""
从 AKShare（百度股市通数据源）下载日频 PE(TTM) / PB 数据。

接口：ak.stock_zh_valuation_baidu(symbol, indicator)
  - indicator="市盈率(TTM)" → PE TTM（日频历史，约1年）
  - indicator="市净率"      → PB（日频历史，约1年）

注意：Baidu 接口只返回近1年数据（约366天）。
      prices.parquet 从2025-01-02开始，前4个月（1-4月）PE/PB会是NaN，可接受。

支持断点续传：中断后重新运行自动跳过已下载的股票。

运行：
    python fetch_pe_pb_akshare.py          # 全量（自动跳过已有股票）
    python fetch_pe_pb_akshare.py --reset  # 清空已有缓存，从头开始
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TEMP_CSV = DATA_DIR / "_pe_pb_tmp.csv"      # 增量缓存，每只股票追加一次
OUT_PARQ = DATA_DIR / "fundamentals.parquet"

SLEEP_SEC   = 0.5   # 每次请求后等待秒数；每只股票需两次请求（PE+PB），建议 0.3~0.8
PRINT_EVERY = 20    # 每下载多少只股票打印一次进度；可调


def _fetch_one(ak, code: str) -> pd.DataFrame | None:
    """
    下载单只股票的历史 PE(TTM) + PB，合并成一张表。
    返回 DataFrame(date, pe_ttm, pb)，失败则返回 None。
    """
    try:
        df_pe = ak.stock_zh_valuation_baidu(symbol=code, indicator="市盈率(TTM)")
        df_pb = ak.stock_zh_valuation_baidu(symbol=code, indicator="市净率")

        if df_pe is None or df_pe.empty:
            return None

        df_pe = df_pe.rename(columns={"value": "pe_ttm"})
        df_pb = df_pb.rename(columns={"value": "pb"}) if df_pb is not None else None

        df = df_pe.copy()
        if df_pb is not None and not df_pb.empty:
            df = df.merge(df_pb, on="date", how="outer")
        else:
            df["pb"] = float("nan")

        df["date"]   = pd.to_datetime(df["date"], errors="coerce")
        df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
        df["pb"]     = pd.to_numeric(df["pb"],     errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")

        return df if not df.empty else None

    except Exception as e:
        print(f"   [错误] {code}: {e}")
        return None


def fetch_pe_pb(reset: bool = False) -> None:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装：pip install akshare")

    # ── 清空模式 ─────────────────────────────────────────────────────────────
    if reset and TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print("   已清空临时文件，重新下载")

    # ── 读取已下载的股票（断点续传）──────────────────────────────────────────
    done_codes: set[str] = set()
    if TEMP_CSV.exists():
        try:
            done_codes = set(
                pd.read_csv(TEMP_CSV, usecols=["stock_code"])["stock_code"].astype(str)
            )
            print(f"   断点续传：跳过已完成的 {len(done_codes)} 只股票")
        except Exception:
            print("   临时文件可能损坏，从头开始")

    # ── 读取成分股列表 ────────────────────────────────────────────────────────
    constituents = pd.read_csv(DATA_DIR / "constituents.csv", dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].str.zfill(6)
    codes = [c for c in constituents["stock_code"].tolist() if c not in done_codes]
    print(f"   还需下载 {len(codes)} 只股票（每只需两次请求：PE+PB）")

    if not codes:
        print("   所有股票已下载完毕，直接生成 parquet")
        _finalize()
        return

    # ── 主循环 ────────────────────────────────────────────────────────────────
    failed = []
    for i, code in enumerate(codes):
        df = _fetch_one(ak, code)

        if df is None or df.empty:
            failed.append(code)
        else:
            df["stock_code"] = code
            df[["date", "stock_code", "pe_ttm", "pb"]].to_csv(
                TEMP_CSV,
                mode="a",
                header=not TEMP_CSV.exists(),
                index=False,
            )

        if (i + 1) % PRINT_EVERY == 0:
            print(f"   进度：{i+1}/{len(codes)}，失败 {len(failed)} 只")

        time.sleep(SLEEP_SEC)   # 每只股票请求两次，稍加等待避免被限速

    if failed:
        print(f"\n   [警告] 最终失败 {len(failed)} 只：{failed[:20]}")
        if len(failed) > 20:
            print(f"   ... 共 {len(failed)} 只")

    _finalize()


def _finalize() -> None:
    """把临时 CSV 汇总成 parquet。"""
    if not TEMP_CSV.exists():
        print("   临时文件不存在，无数据可保存")
        return

    df = pd.read_csv(TEMP_CSV, dtype={"stock_code": str})
    df["date"]   = pd.to_datetime(df["date"])
    df["pe_ttm"] = pd.to_numeric(df["pe_ttm"], errors="coerce")
    df["pb"]     = pd.to_numeric(df["pb"],     errors="coerce")

    # 去重（断点续传偶尔会有重复行）
    df = df.drop_duplicates(subset=["date", "stock_code"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    df.to_parquet(OUT_PARQ, index=False)

    print(f"\n>> 已保存至 {OUT_PARQ}，共 {len(df):,} 行")
    print(f"   日期：{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"   股票：{df['stock_code'].nunique()} 只")
    print(f"   PE 非空率：{df['pe_ttm'].notna().mean():.1%}")
    print(f"   PB 非空率：{df['pb'].notna().mean():.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="清空已有缓存，从头开始")
    args = p.parse_args()
    print(">> 从 AKShare（百度）拉取 PE/PB 数据（支持断点续传）")
    fetch_pe_pb(reset=args.reset)


if __name__ == "__main__":
    main()
