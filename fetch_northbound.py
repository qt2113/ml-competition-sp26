"""
从 AKShare 下载每只股票的主力资金净流入数据（日频，近120个交易日）。

接口：ak.stock_individual_fund_flow(stock, market)
  - market: "sh"（沪市，代码6开头）/ "sz"（深市，代码0/3开头）
  - 返回约120天历史，包含主力/超大单/大单/中单/小单净流入金额及占比

核心特征：
  - main_pct：主力净流入占比（占当日成交额%），已标准化，适合直接做因子
  - main_net：主力净流入金额（绝对值），适合做排名因子

注：2025-10-30之前的数据为 NaN，LightGBM 原生支持 NaN，不影响训练。

支持断点续传：中断后重新运行自动跳过已下载的股票。

运行：
    python fetch_northbound.py          # 全量（自动跳过已有股票）
    python fetch_northbound.py --reset  # 清空已有缓存，从头开始
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA_DIR  = Path(__file__).parent / "data"
TEMP_CSV  = DATA_DIR / "_northbound_tmp.csv"
OUT_PARQ  = DATA_DIR / "northbound.parquet"

SLEEP_SEC   = 1.5   # 每次请求后等待秒数；1.5s 避免被东方财富服务器限速
PRINT_EVERY = 20    # 每下载多少只股票打印一次进度


def _market(code: str) -> str:
    """根据股票代码判断市场：6开头→沪市sh，其余→深市sz"""
    return "sh" if code.startswith("6") else "sz"


# 固定列顺序（接口返回的列顺序是稳定的，但名称在 Windows 下可能乱码）
_FUND_COLS = [
    "date", "close", "pct_chg",
    "main_net", "main_pct",         # 主力（超大单+大单）
    "super_net", "super_pct",       # 超大单（>=100万）
    "big_net", "big_pct",           # 大单（20~100万）
    "mid_net", "mid_pct",           # 中单（4~20万）
    "small_net", "small_pct",       # 小单（<4万）
]


def _fetch_one(ak, code: str) -> pd.DataFrame | None:
    """下载单只股票的主力资金流向历史。失败返回 None。"""
    try:
        df = ak.stock_individual_fund_flow(stock=code, market=_market(code))
        if df is None or df.empty:
            return None
        df.columns = _FUND_COLS
        df["date"]      = pd.to_datetime(df["date"], errors="coerce")
        df["main_net"]  = pd.to_numeric(df["main_net"],  errors="coerce")
        df["main_pct"]  = pd.to_numeric(df["main_pct"],  errors="coerce")
        df["super_pct"] = pd.to_numeric(df["super_pct"], errors="coerce")
        df["big_pct"]   = pd.to_numeric(df["big_pct"],   errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
        return df if not df.empty else None
    except Exception as e:
        print(f"   [错误] {code}: {e}")
        return None


def fetch_fund_flow(reset: bool = False) -> None:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装：pip install akshare")

    # ── 清空模式 ─────────────────────────────────────────────────────────────
    if reset and TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print("   已清空临时文件，重新下载")

    # ── 断点续传：读取已完成的股票 ────────────────────────────────────────────
    done_codes: set[str] = set()
    if TEMP_CSV.exists():
        try:
            done_codes = set(
                pd.read_csv(TEMP_CSV, usecols=["stock_code"])["stock_code"]
                .astype(str).str.zfill(6)
            )
            print(f"   断点续传：跳过已完成的 {len(done_codes)} 只股票")
        except Exception:
            print("   临时文件可能损坏，从头开始")

    # ── 读取成分股列表 ────────────────────────────────────────────────────────
    constituents = pd.read_csv(DATA_DIR / "constituents.csv", dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].str.zfill(6)
    codes = [c for c in constituents["stock_code"].tolist() if c not in done_codes]
    print(f"   还需下载 {len(codes)} 只股票")

    if not codes:
        print("   所有股票已下载完毕，直接生成 parquet")
        _finalize()
        return

    # ── 主循环 ────────────────────────────────────────────────────────────────
    failed = []
    for i, code in enumerate(codes):
        df = _fetch_one(ak, code)

        if df is None:
            failed.append(code)
            # 写占位行，断点续传时不会重复请求
            placeholder = pd.DataFrame([{
                "date": pd.NaT, "stock_code": code,
                "main_net": float("nan"), "main_pct": float("nan"),
                "super_pct": float("nan"), "big_pct": float("nan"),
            }])
            placeholder.to_csv(TEMP_CSV, mode="a", header=not TEMP_CSV.exists(), index=False)
        else:
            df["stock_code"] = code
            keep = ["date", "stock_code", "main_net", "main_pct", "super_pct", "big_pct"]
            df[keep].to_csv(TEMP_CSV, mode="a", header=not TEMP_CSV.exists(), index=False)

        if (i + 1) % PRINT_EVERY == 0:
            print(f"   进度：{i+1}/{len(codes)}，失败 {len(failed)} 只")

        time.sleep(SLEEP_SEC)

    if failed:
        print(f"\n   [警告] 失败 {len(failed)} 只：{failed[:20]}")

    _finalize()


def _finalize() -> None:
    """把临时 CSV 汇总成 parquet，剔除占位行。"""
    if not TEMP_CSV.exists():
        print("   临时文件不存在，无数据可保存")
        return

    df = pd.read_csv(TEMP_CSV, dtype={"stock_code": str})
    df["stock_code"] = df["stock_code"].str.zfill(6)
    df["date"]       = pd.to_datetime(df["date"], errors="coerce")
    for col in ["main_net", "main_pct", "super_pct", "big_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date"])   # 剔除占位行
    df = df.drop_duplicates(subset=["date", "stock_code"])
    df = df.sort_values(["date", "stock_code"]).reset_index(drop=True)
    df.to_parquet(OUT_PARQ, index=False)

    print(f"\n>> 已保存至 {OUT_PARQ}，共 {len(df):,} 行")
    print(f"   日期：{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"   股票：{df['stock_code'].nunique()} 只")
    print(f"   main_pct 非空率：{df['main_pct'].notna().mean():.1%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="清空已有缓存，从头开始")
    args = p.parse_args()
    print(">> 从 AKShare 拉取主力资金净流入数据（支持断点续传）")
    print("   预计耗时：约 5~8 分钟（499只 × 0.6秒/只）")
    fetch_fund_flow(reset=args.reset)


if __name__ == "__main__":
    main()
