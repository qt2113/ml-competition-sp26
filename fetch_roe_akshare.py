"""
从 AKShare 下载季度 ROE（加权净资产收益率），前向填充到日频，保存到 data/roe.parquet。

数据源：ak.stock_financial_analysis_indicator(symbol, start_year)
ROE 列：位置索引 [19]（加权净资产收益率 %，YTD 累计）

防泄露处理：
  季报有披露延迟（例如 Q1=3月31日，但通常4月底才公告）。
  我们在报告期结束日期上加 DISCLOSURE_DELAY_DAYS 天作为"可用日期"，
  只有在可用日期之后才把这个 ROE 值前向填充到 panel 里。

运行：
    python fetch_roe_akshare.py          # 全量（自动跳过已有股票）
    python fetch_roe_akshare.py --reset  # 清空重来
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TEMP_CSV = DATA_DIR / "_roe_tmp.csv"
OUT_PARQ = DATA_DIR / "roe.parquet"

# 按季报类型设置不同披露延迟（A股监管要求）：
#   Q1(3/31)  → 30天内披露（4月30日前）
#   H1(6/30)  → 60天内披露（8月31日前）
#   Q3(9/30)  → 30天内披露（10月31日前）
#   年报(12/31)→ 4个月内披露（次年4月30日前）→ 用120天
# 各值均略微放宽作为保守估计；可调整
DISCLOSURE_DELAY = {3: 45, 6: 65, 9: 45, 12: 120}   # 月份 → 延迟天数

START_YEAR  = "2024"  # 下载起始年份；2024往后有约2年数据，足够使用
SLEEP_SEC   = 0.1     # 每只股票请求间隔；接口本身约3~4秒，无需额外长等
PRINT_EVERY = 50      # 每下载多少只股票打印进度


def _fetch_one(ak, code: str) -> pd.DataFrame | None:
    """
    返回单只股票的季度 ROE 表：columns = [report_date, available_date, roe]
    失败或数据为空返回 None。
    """
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=START_YEAR)
        if df is None or df.empty:
            return None

        # col[0]=报告期日期，col[19]=加权净资产收益率(%)
        result = df.iloc[:, [0, 19]].copy()
        result.columns = ["report_date", "roe"]
        result["report_date"] = pd.to_datetime(result["report_date"], errors="coerce")
        result["roe"] = pd.to_numeric(result["roe"], errors="coerce")
        result = result.dropna(subset=["report_date", "roe"])

        # 按报告季度设不同披露延迟（年报最晚4月底公告，需要120天缓冲）
        delays = result["report_date"].dt.month.map(DISCLOSURE_DELAY)
        result["available_date"] = result["report_date"] + pd.to_timedelta(delays, unit="D")

        # 年化 ROE：YTD 累计值 → 全年等效（消除跨季度比较的季节性偏差）
        # Q1(3月)×4, H1(6月)×2, Q3(9月)×4/3, 年报(12月)×1
        annualize = {3: 4.0, 6: 2.0, 9: 4/3, 12: 1.0}
        result["roe"] = result["roe"] * result["report_date"].dt.month.map(annualize)
        result = result.sort_values("available_date").reset_index(drop=True)
        return result if not result.empty else None

    except Exception as e:
        print(f"   [错误] {code}: {e}")
        return None


def fetch_roe(reset: bool = False) -> None:
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装：pip install akshare")

    if reset and TEMP_CSV.exists():
        TEMP_CSV.unlink()
        print("   已清空临时文件，重新下载")

    # 断点续传
    done_codes: set[str] = set()
    if TEMP_CSV.exists():
        try:
            done_codes = set(
                pd.read_csv(TEMP_CSV, usecols=["stock_code"])["stock_code"].astype(str)
            )
            print(f"   断点续传：跳过已完成的 {len(done_codes)} 只股票")
        except Exception:
            pass

    constituents = pd.read_csv(DATA_DIR / "constituents.csv", dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].str.zfill(6)
    codes = [c for c in constituents["stock_code"].tolist() if c not in done_codes]
    print(f"   还需下载 {len(codes)} 只股票")

    if not codes:
        _finalize()
        return

    failed = []
    for i, code in enumerate(codes):
        df = _fetch_one(ak, code)
        if df is None or df.empty:
            failed.append(code)
        else:
            df["stock_code"] = code
            df[["stock_code", "report_date", "available_date", "roe"]].to_csv(
                TEMP_CSV, mode="a", header=not TEMP_CSV.exists(), index=False
            )

        if (i + 1) % PRINT_EVERY == 0:
            print(f"   进度：{i+1}/{len(codes)}，失败 {len(failed)} 只")

        time.sleep(SLEEP_SEC)

    if failed:
        print(f"\n   [警告] 最终失败 {len(failed)} 只：{failed[:20]}")

    _finalize()


def _finalize() -> None:
    """
    把季度 ROE 前向填充到每个交易日，生成 (date, stock_code, roe) 的日频表。
    关键：只有在 available_date <= 当日 时，这条季报数据才会出现，防止泄露。
    """
    if not TEMP_CSV.exists():
        print("   临时文件不存在")
        return

    quarterly = pd.read_csv(TEMP_CSV, dtype={"stock_code": str})
    quarterly["report_date"]    = pd.to_datetime(quarterly["report_date"])
    quarterly["available_date"] = pd.to_datetime(quarterly["available_date"])
    quarterly["roe"] = pd.to_numeric(quarterly["roe"], errors="coerce")
    quarterly = quarterly.dropna(subset=["roe"])

    # 读取所有交易日（从 prices.parquet 取）
    prices = pd.read_parquet(DATA_DIR / "prices.parquet", columns=["date", "stock_code"])
    prices["date"] = pd.to_datetime(prices["date"])
    trading_dates = pd.to_datetime(sorted(prices["date"].unique()))

    # 对每只股票：把季度 ROE 前向填充到交易日
    rows = []
    for code, grp in quarterly.groupby("stock_code"):
        grp = grp.sort_values("available_date")

        # 以 available_date 为索引，对齐到交易日后前向填充
        tmp = pd.DataFrame({"date": trading_dates})
        tmp = tmp.merge(
            grp[["available_date", "roe"]].rename(columns={"available_date": "date"}),
            on="date", how="left"
        )
        tmp["roe"] = tmp["roe"].ffill()   # 前向填充：用最近一次公告的ROE
        tmp = tmp.dropna(subset=["roe"])
        tmp["stock_code"] = code
        rows.append(tmp[["date", "stock_code", "roe"]])

    if not rows:
        print("   无有效数据")
        return

    result = pd.concat(rows, ignore_index=True)
    result = result.sort_values(["date", "stock_code"]).reset_index(drop=True)
    result.to_parquet(OUT_PARQ, index=False)

    print(f"\n>> 已保存至 {OUT_PARQ}，共 {len(result):,} 行")
    print(f"   日期：{result['date'].min().date()} ~ {result['date'].max().date()}")
    print(f"   股票：{result['stock_code'].nunique()} 只")
    print(f"   ROE 非空率：{result['roe'].notna().mean():.1%}")
    print(f"   ROE 分布：均值 {result['roe'].mean():.2f}%，中位数 {result['roe'].median():.2f}%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true")
    args = p.parse_args()
    print(">> 从 AKShare 拉取季度 ROE（支持断点续传）")
    fetch_roe(reset=args.reset)


if __name__ == "__main__":
    main()
