"""
从 AKShare 下载申万一级行业分类，生成 data/industry_map.csv。

只需运行一次（或行业调整后重新运行）：
    python fetch_industry.py

输出：data/industry_map.csv，包含两列 [stock_code, industry]
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def fetch_sw_industry() -> pd.DataFrame:
    """用申万一级行业指数成分接口，批量获取股票→行业映射。"""
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("请先安装 akshare：pip install akshare")

    print(">> 正在获取申万一级行业列表...")
    info = ak.sw_index_first_info()
    # 列名可能因系统编码而乱码，用位置索引：col0=行业代码, col1=行业名称
    code_col = info.columns[0]
    name_col = info.columns[1]
    # 代码格式 "801010.SI" → 取前6位数字
    industries = [
        (row[code_col].split(".")[0], row[name_col])
        for _, row in info.iterrows()
        if str(row[code_col]).split(".")[0].isdigit()
    ]
    print(f"   共 {len(industries)} 个一级行业")

    rows = []
    for idx_code, name in industries:
        try:
            cons = ak.index_component_sw(symbol=idx_code)
            # 列名也可能乱码，取第2列（证券代码）
            code_col_c = cons.columns[1]
            for code in cons[code_col_c].astype(str):
                rows.append({"stock_code": code.zfill(6), "industry": name})
            time.sleep(0.2)   # 避免请求过快被限速；可调小
        except Exception as e:
            print(f"   [警告] {name}({idx_code}) 获取失败：{e}")

    df = pd.DataFrame(rows).drop_duplicates(subset="stock_code")
    return df


def main():
    df = fetch_sw_industry()
    out = DATA_DIR / "industry_map.csv"
    df.to_csv(out, index=False)
    print(f"\n>> 保存至 {out}，共 {len(df)} 只股票有行业标签")

    # 验证覆盖率
    constituents = pd.read_csv(DATA_DIR / "constituents.csv",
                               dtype={"stock_code": str})
    constituents["stock_code"] = constituents["stock_code"].str.zfill(6)
    merged = constituents.merge(df, on="stock_code", how="left")
    n_covered = merged["industry"].notna().sum()
    print(f"   CSI500 成分股覆盖率：{n_covered}/{len(merged)} "
          f"({100*n_covered/len(merged):.1f}%)")
    if n_covered > 0:
        print(f"   行业分布 TOP10：\n{merged['industry'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
