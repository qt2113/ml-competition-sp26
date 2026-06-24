"""
4月rolling 5天窗口完整检查
用法:
  python april_rolling.py                     # Sub1 默认 config
  python april_rolling.py --train-min 20251101  # 缩短训练窗口测regime adapt
  python april_rolling.py --train-min 20260101  # 极短训练（仅2026年）
"""
import subprocess, sys, re, argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent

ap = argparse.ArgumentParser()
ap.add_argument("--train-min", default=None, help="训练集最早日期 YYYYMMDD")
ap.add_argument("--no-fund-flow", action="store_true", help="跳过 northbound（修复31只股票bug）")
ap.add_argument("--no-fundamentals", action="store_true")
ap.add_argument("--reg-lambda", type=float, default=0.1)
ap.add_argument("--colsample-bytree", type=float, default=0.7)
ap.add_argument("--num-leaves", type=int, default=8)
ap.add_argument("--subsample", type=float, default=0.6)
ap.add_argument("--min-child-weight", type=int, default=20)
ap.add_argument("--top-k", type=int, default=50)
ap.add_argument("--weight-scheme", default="rank", choices=["rank", "exp"])
ap.add_argument("--exp-alpha", type=float, default=3.0)
args_cli = ap.parse_args()

# Sub1 / submission_sub2_final.csv 完全相同的config
ARGS = [
    "--model", "lgbm", "--ensemble",
    "--num-leaves", str(args_cli.num_leaves),
    "--reg-lambda", str(args_cli.reg_lambda),
    "--subsample", str(args_cli.subsample),
    "--colsample-bytree", str(args_cli.colsample_bytree),
    "--min-child-weight", str(args_cli.min_child_weight),
    "--top-k", str(args_cli.top_k),
    "--weight-scheme", args_cli.weight_scheme,
    "--exp-alpha", str(args_cli.exp_alpha),
]
if args_cli.train_min:
    ARGS += ["--train-min-date", args_cli.train_min]
    print(f">> 用 --train-min-date {args_cli.train_min}")
if args_cli.no_fund_flow:
    ARGS += ["--no-fund-flow"]
    print(f">> 跳过 fund_flow (修复31只股票bug)")
if args_cli.no_fundamentals:
    ARGS += ["--no-fundamentals"]
    print(f">> 跳过 fundamentals")

# 取所有交易日，as_of from 3/24 to 4/23（这样score window都能完全落在4月内）
prices = pd.read_parquet(ROOT / "data" / "prices.parquet")
prices["date"] = pd.to_datetime(prices["date"])
all_dates = sorted(prices["date"].unique())

# as_of 范围：让 score window 5天都在 ≤ 5/8
asof_candidates = [d for d in all_dates if pd.Timestamp("2026-03-23") <= d <= pd.Timestamp("2026-04-30")]

print(f">> 共 {len(asof_candidates)} 个 as_of 候选日")
print(f"   from {asof_candidates[0].date()} to {asof_candidates[-1].date()}")
print()

results = []
for i, asof in enumerate(asof_candidates, 1):
    # 找 score start = 下一个交易日，score end = 之后第5个交易日
    asof_idx = all_dates.index(asof)
    if asof_idx + 5 >= len(all_dates):
        print(f"  [{i}] {asof.date()} SKIP (剩余交易日不足5个)")
        continue
    start = all_dates[asof_idx + 1]
    end   = all_dates[asof_idx + 5]
    asof_s, start_s, end_s = asof.strftime("%Y%m%d"), start.strftime("%Y%m%d"), end.strftime("%Y%m%d")

    r1 = subprocess.run(
        [sys.executable, "baseline_xgboost.py"] + ARGS
        + ["--as-of", asof_s, "--out", "_tmp_apr.csv"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if r1.returncode != 0:
        err = (r1.stderr or r1.stdout).strip().splitlines()[-1][:60]
        print(f"  [{i}] {asof.date()} SKIP ({err})")
        continue

    ic_match = re.search(r"validation rank IC:\s*([\-\d.]+)", r1.stdout)
    ic = float(ic_match.group(1)) if ic_match else float("nan")

    r2 = subprocess.run(
        [sys.executable, "score_submission.py", "_tmp_apr.csv",
         "--start", start_s, "--end", end_s],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    er_match = re.search(r"excess return\s*:\s*([+\-]?[\d.]+)%", r2.stdout)
    er = float(er_match.group(1)) if er_match else float("nan")
    results.append((asof.date(), start.date(), end.date(), ic, er))
    flag = "+" if er > 0 else "-"
    print(f"  [{i:>2}] as_of={asof.date()}  score={start.date()}->{end.date()}  IC={ic:+.4f}  ExRet={er:+.3f}% {flag}")

(ROOT / "_tmp_apr.csv").unlink(missing_ok=True)

if not results:
    print("没有成功的窗口。")
    sys.exit(1)

print()
print("=" * 70)
ers = [r[4] for r in results]
ics = [r[3] for r in results]
n = len(ers)
avg = sum(ers)/n
std = (sum((e-avg)**2 for e in ers)/n)**0.5
pos = sum(1 for e in ers if e > 0)
median = sorted(ers)[n//2]
best = max(ers)
worst = min(ers)

print(f"  样本数        : {n}")
print(f"  平均 ExRet    : {avg:+.3f}%")
print(f"  中位数 ExRet  : {median:+.3f}%")
print(f"  Std           : {std:.3f}%")
print(f"  正收益率      : {pos}/{n} = {pos/n*100:.0f}%")
print(f"  最佳单窗口    : {best:+.3f}%")
print(f"  最差单窗口    : {worst:+.3f}%")
print(f"  平均 IC       : {sum(ics)/n:+.4f}")
