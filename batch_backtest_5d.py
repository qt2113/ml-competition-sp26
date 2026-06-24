"""
5天滚动回测：用当前最优 ensemble 模型，看近一个月各 5 天持仓窗口的超额收益。
用法：python batch_backtest_5d.py
"""
import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).parent

# (标签, as_of预测日, score_start, score_end) — 每窗口恰好 5 个交易日
WINDOWS = [
    ("Apr-W1", "20260407", "20260408", "20260414"),  # 4/8,9,10,13,14
    ("Apr-W2", "20260414", "20260415", "20260421"),  # 4/15,16,17,20,21
    ("Apr-W3", "20260421", "20260422", "20260428"),  # 4/22,23,24,27,28
    ("Apr-W4", "20260428", "20260429", "20260430"),  # 4/29,30 (劳动节前仅2天，供参考)
    ("Sub1-ref","20260430", "20260506", "20260508"), # sub1实际评测窗口，as-of用4/30(节前最后交易日)
]

MODEL_ARGS = [
    "--model", "lgbm", "--ensemble",
    "--num-leaves", "8",
    "--reg-lambda", "0.1",
    "--subsample", "0.6",
    "--colsample-bytree", "0.7",
    "--min-child-weight", "20",
    "--top-k", "50",
    "--weight-scheme", "rank",
]

results = []
for label, as_of, start, end in WINDOWS:
    r1 = subprocess.run(
        [sys.executable, "baseline_xgboost.py"] + MODEL_ARGS +
        ["--as-of", as_of, "--out", "_tmp_5d.csv"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if r1.returncode != 0:
        err = (r1.stderr or r1.stdout).strip().splitlines()[-1][:80]
        print(f"  {label}: 跳过（{err}）")
        continue

    ic_match = re.search(r"validation rank IC:\s*([\-\d.]+)", r1.stdout)
    ic = float(ic_match.group(1)) if ic_match else float("nan")

    r2 = subprocess.run(
        [sys.executable, "score_submission.py", "_tmp_5d.csv",
         "--start", start, "--end", end],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if r2.returncode != 0:
        print(f"  {label}: score 失败\n{r2.stdout[:200]}")
        continue

    er_match = re.search(r"excess return\s*:\s*([+\-]?[\d.]+)%", r2.stdout)
    td_match = re.search(r"\((\d+) trading days?\)", r2.stdout)
    er = float(er_match.group(1)) if er_match else float("nan")
    td = int(td_match.group(1)) if td_match else 0

    results.append((label, as_of, f"{start}→{end}", ic, er, td))
    print(f"  {label}  as-of={as_of}  IC={ic:.4f}  ExRet={er:+.3f}%  ({td}d)")

tmp = ROOT / "_tmp_5d.csv"
if tmp.exists():
    tmp.unlink()

if not results:
    print("没有成功的窗口。")
    sys.exit(1)

print(f"\n{'='*65}")
print(f"{'label':<12} {'as-of':<10} {'window':<20} {'valIC':>7} {'ExRet':>8} {'d':>3}")
print("-" * 65)
for label, as_of, win, ic, er, td in results:
    print(f"{label:<12} {as_of:<10} {win:<20} {ic:>7.4f} {er:>+7.3f}% {td:>3}d")

valid = [(er, td) for *_, er, td in results if td >= 4]
if valid:
    avg_er = sum(er for er, _ in valid) / len(valid)
    pos = sum(1 for er, _ in valid if er > 0)
    print("-" * 65)
    print(f"  5d窗口统计（≥4天）: avg ExRet={avg_er:+.3f}%  正收益={pos}/{len(valid)}")
