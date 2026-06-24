"""
批量回测：对多个 as-of 日期生成预测，用 score_submission.py 计算实际超额收益。
用法：python batch_backtest.py

日期说明（2026年4月，April 1=周三）：
  Apr-24(Fri), Apr-22(Wed), Apr-17(Fri), Apr-10(Fri) 均为交易日
  Apr-25/Apr-11 是周六，已跳过
"""
import subprocess, sys, re
from pathlib import Path

ROOT = Path(__file__).parent

# (标签, as_of, score_start, score_end)
# 确保每窗口恰好包含 3 个交易日
WINDOWS = [
    # ── 4月下旬 ────────────────────────────────────────────────────────────
    ("Apr-24", "20260424", "20260427", "20260429"),  # Fri → Mon/Tue/Wed (3d)
    ("Apr-22", "20260422", "20260423", "20260427"),  # Wed → Thu/Fri/Mon (3d)
    ("Apr-17", "20260417", "20260420", "20260422"),  # Fri → Mon/Tue/Wed (3d)
    # ── 4月中旬（解放日）──────────────────────────────────────────────────
    ("Apr-10", "20260410", "20260413", "20260415"),  # Fri → Mon/Tue/Wed (3d)
    ("Apr-09", "20260409", "20260410", "20260414"),  # Thu → Fri/Mon/Tue (3d)
    ("Apr-07", "20260407", "20260408", "20260410"),  # Tue → Wed/Thu/Fri (3d)
    # ── 4月上旬（清明前后）──────────────────────────────────────────────
    ("Apr-03", "20260403", "20260407", "20260409"),  # Fri → Tue/Wed/Thu 节后(3d)
    ("Apr-02", "20260402", "20260403", "20260407"),  # Thu → Fri/Tue(节后)(2~3d)
    # ── 3月（平静期）────────────────────────────────────────────────────
    ("Mar-31", "20260331", "20260401", "20260403"),  # Tue → Wed/Thu/Fri (3d)
    ("Mar-26", "20260326", "20260327", "20260331"),  # Thu → Fri/Mon/Tue (3d)
    ("Mar-24", "20260324", "20260325", "20260327"),  # Tue → Wed/Thu/Fri (3d)
    ("Mar-19", "20260319", "20260320", "20260324"),  # Thu → Fri/Mon/Tue (3d)
    ("Mar-12", "20260312", "20260313", "20260317"),  # Thu → Fri/Mon/Tue (3d)
    # ── 2月末～3月初（同学脚本对齐窗口）────────────────────────────────────
    ("Mar-06", "20260306", "20260309", "20260311"),  # Fri → Mon/Tue/Wed (3d)
    ("Feb-27", "20260227", "20260302", "20260304"),  # Fri → Mon/Tue/Wed (3d)
]

results = []
for label, as_of, start, end in WINDOWS:
    r1 = subprocess.run(
        [sys.executable, "baseline_xgboost.py", "--model", "lgbm",
         "--as-of", as_of, "--out", "_tmp_batch.csv"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if r1.returncode != 0:
        short_err = (r1.stderr or r1.stdout).strip().splitlines()[-1][:80]
        print(f"  {label}: 跳过（{short_err}）")
        continue

    ic_match = re.search(r"validation rank IC:\s*([\-\d.]+)", r1.stdout)
    ic = float(ic_match.group(1)) if ic_match else float("nan")

    r2 = subprocess.run(
        [sys.executable, "score_submission.py", "_tmp_batch.csv",
         "--start", start, "--end", end],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT),
    )
    if r2.returncode != 0:
        print(f"  {label}: score 失败")
        continue

    er_match = re.search(r"excess return\s*:\s*([+\-][\d.]+)%", r2.stdout)
    td_match = re.search(r"\((\d+) trading days\)", r2.stdout)
    er = float(er_match.group(1)) if er_match else float("nan")
    td = int(td_match.group(1)) if td_match else 0

    results.append((label, as_of, f"{start}->{end}", ic, er, td))
    flag = "+" if er > 0 else "-"
    print(f"  {label}  IC={ic:.4f}  ExRet={er:+.3f}%  ({td}d) {flag}")

tmp = ROOT / "_tmp_batch.csv"
if tmp.exists():
    tmp.unlink()

if not results:
    print("No successful windows.")
    sys.exit(1)

print(f"\n{'='*68}")
print(f"{'label':<10} {'as-of':<10} {'score_window':<22} {'valIC':>7} {'ExRet':>8} {'d':>3}")
print("-" * 68)
for label, as_of, win, ic, er, td in results:
    flag = "+" if er > 0 else "-"
    print(f"{label:<10} {as_of:<10} {win:<22} {ic:>7.4f} {er:>+7.3f}% {flag} {td:>2}d")

pos    = sum(1 for *_, er, _ in results if er > 0)
avg_er = sum(er for *_, er, _ in results) / len(results)
avg_ic = sum(ic for _, _, _, ic, _, _ in results) / len(results)
print("-" * 68)
print(f"  positive: {pos}/{len(results)}    avg ExRet: {avg_er:+.3f}%    avg valIC: {avg_ic:.4f}")
