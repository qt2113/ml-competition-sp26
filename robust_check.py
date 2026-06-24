"""
Robust overfitting check (per teacher's advice):
For a fixed config, run rolling 5-day windows across Feb-Apr 2026.
If config is positive across MOST windows → low overfit risk.

Usage:
  python robust_check.py --note "B2"      --model lgbm --ensemble --colsample-bytree 0.5 --reg-lambda 1.0
  python robust_check.py --note "Sub1"    --model lgbm --ensemble --colsample-bytree 0.7 --reg-lambda 0.1
  python robust_check.py --note "B2+F5"   --model lgbm --ensemble --colsample-bytree 0.5 --reg-lambda 1.0 --extra-features "vol_5d,vol_ratio_5_60"
"""
import argparse, subprocess, sys, re, csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
LOG  = ROOT / "robust_log.csv"

# ── 11 clean 5-day rolling windows + 1 Sub1 ref ────────────────────────────
# 已避开春节（2/14-2/23）、清明（4/4-4/6）非交易日
# 各窗口 score period 包含 5 个交易日，as_of 都是真实交易日
WINDOWS = [
    # Feb 节前
    ("Feb-W1",      "20260202", "20260203", "20260209"),  # 3,4,5,6,9
    # Feb 节后 + March
    ("Feb-W2",      "20260224", "20260225", "20260303"),  # 25,26,27,2,3
    ("Mar-W1",      "20260303", "20260304", "20260310"),
    ("Mar-W2",      "20260310", "20260311", "20260317"),
    ("Mar-W3",      "20260317", "20260318", "20260324"),
    ("Mar-W4",      "20260324", "20260325", "20260331"),
    # March-April 跨清明
    ("Apr-W0",      "20260331", "20260401", "20260408"),  # 1,2,3,7,8
    # April clean
    ("Apr-W1",      "20260408", "20260409", "20260415"),
    ("Apr-W2",      "20260415", "20260416", "20260422"),
    ("Apr-W3",      "20260422", "20260423", "20260429"),
    # Sub1 实盘窗口（3 天，仅参考）
    ("May-Sub1",    "20260430", "20260506", "20260508"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--note", required=True)
    p.add_argument("--model", default="lgbm", choices=["lgbm", "xgb"])
    p.add_argument("--num-leaves",       type=int,   default=8)
    p.add_argument("--reg-lambda",       type=float, default=0.1)
    p.add_argument("--subsample",        type=float, default=0.6)
    p.add_argument("--colsample-bytree", type=float, default=0.7)
    p.add_argument("--min-child-weight", type=int,   default=20)
    p.add_argument("--extra-features",   default="")
    p.add_argument("--ensemble",         action="store_true")
    p.add_argument("--train-min-date",   default=None,
                   help="训练集最早日期 YYYYMMDD")
    p.add_argument("--top-k",            type=int, default=50)
    p.add_argument("--weight-scheme",    default="rank", choices=["rank", "exp"])
    p.add_argument("--exp-alpha",        type=float, default=None)
    args = p.parse_args()

    common = [
        "--model", args.model,
        "--num-leaves",       str(args.num_leaves),
        "--reg-lambda",       str(args.reg_lambda),
        "--subsample",        str(args.subsample),
        "--colsample-bytree", str(args.colsample_bytree),
        "--min-child-weight", str(args.min_child_weight),
        "--top-k", str(args.top_k),
        "--weight-scheme", args.weight_scheme,
    ]
    if args.extra_features:
        common += ["--extra-features", args.extra_features]
    if args.ensemble:
        common.append("--ensemble")
    if args.exp_alpha is not None:
        common += ["--exp-alpha", str(args.exp_alpha)]
    if args.train_min_date:
        common += ["--train-min-date", args.train_min_date]

    print(f"\n{'='*72}")
    print(f"Robust check: {args.note}")
    print(f"  model={args.model}  nl={args.num_leaves}  lam={args.reg_lambda}  "
          f"col={args.colsample_bytree}  ensemble={args.ensemble}")
    print(f"  top_k={args.top_k}  weight={args.weight_scheme}  "
          f"alpha={args.exp_alpha}  train_min={args.train_min_date}")
    print(f"  extra='{args.extra_features or '(none)'}'")
    print(f"{'='*72}\n")

    print(f"{'label':<14} {'as_of':<10} {'window':<22} {'IC':>8} {'ExRet':>9} {'days':>5}")
    print("-" * 72)

    rows = []
    for label, asof, start, end in WINDOWS:
        r1 = subprocess.run(
            [sys.executable, "baseline_xgboost.py"] + common
            + ["--as-of", asof, "--out", "_tmp_robust.csv"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        if r1.returncode != 0:
            err = (r1.stderr or r1.stdout).strip().splitlines()[-1][:60]
            print(f"  {label:<14} SKIP ({err})")
            continue
        ic_match = re.search(r"validation rank IC:\s*([\-\d.]+)", r1.stdout)
        ic = float(ic_match.group(1)) if ic_match else float("nan")

        r2 = subprocess.run(
            [sys.executable, "score_submission.py", "_tmp_robust.csv",
             "--start", start, "--end", end],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        er_match = re.search(r"excess return\s*:\s*([+\-]?[\d.]+)%", r2.stdout)
        td_match = re.search(r"\((\d+) trading days?\)", r2.stdout)
        er = float(er_match.group(1)) if er_match else float("nan")
        td = int(td_match.group(1)) if td_match else 0

        rows.append((label, asof, f"{start}->{end}", ic, er, td))
        marker = "+" if er > 0 else "-"
        print(f"  {label:<14} {asof:<10} {start}->{end}  {ic:>+7.4f} {er:>+7.3f}% {marker} {td:>3}d")

    if not rows:
        print("All windows failed.")
        return

    # ── 5-day windows only summary ───────────────────────────────────────────
    full = [r for r in rows if r[5] >= 4]
    pos5 = sum(1 for r in full if r[4] > 0)
    avg5 = sum(r[4] for r in full) / len(full) if full else float("nan")
    std5 = (sum((r[4] - avg5)**2 for r in full) / len(full))**0.5 if full else float("nan")

    # All windows (incl. partial)
    pos_all = sum(1 for r in rows if r[4] > 0)
    avg_all = sum(r[4] for r in rows) / len(rows)

    print("-" * 72)
    print(f"  5-day only   (n={len(full):>2}): avg ExRet={avg5:>+7.3f}%  std={std5:>5.3f}%  positive={pos5}/{len(full)}")
    print(f"  All windows  (n={len(rows):>2}): avg ExRet={avg_all:>+7.3f}%  positive={pos_all}/{len(rows)}")

    # Verdict
    win_rate = pos5 / len(full) if full else 0.0
    print(f"\n  win_rate (5d only) = {win_rate:.0%}")
    if win_rate >= 0.8 and avg5 > 0.5:
        verdict = "STRONG (low overfit risk per teacher)"
    elif win_rate >= 0.6 and avg5 > 0:
        verdict = "OK (acceptable but not stellar)"
    else:
        verdict = "WEAK (likely overfit to recent data)"
    print(f"  verdict = {verdict}")

    # Log
    is_new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            cols = ["timestamp", "note", "config", "n_windows", "n_5d", "pos_5d",
                    "avg_5d", "std_5d", "win_rate", "verdict"]
            for label, *_ in WINDOWS:
                cols.append(f"{label}_er")
            w.writerow(cols)
        cfg = (f"model={args.model} nl={args.num_leaves} lam={args.reg_lambda} "
               f"col={args.colsample_bytree} ensemble={args.ensemble} "
               f"extra={args.extra_features or '-'}")
        row_data = [datetime.now().strftime("%H:%M:%S"), args.note, cfg, len(rows),
                    len(full), pos5, f"{avg5:.3f}", f"{std5:.3f}",
                    f"{win_rate:.2f}", verdict]
        # Pad ER values per WINDOWS order (NaN for skipped)
        er_by_label = {r[0]: r[4] for r in rows}
        for label, *_ in WINDOWS:
            er = er_by_label.get(label, float("nan"))
            row_data.append(f"{er:+.3f}" if er == er else "")
        w.writerow(row_data)
    print(f"\n  Logged to {LOG.name}")

    tmp = ROOT / "_tmp_robust.csv"
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
