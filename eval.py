"""
Dual evaluator for Sub2 tuning.

PRIMARY metric  : avg ExRet over 3 complete 5-day April windows (Apr-W1/W2/W3)
SECONDARY check : walk-forward CV on training data (Aug-Jan)
                  → if April good but walk CV crashes (< -0.02), suspect overfit

Results auto-logged to eval_log.csv.

Usage examples:
  python eval.py --note "baseline" --model lgbm --ensemble
  python eval.py --note "col0.5"   --model lgbm --ensemble --colsample-bytree 0.5
  python eval.py --note "+ret120d" --model lgbm --ensemble --extra-features ret_120d
  python eval.py --note "winner"   --model lgbm --ensemble --colsample-bytree 0.5 --reg-lambda 5.0 --skip-walk-cv
"""
import argparse, subprocess, sys, re, csv
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
LOG  = ROOT / "eval_log.csv"

# 5d April windows (3 complete + Sub1 reference)
APRIL_WINDOWS = [
    ("Apr-W1",   "20260407", "20260408", "20260414"),
    ("Apr-W2",   "20260414", "20260415", "20260421"),
    ("Apr-W3",   "20260421", "20260422", "20260428"),
    ("Sub1-ref", "20260430", "20260506", "20260508"),  # Sub1 actual eval window (3d)
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
    p.add_argument("--no-fundamentals",  action="store_true")
    p.add_argument("--skip-walk-cv",     action="store_true")
    # ── Submission tuning flags（仅传给 baseline_xgboost.py，不进 walk CV）─────
    p.add_argument("--train-min-date",   default=None,
                   help="训练集最早日期 YYYYMMDD（仅 April batch；walk CV 不受影响）")
    p.add_argument("--top-k",            type=int, default=None,
                   help="组合股票数（默认 50）")
    p.add_argument("--weight-scheme",    default=None, choices=[None, "rank", "exp"])
    p.add_argument("--exp-alpha",        type=float, default=None)
    args = p.parse_args()

    common = [
        "--model", args.model,
        "--num-leaves",       str(args.num_leaves),
        "--reg-lambda",       str(args.reg_lambda),
        "--subsample",        str(args.subsample),
        "--colsample-bytree", str(args.colsample_bytree),
        "--min-child-weight", str(args.min_child_weight),
    ]
    if args.extra_features:
        common += ["--extra-features", args.extra_features]
    if args.train_min_date:
        common += ["--train-min-date", args.train_min_date]

    # April batch 用的参数 = walk CV 的参数 + 组合/数据相关 flag
    top_k = args.top_k if args.top_k is not None else 50
    weight_scheme = args.weight_scheme if args.weight_scheme is not None else "rank"
    apr_args = common + ["--top-k", str(top_k), "--weight-scheme", weight_scheme]
    if args.ensemble:
        apr_args.append("--ensemble")
    if args.exp_alpha is not None:
        apr_args += ["--exp-alpha", str(args.exp_alpha)]
    # train-min-date 已在 common 里，会被 baseline_xgboost.py 接住

    print(f"\n{'='*72}")
    print(f"Eval: {args.note}")
    print(f"  model={args.model}  nl={args.num_leaves}  lam={args.reg_lambda}  "
          f"sub={args.subsample}  col={args.colsample_bytree}  mcs={args.min_child_weight}")
    print(f"  extra='{args.extra_features or '(none)'}'  ensemble={args.ensemble}")
    print(f"{'='*72}")

    # ── Primary: April 5d batch backtest ──────────────────────────────────────
    print("\n>> April 5-day windows (PRIMARY metric):")
    apr_rows = []
    for label, asof, start, end in APRIL_WINDOWS:
        r = subprocess.run(
            [sys.executable, "baseline_xgboost.py"] + apr_args
            + ["--as-of", asof, "--out", "_tmp_eval.csv"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            err = (r.stderr or r.stdout).strip().splitlines()[-1][:80]
            print(f"  {label}  FAILED ({err})")
            continue
        ic_match = re.search(r"validation rank IC:\s*([\-\d.]+)", r.stdout)
        ic = float(ic_match.group(1)) if ic_match else float("nan")

        s = subprocess.run(
            [sys.executable, "score_submission.py", "_tmp_eval.csv",
             "--start", start, "--end", end],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT),
        )
        er_match = re.search(r"excess return\s*:\s*([+\-]?[\d.]+)%", s.stdout)
        td_match = re.search(r"\((\d+) trading days?\)", s.stdout)
        er = float(er_match.group(1)) if er_match else float("nan")
        td = int(td_match.group(1)) if td_match else 0
        apr_rows.append((label, ic, er, td))
        print(f"  {label:<10}  IC={ic:>+7.4f}  ExRet={er:>+7.3f}%  ({td}d)")

    # Aggregate (5d windows only — Sub1-ref is 3d, listed for reference but excluded from avg)
    full = [r for r in apr_rows if r[3] >= 4]
    avg_er = sum(r[2] for r in full) / len(full) if full else float("nan")
    avg_ic = sum(r[1] for r in full) / len(full) if full else float("nan")
    pos    = sum(1 for r in full if r[2] > 0)
    print(f"\n  >> 5d-only AVG ExRet = {avg_er:+.3f}%  positive={pos}/{len(full)}  avg IC={avg_ic:.4f}")

    # ── Secondary: walk-forward CV on training data ───────────────────────────
    wcv_ic, wcv_er = float("nan"), float("nan")
    if not args.skip_walk_cv:
        print("\n>> Walk-forward CV (overfit check, Aug-Jan training windows):")
        wcv_cmd = [sys.executable, "walk_forward_cv.py"] + common + [
            "--no-fund-flow",  # northbound only covers 31 stocks → kills training
            "--note", f"eval-{args.note}",
        ]
        if args.no_fundamentals:
            wcv_cmd.append("--no-fundamentals")
        r = subprocess.run(wcv_cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(ROOT))
        ic_m = re.search(r"mean IC\s*=\s*([\-\d.]+)", r.stdout)
        er_m = re.search(r"mean ExRet\s*=\s*([+\-]?[\d.]+)%", r.stdout)
        if ic_m: wcv_ic = float(ic_m.group(1))
        if er_m: wcv_er = float(er_m.group(1))
        print(f"  >> walk CV mean IC = {wcv_ic:+.4f}  mean ExRet = {wcv_er:+.3f}%")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n>> VERDICT [{args.note}]")
    print(f"   April 5d  : avg ExRet={avg_er:+.3f}%  pos={pos}/{len(full)}")
    if not args.skip_walk_cv:
        print(f"   walk CV   : IC={wcv_ic:+.4f}")
        if avg_er > 0 and wcv_ic > -0.02:
            print(f"   STATUS    : OK (April positive + walk CV stable)")
        elif avg_er > 0 and wcv_ic <= -0.02:
            print(f"   STATUS    : WARN (April good but walk CV crashed → overfit risk)")
        else:
            print(f"   STATUS    : POOR (April not positive)")

    # ── Log ───────────────────────────────────────────────────────────────────
    is_new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["timestamp", "note", "model", "ensemble", "nl", "lam", "sub", "col",
                        "mcs", "extra", "apr_avg_exret", "apr_pos", "apr_total",
                        "apr_avg_ic", "wcv_ic", "wcv_exret"])
        w.writerow([datetime.now().strftime("%H:%M:%S"), args.note, args.model,
                    args.ensemble, args.num_leaves, args.reg_lambda, args.subsample,
                    args.colsample_bytree, args.min_child_weight, args.extra_features,
                    f"{avg_er:+.3f}", pos, len(full), f"{avg_ic:+.4f}",
                    f"{wcv_ic:+.4f}", f"{wcv_er:+.3f}"])
    print(f"   Logged to {LOG.name}")

    tmp = ROOT / "_tmp_eval.csv"
    if tmp.exists():
        tmp.unlink()


if __name__ == "__main__":
    main()
