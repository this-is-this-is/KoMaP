"""Mean +- sd over seeds for every group of runs, against the ce_dice baseline.

A group is every run directory named <name>_s<seed> (run_boundary_suite.sh
writes runs/b_<method>_s<seed>). The baseline is plain ce_dice at seeds 0-2
with train_v2.py's defaults, like the suite. runs/s2_micronet_unetpp is left
out: it was trained with --enc_lr 1e-4, the suite with the default 3e-4.
Seed noise on this data is about +-0.15pp, so a single run's gain means
little: read the 3-seed mean and Welch's p.

    ./.conda/bin/python summarize_runs.py
    ./.conda/bin/python summarize_runs.py --pattern "runs/b_*_s*"
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats

CLASSES = ["Al", "Al3Ni", "EutecticSi", "PrimarySi"]
BASELINE = ["runs/seed_base_0", "runs/seed_base_1", "runs/seed_base_2"]


def load(run):
    """Best val mIoU and that checkpoint's per-class IoU (nan-mean over images)."""
    with open(os.path.join(run, "history.json")) as f:
        best = json.load(f)["best_miou"]
    with open(os.path.join(run, "val_per_image.json")) as f:
        rows = json.load(f)["per_image"]
    per_class = [np.nanmean([np.nan if r[c] is None else r[c] for r in rows]) for c in CLASSES]
    return best, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/b_*_s*")
    ap.add_argument("--baseline", nargs="+", default=BASELINE)
    args = ap.parse_args()

    groups = {}
    for run in sorted(glob.glob(args.pattern)):
        m = re.match(r"(.+)_s(\d+)$", os.path.basename(run))
        if m and os.path.exists(os.path.join(run, "history.json")):
            groups.setdefault(m.group(1), []).append(load(run))
    base = [load(r) for r in args.baseline if os.path.exists(os.path.join(r, "history.json"))]
    if not base:
        raise SystemExit("baseline runs not found")
    base_scores = np.array([b for b, _ in base])

    def row(name, runs, compare):
        s = np.array([b for b, _ in runs])
        pc = 100 * np.mean([p for _, p in runs], 0)
        sd = f"{100 * s.std(ddof=1):4.2f}" if len(s) > 1 else "   -"
        if compare and len(s) > 1:
            p = stats.ttest_ind(s, base_scores, equal_var=False).pvalue
            diff = f"{100 * (s.mean() - base_scores.mean()):+5.2f}  {p:4.2f}"
        elif compare:
            diff = f"{100 * (s.mean() - base_scores.mean()):+5.2f}     -"
        else:
            diff = "    -     -"
        return (f"{name:22s} {len(s):2d}  {100 * s.mean():6.2f}  {sd}  {diff}   "
                + "  ".join(f"{v:5.1f}" for v in pc))

    print(f"{'group':22s} {'n':>2s}  {'mean':>6s}  {'sd':>4s}  {'diff':>5s}  {'p':>4s}   "
          + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
    print(row("baseline (ce_dice)", base, compare=False))
    for name, runs in sorted(groups.items(), key=lambda kv: -np.mean([b for b, _ in kv[1]])):
        print(row(name, runs, compare=True) + ("" if len(runs) >= 3 else "   <- fewer than 3 seeds"))
    print("\nmean/sd: best val mIoU (%), per-image metric, tile 512, no TTA; "
          "diff vs baseline mean, p = Welch t-test")


if __name__ == "__main__":
    main()
