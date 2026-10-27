"""Is a "best checkpoint" gain real, or an artefact of picking on the validation scans?

train_v2.py keeps the checkpoint with the highest mIoU over the 20 validation
scans and reports that same number, so a run whose curve wobbles more gets a
better "best" by luck alone. Here the epoch is picked on one half of the scans
and scored on the other half, over many random halves; the held-out score has
no selection effect. Needs runs trained after per-image scores were added to
history.json.

    ./.conda/bin/python split_half_check.py b_ep60 b_ep120 b_ep200
    (a group is runs/<group>_s<seed>; the first group is the reference)
"""
import argparse
import glob
import json
import re

import numpy as np


def load(group):
    """(seeds x evals x images) mIoU arrays, one per seed, plus the eval epochs."""
    runs = sorted(p for p in glob.glob(f"runs/{group}_s*/history.json")
                  if re.search(rf"/{re.escape(group)}_s\d+/", p))
    out = []
    for path in runs:
        with open(path) as f:
            hist = json.load(f)["history"]
        if "per_image" not in hist[0]:
            raise SystemExit(f"{path} has no per-image scores; it was trained before they were logged")
        names = sorted(hist[0]["per_image"])
        out.append(np.array([[h["per_image"][n] for n in names] for h in hist]))
    if not out:
        raise SystemExit(f"no runs found for group {group}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="+")
    ap.add_argument("--splits", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n_images = load(args.groups[0])[0].shape[1]
    halves = [rng.permutation(n_images) for _ in range(args.splits)]
    pick = [h[:n_images // 2] for h in halves]
    hold = [h[n_images // 2:] for h in halves]

    results = {}
    print(f"{'group':16s} seeds  best-on-all  held-out  selection gap   final epoch")
    for g in args.groups:
        runs = load(g)
        best_all = np.mean([r.mean(1).max() for r in runs])
        final = np.mean([r[-1].mean() for r in runs])
        # held[k]: mean over seeds of the held-out score for split k
        held = np.mean([[r[r[:, p].mean(1).argmax(), h].mean() for p, h in zip(pick, hold)]
                        for r in runs], 0)
        # the same picked epoch scored on the half it was picked on
        seen = np.mean([[r[:, p].mean(1).max() for p in pick] for r in runs], 0)
        results[g] = held
        print(f"{g:16s} {len(runs):3d}    {100 * best_all:6.2f}     {100 * held.mean():6.2f}     "
              f"{100 * (seen.mean() - held.mean()):+5.2f}        {100 * final:6.2f}")

    ref = results[args.groups[0]]
    print(f"\nheld-out difference vs {args.groups[0]} (same {args.splits} random halves):")
    for g in args.groups[1:]:
        d = results[g] - ref
        print(f"  {g:16s} {100 * d.mean():+5.2f}pp   better in {100 * np.mean(d > 0):.0f}% of the splits")
    print("\nbest-on-all: what train_v2.py reports. held-out: epoch picked on 10 scans, scored on the "
          "other 10.\nA gain that shows in best-on-all but not in held-out came from the picking.")


if __name__ == "__main__":
    main()
