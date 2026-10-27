"""Evaluate a small, predeclared multiscale/tile inference grid.

This deliberately avoids searching arbitrary weights on the 20 validation
scans.  Each scale set averages probabilities inside every checkpoint before
the model ensemble and Si-particle vote are applied by ``predict.py``.

Example:
    ./.conda/bin/python evaluate_multiscale.py --ckpt runs/a/best.pth runs/b/best.pth \
        --scale_sets 1 0.875,1,1.125 --tiles 768 896
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_scale_set(value: str) -> list[float]:
    scales = [float(item) for item in value.split(",") if item]
    if not scales or any(item <= 0 for item in scales):
        raise argparse.ArgumentTypeError("scale sets are comma-separated positive numbers")
    return scales


def slug(scales: list[float]) -> str:
    return "-".join(f"{value:g}".replace(".", "p") for value in scales)


def paired_interval(candidate: dict, reference: dict, repeats=10000, seed=0):
    ref = {row["name"]: row["miou"] for row in reference["per_image"]}
    new = {row["name"]: row["miou"] for row in candidate["per_image"]}
    if set(ref) != set(new):
        raise ValueError("multiscale configurations scored different images")
    names = sorted(ref)
    difference = np.array([new[name] - ref[name] for name in names])
    sample = np.random.default_rng(seed).integers(0, len(names), (repeats, len(names)))
    low, high = 100 * np.percentile(difference[sample].mean(1), [2.5, 97.5])
    return 100 * float(difference.mean()), [float(low), float(high)], \
        [int(np.sum(difference > 1e-6)), int(np.sum(difference < -1e-6))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", nargs="+", required=True)
    parser.add_argument("--data_root", default=str(ROOT))
    parser.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--scale_sets", nargs="+", type=parse_scale_set,
                        default=[[1.0], [0.875, 1.0, 1.125]],
                        help="space-separated sets; values inside a set use commas")
    parser.add_argument("--tiles", nargs="+", type=int, default=[768])
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out", default="runs/multiscale_eval")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if any(tile <= 0 for tile in args.tiles):
        parser.error("tiles must be positive")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for tile in args.tiles:
        for scales in args.scale_sets:
            name = f"tile{tile}_scale-{slug(scales)}"
            destination = out / f"{name}.json"
            if args.force or not destination.is_file():
                command = [sys.executable, str(ROOT / "predict.py"), "--ckpt", *args.ckpt,
                           "--data_root", args.data_root, "--split", args.split,
                           "--tile", str(tile), "--overlap", str(args.overlap),
                           "--tta", "--d4", "--si_vote", "--scale_tta",
                           *map(str, scales), "--scores_out", str(destination)]
                if args.manifest:
                    command += ["--manifest", args.manifest]
                print("run:", " ".join(command), flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
            value = json.loads(destination.read_text())
            value.update({"name": name, "tile": tile, "scale_tta": scales,
                          "scores_file": str(destination)})
            results.append(value)

    reference = results[0]
    rows = []
    for value in results:
        delta, interval, counts = paired_interval(value, reference)
        rows.append({"name": value["name"], "miou": 100 * value["mean_miou"],
                     "delta_vs_first_pp": delta, "interval95_pp": interval,
                     "images_better_worse": counts, "scores_file": value["scores_file"]})
    rows.sort(key=lambda row: -row["miou"])
    summary = {"reference": reference["name"], "checkpoints": args.ckpt, "results": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\n{'configuration':36s}  mIoU   vs first      paired bootstrap 95%   img +/-")
    for row in rows:
        lo, hi = row["interval95_pp"]
        better, worse = row["images_better_worse"]
        print(f"{row['name']:36s} {row['miou']:6.2f}  {row['delta_vs_first_pp']:+7.2f}pp"
              f"   [{lo:+.2f}, {hi:+.2f}]       {better:2d}/{worse:2d}")
    print(f"wrote {out / 'summary.json'}")


if __name__ == "__main__":
    main()
