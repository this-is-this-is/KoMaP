"""Compare augmentation ensembles on the original 20 held-out scans."""
import glob
import json
import os

import cv2
import numpy as np


PALETTE = np.array([[153, 127, 76], [25, 76, 153],
                    [204, 204, 204], [76, 178, 76]], dtype=np.uint8)


def read(path):
    with open(path) as f:
        value = json.load(f)
    return {row["name"]: row for row in value["per_image"]}


def false_bridges(name):
    """Predicted Si E objects that join two or more distinct GT Si E objects."""
    directory = f"runs/a_ensemble_{name}_valid"
    total = 0
    for path in glob.glob(os.path.join(directory, "*_pred.png")):
        base = os.path.basename(path).replace("_pred.png", "_mask.png")
        gt_path = os.path.join("valid/valid/masks", base)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        pred = cv2.imread(path, cv2.IMREAD_COLOR)
        if gt is None or pred is None:
            continue
        gt_si = np.all(gt == PALETTE[2, ::-1], axis=2).astype(np.uint8)
        pred_si = np.all(pred == PALETTE[2, ::-1], axis=2).astype(np.uint8)
        _, gt_components = cv2.connectedComponents(gt_si, 8)
        count, predicted_components = cv2.connectedComponents(pred_si, 8)
        for idx in range(1, count):
            ids, overlap = np.unique(gt_components[predicted_components == idx],
                                     return_counts=True)
            total += np.count_nonzero((ids != 0) & (overlap >= 2)) >= 2
    return total


def main():
    base_path = "runs/a_ensemble_baseline_valid/scores.json"
    if not os.path.isfile(base_path):
        raise SystemExit(f"Missing {base_path}; run MODE=eval ./run_augmentation_suite.sh")
    base = read(base_path)
    base_mean = np.mean([row["miou"] for row in base.values()])
    print(f"baseline (aug_color x3): {100 * base_mean:.2f}%  ({len(base)} scans), "
          f"Si E false bridges: {false_bridges('baseline')}")
    print(f"{'method':20s} {'mIoU':>7s} {'delta':>7s} {'improved':>9s} {'Si E':>7s} {'bridges':>8s}")
    for path in sorted(glob.glob("runs/a_ensemble_*_valid/scores.json")):
        name = path.split("a_ensemble_", 1)[1].split("_valid/", 1)[0]
        if name == "baseline":
            continue
        rows = read(path)
        if set(rows) != set(base):
            print(f"{name:20s} skipped: scan set differs")
            continue
        scores = np.array([rows[k]["miou"] for k in base])
        differences = np.array([rows[k]["miou"] - base[k]["miou"] for k in base])
        si = np.nanmean([np.nan if rows[k]["EutecticSi"] is None
                         else rows[k]["EutecticSi"] for k in base])
        print(f"{name:20s} {100 * scores.mean():7.2f} {100 * differences.mean():+7.2f} "
              f"{np.sum(differences > 0):>4d}/{len(base):<4d} {100 * si:7.1f} "
              f"{false_bridges(name):8d}")


if __name__ == "__main__":
    main()
