"""Rank possible label errors using strictly out-of-fold predictions.

Nothing is edited automatically.  The tool exports a CSV/JSON queue and crop
panels for human review.  By default it looks for the two long-range
Al3Ni/Eutectic-Si confusions identified in the existing error analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from group_cv import mask_path
from komap import CLASSES, COLORS, rgb_to_index
from postprocess import si_particle_vote


def parse_pair(value: str):
    try:
        truth, predicted, distance = value.split(":")
        truth, predicted, distance = int(truth), int(predicted), float(distance)
    except ValueError as error:
        raise argparse.ArgumentTypeError("pair must be TRUE:PREDICTED:MIN_DISTANCE") from error
    if truth == predicted or not (0 <= truth < len(CLASSES)) or not (0 <= predicted < len(CLASSES)):
        raise argparse.ArgumentTypeError("class indices must be distinct values from 0 to 3")
    return truth, predicted, distance


def probability_files(directories):
    grouped = defaultdict(list)
    for directory in directories:
        for path in Path(directory).rglob("*_prob.npz"):
            name = path.name.replace("_prob.npz", "_image.png")
            grouped[name].append(path)
    if not grouped:
        raise FileNotFoundError("no *_prob.npz files under the supplied OOF directories")
    return grouped


def read_oof(paths):
    probabilities, images = [], []
    for path in sorted(paths):
        with np.load(path) as value:
            probabilities.append(value["prob"].astype(np.float32))
            image = value["image"] if "image" in value.files else None
            images.append(str(image.item()) if image is not None else "")
    shape = probabilities[0].shape
    if any(probability.shape != shape for probability in probabilities):
        raise ValueError(f"OOF probability shapes disagree: {paths}")
    known = {image for image in images if image}
    if len(known) != 1:
        raise ValueError(f"OOF files do not identify one source image: {paths}")
    return np.stack(probabilities), Path(known.pop())


def overlay(image, labels):
    colour = COLORS[labels]
    return cv2.addWeighted(image, 0.55, colour, 0.45, 0)


def panel(image, gt, pred, region, bbox, padding=48):
    x, y, w, h = bbox
    y0, y1 = max(0, y - padding), min(image.shape[0], y + h + padding)
    x0, x1 = max(0, x - padding), min(image.shape[1], x + w + padding)
    views = [image.copy(), overlay(image, gt), overlay(image, pred), image.copy()]
    views[-1][region] = (255, 40, 40)
    cv2.rectangle(views[-1], (x, y), (x + w - 1, y + h - 1), (255, 255, 0), 2)
    labels = ("image", "ground truth", "OOF prediction", "review region")
    crops = []
    for value, label in zip(views, labels):
        crop = value[y0:y1, x0:x1].copy()
        cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 24), (0, 0, 0), -1)
        cv2.putText(crop, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(crop)
    return np.concatenate(crops, axis=1)


def find_regions(probabilities, image_path, pairs, confidence, margin, agreement, min_area):
    mean = probabilities.mean(0)
    pred = si_particle_vote(mean)
    raw_votes = probabilities.argmax(1)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gt_path = mask_path(image_path)
    gt_bgr = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
    if gt_bgr is None:
        raise FileNotFoundError(gt_path)
    gt = rgb_to_index(cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB))
    if mean.shape[1:] != gt.shape:
        raise ValueError(f"probability/GT shape mismatch for {image_path}: {mean.shape[1:]} vs {gt.shape}")

    rows = []
    for true_class, predicted_class, min_distance in pairs:
        predicted_confidence = mean[predicted_class]
        probability_margin = predicted_confidence - mean[true_class]
        vote_agreement = (raw_votes == predicted_class).mean(0)
        if np.any(gt == predicted_class):
            to_predicted_gt = cv2.distanceTransform(
                (gt != predicted_class).astype(np.uint8), cv2.DIST_L2, 5)
        else:
            # OpenCV returns FLT_MAX when the binary image has no zero pixel.
            # "Farther than this image's diagonal" is the finite equivalent.
            diagonal = float(math.hypot(*gt.shape))
            to_predicted_gt = np.full(gt.shape, diagonal, np.float32)
        candidate = ((gt == true_class) & (pred == predicted_class)
                     & (predicted_confidence >= confidence) & (probability_margin >= margin)
                     & (vote_agreement >= agreement) & (to_predicted_gt >= min_distance))
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            candidate.astype(np.uint8), connectivity=8)
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            region = components == component
            bbox = tuple(int(stats[component, item]) for item in
                         (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                          cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            values = {"mean_confidence": float(predicted_confidence[region].mean()),
                      "mean_margin": float(probability_margin[region].mean()),
                      "mean_agreement": float(vote_agreement[region].mean()),
                      "mean_distance": float(to_predicted_gt[region].mean())}
            score = (math.sqrt(area) * values["mean_confidence"] * values["mean_margin"]
                     * values["mean_agreement"])
            rows.append({"image": image_path.name, "image_path": str(image_path),
                         "true_class": true_class, "true_name": CLASSES[true_class],
                         "predicted_class": predicted_class,
                         "predicted_name": CLASSES[predicted_class],
                         "min_distance_rule": min_distance, "area": area,
                         "x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3],
                         "score": score, **values, "_region": region, "_bbox": bbox,
                         "_image": image, "_gt": gt, "_pred": pred})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof_dirs", nargs="+", required=True,
                        help="one or more roots containing predict.py --save_prob outputs")
    parser.add_argument("--pairs", nargs="+", type=parse_pair,
                        default=[(1, 2, 5.0), (2, 1, 2.0)],
                        help="TRUE:PREDICTED:MIN_DISTANCE; defaults to Al3Ni<->Eutectic Si")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--agreement", type=float, default=0.5,
                        help="share of supplied OOF models voting for the predicted class")
    parser.add_argument("--min_area", type=int, default=16)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--out", default="runs/group_cv/label_audit")
    args = parser.parse_args()
    for name in ("confidence", "agreement"):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f"--{name} must be between 0 and 1")

    grouped = probability_files(args.oof_dirs)
    rows = []
    for name, paths in sorted(grouped.items()):
        probabilities, image_path = read_oof(paths)
        if image_path.name != name:
            raise ValueError(f"probability filename/source mismatch: {name} vs {image_path}")
        found = find_regions(probabilities, image_path, args.pairs, args.confidence,
                             args.margin, args.agreement, args.min_area)
        for row in found:
            row["oof_models"] = len(paths)
        rows.extend(found)
    rows.sort(key=lambda row: (-row["score"], -row["area"], row["image"]))

    out = Path(args.out)
    panels = out / "panels"
    panels.mkdir(parents=True, exist_ok=True)
    public = []
    for rank, row in enumerate(rows[:args.top], 1):
        name = (f"{rank:04d}_{Path(row['image']).stem}_{row['true_name']}-to-"
                f"{row['predicted_name']}.png")
        value = panel(row["_image"], row["_gt"], row["_pred"], row["_region"], row["_bbox"])
        cv2.imwrite(str(panels / name), cv2.cvtColor(value, cv2.COLOR_RGB2BGR))
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        clean.update({"rank": rank, "panel": str(panels / name), "decision": "",
                      "review_note": ""})
        public.append(clean)

    fields = (["rank", "image", "true_name", "predicted_name", "area", "score",
               "mean_confidence", "mean_margin", "mean_agreement", "mean_distance",
               "x", "y", "width", "height", "oof_models", "panel", "decision",
               "review_note"] if public else [])
    with (out / "audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if fields:
            writer.writeheader()
            writer.writerows(public)
    payload = {"settings": {"oof_dirs": args.oof_dirs, "pairs": args.pairs,
                            "confidence": args.confidence, "margin": args.margin,
                            "agreement": args.agreement, "min_area": args.min_area},
               "images_with_oof": len(grouped), "regions_found": len(rows),
               "regions_exported": len(public), "regions": public}
    (out / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"OOF images: {len(grouped)}  candidate regions: {len(rows)}  exported: {len(public)}")
    print(f"review queue: {out / 'audit.csv'}")
    print("No labels were modified. Fill decision/review_note only after inspecting each panel.")


if __name__ == "__main__":
    main()
