"""Evaluate microscopy-transfer models at the established KoMaP ensemble setting."""
import argparse
import glob
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from komap import CLASSES, NUM_CLASSES, per_image_miou, rgb_to_index
from postprocess import si_particle_vote
from predict import load_model
from preprocess import parse_scale_table, prepare, scan_scale
from tiling import predict_tiled, scaled_tile


def expand(specification):
    name, pattern = specification.split("=", 1)
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f"candidate {name} matched no checkpoints: {pattern}")
    return name, paths


def score(probabilities, names, masks):
    scores, per_class, per_image = [], [], {}
    for probability, name, target in zip(probabilities, names, masks):
        prediction = si_particle_vote(torch.from_numpy(probability))
        miou, classes = per_image_miou(prediction, target)
        scores.append(miou)
        per_class.append(classes)
        per_image[name] = 100 * miou
    return {"miou": 100 * float(np.mean(scores)),
            "class_iou": dict(zip(CLASSES, (100 * np.nanmean(per_class, axis=0)).tolist())),
            "per_image": per_image}


def aggregate(checkpoints, images, names, masks, tile, device):
    total, singles = [None] * len(images), {}
    for checkpoint in checkpoints:
        model, settings = load_model(checkpoint, device)
        scale = settings.get("input_scale", 1.0)
        table = parse_scale_table(settings.get("scale_by_height"))
        current = []
        for index, image in enumerate(images):
            value = predict_tiled(
                model, prepare(image, settings).to(device), tile=scaled_tile(tile, scale),
                overlap=0.5, num_classes=NUM_CLASSES, flip_tta=True, transpose_tta=True,
                scale=scan_scale(image.shape[0], scale, table)).float().cpu().numpy()
            current.append(value)
            total[index] = value if total[index] is None else total[index] + value
        singles[checkpoint] = score(current, names, masks)
        print(f"cached {checkpoint}: {singles[checkpoint]['miou']:.2f}%", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return [value / len(checkpoints) for value in total], singles


def blend(a, b, weight_b):
    return [(1 - weight_b) * x + weight_b * y for x, y in zip(a, b)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--candidate", action="append", default=[], metavar="NAME=GLOB")
    ap.add_argument("--tile", type=int, default=768)
    ap.add_argument("--data_root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    candidates = dict(expand(value) for value in args.candidate)
    if not candidates:
        ap.error("at least one --candidate NAME=GLOB is required")
    missing = [path for path in args.baseline if not Path(path).is_file()]
    if missing:
        ap.error(f"missing baseline checkpoints: {missing}")

    paths = sorted((args.data_root / "valid/valid/images").glob("*.png"))
    names = [path.name.replace("_image.png", "") for path in paths]
    images = [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]
    masks = []
    for name in names:
        mask_path = args.data_root / "valid/valid/masks" / f"{name}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        masks.append(rgb_to_index(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, baseline_singles = aggregate(args.baseline, images, names, masks, args.tile, device)
    result = {"setting": {"tta": "D4/8-way", "tile": args.tile, "si_vote": True},
              "baseline_checkpoints": args.baseline, "baseline_singles": baseline_singles,
              "baseline": score(baseline, names, masks), "candidates": {}}
    print(f"baseline x{len(args.baseline)}: {result['baseline']['miou']:.2f}%", flush=True)

    for name, checkpoints in candidates.items():
        probability, singles = aggregate(checkpoints, images, names, masks, args.tile, device)
        group = score(probability, names, masks)
        # Fixed weights only.  Equal-members is the literal 6+3 (or 6+6)
        # ensemble; half gives the transfer recipe and incumbent equal votes.
        member_weight = len(checkpoints) / (len(args.baseline) + len(checkpoints))
        equal_members = score(blend(baseline, probability, member_weight), names, masks)
        half = score(blend(baseline, probability, 0.5), names, masks)
        result["candidates"][name] = {
            "checkpoints": checkpoints, "singles": singles, "ensemble": group,
            "baseline_plus_equal_members": equal_members,
            "baseline_plus_half": half,
            "delta_equal_members_pp": equal_members["miou"] - result["baseline"]["miou"],
            "delta_half_pp": half["miou"] - result["baseline"]["miou"],
        }
        print(f"{name} x{len(checkpoints)} {group['miou']:.2f}% | baseline+equal-members "
              f"{equal_members['miou']:.2f}% ({equal_members['miou']-result['baseline']['miou']:+.2f}pp)",
              flush=True)

    winner = max(result["candidates"],
                 key=lambda key: result["candidates"][key]["delta_equal_members_pp"])
    result["winner"] = winner
    result["winner_delta_pp"] = result["candidates"][winner]["delta_equal_members_pp"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}; winner={winner} ({result['winner_delta_pp']:+.2f}pp)")


if __name__ == "__main__":
    main()
