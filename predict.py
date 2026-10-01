"""Score checkpoints and write predictions, using overlapping-tile inference.

The reported number is the competition's: per-image 4-class mIoU, averaged over
the images. Several checkpoints can be given at once to ensemble them -- their
class probabilities are averaged before the argmax.

Examples:
    ./.conda/bin/python predict.py --ckpt runs/s2_micronet_unetpp/best.pth --split valid --tta
    ./.conda/bin/python predict.py --ckpt runs/a/best.pth runs/b/best.pth \
        --split test --tta --save_dir preds/ensemble
    # best no-retraining setting on valid (78.1% for s2+s4): 8-way TTA, 768 tiles, Si vote
    ./.conda/bin/python predict.py --ckpt runs/a/best.pth runs/b/best.pth \
        --split valid --tta --d4 --tile 768 --si_vote --breakdown
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from diagnostics import error_breakdown, summarise
from komap import CLASSES, NUM_CLASSES, index_to_rgb, per_image_miou, rgb_to_index
from models import build
from postprocess import si_particle_vote
from tiling import predict_tiled, scaled_tile

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
SPLIT_DIRS = {"train": "train/train", "valid": "valid/valid", "test": "test/test"}


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    a = ckpt.get("args", {})
    model = build(a.get("arch", "unetpp"), a.get("encoder", "efficientnet-b3"),
                  # Weights come from the checkpoint, so skip the pretrained
                  # download entirely.
                  pretrain="none",
                  fullres_skip=a.get("fullres_skip", False), fullres_ch=a.get("fullres_ch", 32))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints to ensemble")
    ap.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--split", default="valid", choices=sorted(SPLIT_DIRS))
    ap.add_argument("--tile", type=int, default=None,
                    help="tile in native pixels (models trained with --input_scale enlarge it "
                         "by their scale); defaults to the training tile")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    ap.add_argument("--d4", action="store_true",
                    help="also average transposed tiles; with --tta, all 8 rotations/reflections")
    ap.add_argument("--si_vote", action="store_true",
                    help="one label (Eutectic or Primary Si) per connected Si particle")
    ap.add_argument("--breakdown", action="store_true",
                    help="split errors into boundary shifts vs whole-structure mislabels")
    ap.add_argument("--save_dir", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, tiles, scales = [], [], []
    for c in args.ckpt:
        m, a = load_model(c, device)
        models.append(m)
        tiles.append(a.get("tile", 512))
        scales.append(a.get("input_scale", 1.0))
    tile = args.tile or max(tiles)

    split_dir = os.path.join(args.data_root, SPLIT_DIRS[args.split])
    has_mask = os.path.isdir(os.path.join(split_dir, "masks"))
    import glob
    paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"checkpoints: {len(models)}  split: {args.split} ({len(paths)} images)")
    print(f"tile={tile} overlap={args.overlap} tta={args.tta} d4={args.d4} "
          f"si_vote={args.si_vote} ground truth: {'yes' if has_mask else 'no'}")
    if any(s != 1 for s in scales):
        print("input scales: " + " ".join(f"{s:g}" for s in scales))
    print()

    scores, per_class, rows, breakdown = [], [], [], []
    for p in paths:
        name = os.path.basename(p)
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = ((img / 255.0 - np.array(MEAN)) / np.array(STD)).transpose(2, 0, 1)
        x = torch.from_numpy(x).float()[None].to(device)

        # Average probabilities across models, never labels: a majority vote on
        # argmax discards how confident each model was.
        prob = None
        for m, s in zip(models, scales):
            q = predict_tiled(m, x, tile=scaled_tile(tile, s), overlap=args.overlap,
                              num_classes=NUM_CLASSES, flip_tta=args.tta, transpose_tta=args.d4,
                              scale=s)
            prob = q if prob is None else prob + q
        prob = prob / len(models)
        if args.si_vote:
            pred = si_particle_vote(prob)
        else:
            pred = prob.argmax(0).cpu().numpy().astype(np.uint8)

        if has_mask:
            mp = os.path.join(split_dir, "masks", name.replace("_image.png", "_mask.png"))
            gt = rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            miou, ious = per_image_miou(pred, gt)
            if args.breakdown:
                breakdown.append(error_breakdown(pred, gt))
            scores.append(miou)
            per_class.append(ious)
            rows.append({"name": name, "miou": miou,
                         **{c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, ious)}})
            print(f"  {name:42s} mIoU {100*miou:6.2f}%   " + "  ".join(
                f"{c}={'  n/a' if np.isnan(v) else f'{100*v:5.1f}%'}"
                for c, v in zip(CLASSES, ious)))

        if args.save_dir:
            cv2.imwrite(os.path.join(args.save_dir, name.replace("_image.png", "_pred.png")),
                        cv2.cvtColor(index_to_rgb(pred), cv2.COLOR_RGB2BGR))

    if has_mask:
        mean_miou = float(np.mean(scores))
        pc = np.nanmean(np.stack(per_class), 0)
        print("\n" + "=" * 74)
        print(f"  mean mIoU over {len(scores)} images : {100*mean_miou:.2f}%")
        print("  per-class IoU                : " + "  ".join(
            f"{c}={100*v:.1f}%" for c, v in zip(CLASSES, pc)))
        print("  competition reference        : ~75%")
        if breakdown:
            summarise(breakdown, scores)
        print("=" * 74)
        if args.save_dir:
            with open(os.path.join(args.save_dir, "scores.json"), "w") as f:
                json.dump({"mean_miou": mean_miou, "per_image": rows}, f, indent=2)
    if args.save_dir:
        print(f"\npredictions -> {args.save_dir}/")


if __name__ == "__main__":
    main()
