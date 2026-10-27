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
from group_cv import mask_path, read_manifest
from komap import CLASSES, NUM_CLASSES, index_to_rgb, per_image_miou, rgb_to_index
from models import build
from postprocess import si_particle_vote
from preprocess import parse_scale_table, prepare, scan_scale
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
                  fullres_skip=a.get("fullres_skip", False), fullres_ch=a.get("fullres_ch", 32),
                  # The edge head is training-only but its weights are in the
                  # checkpoint, so the module has to exist for a strict load.
                  edge_head=(a.get("edge_w") or 0) > 0,
                  decoder_attention=a.get("decoder_attention"))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), a


def inference_views(tile, trained_scale, scan_height, scale_table, multipliers):
    """Tile/model scales for multiscale inference, preserving native field of view.

    A multiplier changes both the resized scan and the tile in model pixels.
    Multiplier 1 exactly reproduces the existing inference path, including the
    special small-scan scale used by ``cs2_plus``.
    """
    base = scan_scale(scan_height, trained_scale, scale_table)
    views = []
    for multiplier in multipliers:
        if multiplier <= 0:
            raise ValueError("scale_tta multipliers must be positive")
        views.append((scaled_tile(tile, trained_scale * multiplier), base * multiplier))
    return views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="one or more checkpoints to ensemble")
    ap.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--split", default="valid", choices=sorted(SPLIT_DIRS))
    ap.add_argument("--manifest", default=None,
                    help="JSON/text image manifest; overrides --split (used by grouped OOF folds)")
    ap.add_argument("--tile", type=int, default=None,
                    help="tile in native pixels (models trained with --input_scale enlarge it "
                         "by their scale); defaults to the training tile")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--tta", action="store_true", help="flip test-time augmentation")
    ap.add_argument("--d4", action="store_true",
                    help="also average transposed tiles; with --tta, all 8 rotations/reflections")
    ap.add_argument("--scale_tta", type=float, nargs="+", default=[1.0], metavar="MULT",
                    help="relative inference scales to probability-average, e.g. "
                         "--scale_tta 0.875 1 1.125; 1 exactly matches legacy inference")
    ap.add_argument("--si_vote", action="store_true",
                    help="one label (Eutectic or Primary Si) per connected Si particle")
    ap.add_argument("--breakdown", action="store_true",
                    help="split errors into boundary shifts vs whole-structure mislabels")
    ap.add_argument("--save_dir", default=None)
    ap.add_argument("--scores_out", default=None,
                    help="write score JSON here without saving prediction PNGs")
    ap.add_argument("--save_prob", action="store_true",
                    help="save compressed float16 probabilities for OOF audit/distillation")
    args = ap.parse_args()
    if args.save_prob and not args.save_dir:
        ap.error("--save_prob requires --save_dir")
    if not args.scale_tta or any(value <= 0 for value in args.scale_tta):
        ap.error("--scale_tta values must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, tiles, scales, settings = [], [], [], []
    for c in args.ckpt:
        m, a = load_model(c, device)
        models.append(m)
        tiles.append(a.get("tile", 512))
        scales.append(a.get("input_scale", 1.0))
        settings.append(a)
    tile = args.tile or max(tiles)

    import glob
    split_dir = os.path.join(args.data_root, SPLIT_DIRS[args.split])
    paths = read_manifest(args.manifest, args.data_root) if args.manifest else \
        sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
    has_mask = bool(paths) and all(mask_path(p).is_file() for p in paths)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    source = f"manifest {args.manifest}" if args.manifest else f"split {args.split}"
    print(f"checkpoints: {len(models)}  {source} ({len(paths)} images)")
    print(f"tile={tile} overlap={args.overlap} tta={args.tta} d4={args.d4} "
          f"scale_tta={args.scale_tta} si_vote={args.si_vote} "
          f"ground truth: {'yes' if has_mask else 'no'}")
    if any(s != 1 for s in scales):
        print("input scales: " + " ".join(f"{s:g}" for s in scales))
    if any(a.get("tone_norm") or a.get("scale_by_height") for a in settings):
        print("per-scan standardisation: " + " | ".join(
            f"tone {a.get('tone_norm') or '-'} scale {a.get('scale_by_height') or '-'}"
            for a in settings))
    print()

    scores, per_class, rows, breakdown = [], [], [], []
    for p in paths:
        name = os.path.basename(p)
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Average probabilities across models, never labels: a majority vote on
        # argmax discards how confident each model was. Each model gets the
        # scan the way it was trained: its own tone standardisation and scale.
        prob, inputs = None, {}
        for m, s, a in zip(models, scales, settings):
            key = (tuple(a.get("tone_norm") or ()), tuple(a.get("tone_target") or ()))
            if key not in inputs:
                inputs[key] = prepare(img, a).to(device)
            table = parse_scale_table(a.get("scale_by_height"))
            q = None
            for view_tile, view_scale in inference_views(tile, s, img.shape[0], table,
                                                         args.scale_tta):
                view = predict_tiled(m, inputs[key], tile=view_tile, overlap=args.overlap,
                                     num_classes=NUM_CLASSES, flip_tta=args.tta,
                                     transpose_tta=args.d4, scale=view_scale)
                q = view if q is None else q + view
            q = q / len(args.scale_tta)
            prob = q if prob is None else prob + q
        prob = prob / len(models)
        if args.si_vote:
            pred = si_particle_vote(prob)
        else:
            pred = prob.argmax(0).cpu().numpy().astype(np.uint8)

        if has_mask:
            mp = str(mask_path(p))
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
            if args.save_prob:
                np.savez_compressed(
                    os.path.join(args.save_dir, name.replace("_image.png", "_prob.npz")),
                    prob=prob.detach().cpu().numpy().astype(np.float16), image=str(p))

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
        payload = {"mean_miou": mean_miou, "per_image": rows,
                   "tile": tile, "overlap": args.overlap,
                   "flip_tta": args.tta, "d4": args.d4,
                   "scale_tta": args.scale_tta, "si_vote": args.si_vote,
                   "split": args.split, "manifest": args.manifest,
                   "checkpoints": args.ckpt}
        destinations = ([os.path.join(args.save_dir, "scores.json")] if args.save_dir else [])
        if args.scores_out:
            destinations.append(args.scores_out)
        for destination in dict.fromkeys(destinations):
            os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
            with open(destination, "w") as f:
                json.dump(payload, f, indent=2)
    if args.save_dir:
        print(f"\npredictions -> {args.save_dir}/")


if __name__ == "__main__":
    main()
