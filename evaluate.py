"""Score a checkpoint the way the KoMaP competition does, and/or write predictions.

The reported number is the mean over images of the 4-class mIoU, which is the
competition's stated metric (per-image mIoU, averaged over the 10 evaluation
images).

Examples:
    ./.conda/bin/python evaluate.py --ckpt runs/unet_r34/best.pth --split valid
    ./.conda/bin/python evaluate.py --ckpt runs/unet_r34/best.pth --split test --save_dir preds/test
"""
import argparse
import json
import os
import sys

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from komap import CLASSES, KomapDataset, build_model, index_to_rgb, per_image_miou

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

SPLIT_DIRS = {"train": "train/train", "valid": "valid/valid", "test": "test/test"}


@torch.no_grad()
def predict_logits(model, image, size, device, tta):
    """Full-resolution logits, optionally averaged over flip TTA.

    Flips are label-preserving for these micrographs, so averaging over them
    costs a little time and reliably adds a bit of accuracy.
    """
    h, w = image.shape[-2:]
    acc = None
    # (horizontal flip, vertical flip)
    variants = [(False, False)] if not tta else [(False, False), (True, False), (False, True), (True, True)]
    for fh, fv in variants:
        x = image
        if fh:
            x = torch.flip(x, [-1])
        if fv:
            x = torch.flip(x, [-2])
        logits = model(x)
        if fh:
            logits = torch.flip(logits, [-1])
        if fv:
            logits = torch.flip(logits, [-2])
        if logits.shape[-2:] != (h, w):
            logits = torch.nn.functional.interpolate(
                logits, size=(h, w), mode="bilinear", align_corners=False
            )
        prob = logits.softmax(1)
        acc = prob if acc is None else acc + prob
    return acc / len(variants)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--split", default="valid", choices=sorted(SPLIT_DIRS))
    ap.add_argument("--size", type=int, default=None, help="defaults to the training size")
    ap.add_argument("--tta", action="store_true", help="average over flip variants")
    ap.add_argument("--native", action="store_true",
                    help="force native-resolution inference (inferred from the checkpoint by default)")
    ap.add_argument("--save_dir", default=None, help="write colour-coded mask PNGs here")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    targs = ckpt.get("args", {})
    size = args.size or targs.get("size", 512)

    # Match how the checkpoint was trained unless explicitly overridden.
    native = args.native or targs.get("native", False)

    model = build_model(targs.get("arch", "unet"), targs.get("encoder", "resnet34"), weights=None)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    split_dir = os.path.join(args.data_root, SPLIT_DIRS[args.split])
    has_mask = os.path.isdir(os.path.join(split_dir, "masks"))
    if native:
        # Native resolution, padded to a multiple of 32 at the bottom/right so
        # the pad slices straight back off.
        tf = A.Compose([
            A.PadIfNeeded(None, None, pad_height_divisor=32, pad_width_divisor=32,
                          position="top_left", border_mode=cv2.BORDER_REFLECT_101),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    else:
        tf = A.Compose([A.Resize(size, size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])
    ds = KomapDataset(split_dir, tf, with_mask=has_mask)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.workers)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    print(f"checkpoint: {args.ckpt}")
    print(f"split: {args.split}  ({len(ds)} images)  "
          f"{'native-res' if native else f'size={size}'}  tta={args.tta}  "
          f"ground truth: {'yes' if has_mask else 'no'}\n")

    scores, per_class, rows = [], [], []
    for batch in dl:
        image = batch["image"].to(device)
        name = batch["name"][0]
        # The mask was resized with the image; score against the original file
        # so the number matches full-resolution ground truth.
        prob = predict_logits(model, image, size, device, args.tta)
        pred = prob.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # Restore the original resolution: both the score and any submitted mask
        # live on the source grid.
        src = cv2.imread(os.path.join(split_dir, "images", name), cv2.IMREAD_COLOR)
        sh, sw = src.shape[:2]
        if pred.shape != (sh, sw):
            if native:
                # Padding was added at the bottom/right, so slice it off. Resizing
                # from the padded size would squash the map off the true grid.
                pred = pred[:sh, :sw]
            else:
                pred = cv2.resize(pred, (sw, sh), interpolation=cv2.INTER_NEAREST)

        if has_mask:
            gt_path = os.path.join(split_dir, "masks", name.replace("_image.png", "_mask.png"))
            from komap import rgb_to_index
            gt = rgb_to_index(cv2.cvtColor(cv2.imread(gt_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            miou, ious = per_image_miou(pred, gt)
            scores.append(miou)
            per_class.append(ious)
            rows.append({"name": name, "miou": miou,
                         **{c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, ious)}})
            print(f"  {name:42s} mIoU {100*miou:6.2f}%   " +
                  "  ".join(f"{c}={'  n/a' if np.isnan(v) else f'{100*v:5.1f}%'}"
                            for c, v in zip(CLASSES, ious)))

        if args.save_dir:
            cv2.imwrite(
                os.path.join(args.save_dir, name.replace("_image.png", "_pred.png")),
                cv2.cvtColor(index_to_rgb(pred), cv2.COLOR_RGB2BGR),
            )

    if has_mask:
        mean_miou = float(np.mean(scores))
        pc = np.nanmean(np.stack(per_class), axis=0)
        print("\n" + "=" * 72)
        print(f"  mean mIoU over {len(scores)} images : {100*mean_miou:.2f}%")
        print("  per-class IoU                : " +
              "  ".join(f"{c}={100*v:.1f}%" for c, v in zip(CLASSES, pc)))
        print(f"  competition reference        : ~75%")
        print("=" * 72)
        if args.save_dir:
            with open(os.path.join(args.save_dir, "scores.json"), "w") as f:
                json.dump({"mean_miou": mean_miou, "per_image": rows}, f, indent=2)

    if args.save_dir:
        print(f"\npredictions written to {args.save_dir}/")


if __name__ == "__main__":
    main()
