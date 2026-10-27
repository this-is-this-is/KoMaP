"""Inference for the B3 + SE-ResNeXt OOF-trained logit fusion head."""
from __future__ import annotations

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
from group_cv import mask_path, read_manifest
from komap import CLASSES, NUM_CLASSES, index_to_rgb, per_image_miou, rgb_to_index
from logit_fusion import load_fusion_checkpoint, normalise_probability
from postprocess import si_particle_vote
from predict import inference_views, load_model
from preprocess import parse_scale_table, prepare
from tiling import predict_tiled

SPLIT_DIRS = {"train": "train/train", "valid": "valid/valid", "test": "test/test"}


def load_family(paths, device):
    models, settings = [], []
    for path in paths:
        model, args = load_model(path, device)
        models.append(model)
        settings.append(args)
    return models, settings


@torch.inference_mode()
def family_probability(image, models, settings, signature, device):
    total, prepared = None, {}
    for model, args in zip(models, settings):
        key = (tuple(args.get("tone_norm") or ()), tuple(args.get("tone_target") or ()))
        if key not in prepared:
            prepared[key] = prepare(image, args).to(device)
        trained_scale = args.get("input_scale", 1.0)
        scale_table = parse_scale_table(args.get("scale_by_height"))
        probability = None
        for tile, scale in inference_views(
                signature["tile"], trained_scale, image.shape[0], scale_table,
                signature["scale_tta"]):
            view = predict_tiled(
                model, prepared[key], tile=tile, overlap=signature["overlap"],
                num_classes=NUM_CLASSES, flip_tta=signature["flip_tta"],
                transpose_tta=signature["d4"], scale=scale)
            probability = view if probability is None else probability + view
        probability = probability / len(signature["scale_tta"])
        total = probability if total is None else total + probability
    return normalise_probability((total / len(models))[None])[0]


def metric_payload(rows):
    output = {}
    for method in ("b3", "se", "equal", "fusion"):
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        per_class = np.array([
            [np.nan if value is None else value for value in row["per_class"]]
            for row in selected
        ], np.float64)
        output[method] = {
            "mean_miou": float(np.mean([row["miou"] for row in selected])),
            "per_class_iou": {
                name: float(value) for name, value in zip(
                    CLASSES, np.nanmean(per_class, 0))
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3_ckpt", nargs="+", required=True)
    parser.add_argument("--se_ckpt", nargs="+", required=True)
    parser.add_argument("--fusion_ckpt", required=True)
    parser.add_argument("--data_root", default=str(ROOT))
    parser.add_argument("--split", choices=sorted(SPLIT_DIRS), default="test")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--save_prob", action="store_true")
    parser.add_argument("--no_si_vote", action="store_true",
                        help="disable the same Si-particle vote used for OOF scoring")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion, fusion_value = load_fusion_checkpoint(args.fusion_ckpt, device)
    signature = fusion_value.get("inference_signature")
    if not signature or any(key not in signature for key in
                            ("tile", "overlap", "flip_tta", "d4", "scale_tta")):
        raise ValueError("fusion checkpoint has no complete OOF inference signature")
    b3_models, b3_settings = load_family(args.b3_ckpt, device)
    se_models, se_settings = load_family(args.se_ckpt, device)
    b3_encoders = {value.get("encoder", "") for value in b3_settings}
    se_encoders = {value.get("encoder", "") for value in se_settings}
    if b3_encoders != {"efficientnet-b3"}:
        raise ValueError(f"--b3_ckpt must contain EfficientNet-B3 models, got {b3_encoders}")
    if se_encoders != {"se_resnext50_32x4d"}:
        raise ValueError(f"--se_ckpt must contain SE-ResNeXt-50 models, got {se_encoders}")

    data_root = Path(args.data_root).resolve()
    if args.manifest:
        paths = read_manifest(args.manifest, data_root)
        source = f"manifest {args.manifest}"
    else:
        paths = sorted(glob.glob(str(data_root / SPLIT_DIRS[args.split] / "images" / "*.png")))
        source = f"split {args.split}"
    if not paths:
        raise FileNotFoundError(f"no images for {source}")
    has_mask = all(mask_path(path).is_file() for path in paths)
    out = Path(args.save_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  {source}: {len(paths)} images  ground truth="
          f"{'yes' if has_mask else 'no'}")
    print(f"B3={len(b3_models)} model(s)  SE={len(se_models)} model(s)")
    print(f"OOF-matched inference: {signature}  Si vote={not args.no_si_vote}\n")

    rows = []
    for path in paths:
        name = Path(path).name
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(path)
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        b3 = family_probability(image, b3_models, b3_settings, signature, device)
        se = family_probability(image, se_models, se_settings, signature, device)
        equal = (b3 + se) * 0.5
        with torch.inference_mode():
            fused = fusion(b3[None], se[None]).softmax(1)[0]
        probabilities = {"b3": b3, "se": se, "equal": equal, "fusion": fused}
        predictions = {
            method: (prob.argmax(0).cpu().numpy().astype(np.uint8)
                     if args.no_si_vote else si_particle_vote(prob))
            for method, prob in probabilities.items()
        }
        prediction = predictions["fusion"]
        cv2.imwrite(str(out / name.replace("_image.png", "_pred.png")),
                    cv2.cvtColor(index_to_rgb(prediction), cv2.COLOR_RGB2BGR))
        if args.save_prob:
            np.savez_compressed(out / name.replace("_image.png", "_prob.npz"),
                                prob=fused.cpu().numpy().astype(np.float16), image=str(path))

        if has_mask:
            target_bgr = cv2.imread(str(mask_path(path)), cv2.IMREAD_COLOR)
            target = rgb_to_index(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB))
            image_scores = {}
            for method, pred in predictions.items():
                miou, per_class = per_image_miou(pred, target)
                image_scores[method] = miou
                rows.append({"name": name, "method": method, "miou": miou,
                             "per_class": [None if np.isnan(v) else float(v)
                                           for v in per_class]})
            print(f"  {name:42s} fusion {100*image_scores['fusion']:6.2f}%  "
                  f"equal {100*image_scores['equal']:6.2f}%")
        else:
            print(f"  {name}")

    payload = {"format": "komap_logit_fusion_predictions_v1",
               "fusion_checkpoint": args.fusion_ckpt,
               "b3_checkpoints": args.b3_ckpt, "se_checkpoints": args.se_ckpt,
               "inference_signature": signature, "si_vote": not args.no_si_vote,
               "split": args.split, "manifest": args.manifest,
               "summaries": metric_payload(rows) if has_mask else {},
               "per_image": rows}
    (out / "scores.json").write_text(json.dumps(payload, indent=2) + "\n")
    if has_mask:
        print("\nscore summary")
        for method, value in payload["summaries"].items():
            print(f"  {method:7s}: {100*value['mean_miou']:.2f}%")
    print(f"\npredictions -> {out}")


if __name__ == "__main__":
    main()
