"""Is a folder of synthetic image-mask pairs fit to train on?

Four checks, written to <synth_dir>/quality.json:

  agreement   A segmentation ensemble trained on real data only segments the
              synthetic images; its mIoU against the synthetic masks says
              whether image and mask belong together (the mIoU test of
              semantic image synthesis, and FreeMask's filtering signal). The
              same ensemble, same settings, on the real validation scans is the
              yardstick: synthetic pairs should not score far below it.
  phases      Pixel share of each phase, real training scans vs synthetic.
  gray        Mean gray level of each phase, real vs synthetic.
  copies      Highest normalised cross-correlation of each synthetic image with
              any training scan (quarter resolution, all 8 orientations).
              Validation crops against the training scans show what unseen
              real data scores; training crops show what a copy scores.

    ./.conda/bin/python check_synth.py synth/d1
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from komap import CLASSES, NUM_CLASSES, per_image_miou, rgb_to_index
from postprocess import si_particle_vote
from predict import load_model
from preprocess import parse_scale_table, prepare, scan_scale
from tiling import predict_tiled, scaled_tile


def load_pairs(split_dir, limit=None):
    pairs = []
    for p in sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))[:limit]:
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        pairs.append((cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                      rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))))
    return pairs


def agreement(pairs, runs, device, tile=512):
    """Per-image mIoU of the real-trained ensemble against the given masks."""
    probs = None
    for r in runs:
        model, a = load_model(f"runs/{r}/best.pth", device)
        s = a.get("input_scale", 1.0)
        table = parse_scale_table(a.get("scale_by_height"))
        p = [predict_tiled(model, prepare(img, a).to(device), tile=scaled_tile(tile, s), overlap=0.5,
                           num_classes=NUM_CLASSES, flip_tta=True, transpose_tta=True,
                           # Synthetic tiles are 512 px whatever scan they imitate, so a
                           # per-height scale table must not apply to them.
                           scale=s if img.shape[0] == img.shape[1] else scan_scale(img.shape[0], s, table)
                           ).half().cpu().numpy() for img, _ in pairs]
        probs = p if probs is None else [x + y for x, y in zip(probs, p)]
        del model
        torch.cuda.empty_cache()
    scores, ious = [], []
    for prob, (_, mask) in zip(probs, pairs):
        miou, iou = per_image_miou(si_particle_vote(prob.astype(np.float32)), mask)
        scores.append(miou)
        ious.append(iou)
    return np.array(scores), 100 * np.nanmean(ious, 0)


def phase_stats(pairs):
    count, gray = np.zeros(NUM_CLASSES), np.zeros(NUM_CLASSES)
    for img, mask in pairs:
        count += np.bincount(mask.ravel(), minlength=NUM_CLASSES)
        gray += np.bincount(mask.ravel(), weights=img[..., 0].ravel().astype(np.float64), minlength=NUM_CLASSES)
    return (100 * count / count.sum()).tolist(), (gray / np.maximum(count, 1)).tolist()


def best_match(templates, scans):
    """For each template the highest correlation with any scan, over the 8 orientations."""
    out = []
    for t in templates:
        best = -1.0
        for k in range(8):
            v = np.ascontiguousarray(np.rot90(t[:, ::-1] if k >= 4 else t, k % 4))
            for s in scans:
                if s.shape[0] >= v.shape[0] and s.shape[1] >= v.shape[1]:
                    best = max(best, float(cv2.matchTemplate(s, v, cv2.TM_CCOEFF_NORMED).max()))
        out.append(best)
    return np.array(out)


def quarter(img):
    return cv2.resize(img[..., 0], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)


def random_crops(pairs, size, rng):
    out = []
    for img, _ in pairs:
        y = int(rng.integers(0, img.shape[0] - size + 1))
        x = int(rng.integers(0, img.shape[1] - size + 1))
        out.append(quarter(img[y:y + size, x:x + size]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("synth_dir")
    ap.add_argument("--runs", nargs="+", default=[f"b_cs2_aug3_s{s}" for s in range(3)],
                    help="real-only segmentation runs used as the judge")
    ap.add_argument("--count", type=int, default=200, help="synthetic pairs scored for agreement")
    ap.add_argument("--copies", type=int, default=40, help="synthetic images checked against the scans")
    args = ap.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)

    synth = load_pairs(args.synth_dir)
    train = load_pairs("train/train")
    valid = load_pairs("valid/valid")
    print(f"{len(synth)} synthetic pairs, {len(train)} training scans, {len(valid)} validation scans")

    s_score, s_cls = agreement(synth[:args.count], args.runs, device)
    v_score, v_cls = agreement(valid, args.runs, device)
    print(f"\nagreement of the real-trained ensemble ({', '.join(args.runs)}; 8-way TTA, tile 512, Si vote)")
    print(f"  {'':22s}  mIoU   " + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
    print(f"  {'real validation scans':22s} {100 * v_score.mean():6.2f}  " + "  ".join(f"{x:5.1f}" for x in v_cls))
    print(f"  {'synthetic pairs':22s} {100 * s_score.mean():6.2f}  " + "  ".join(f"{x:5.1f}" for x in s_cls)
          + f"   (per image: 10th percentile {100 * np.percentile(s_score, 10):.1f}, "
          f"median {100 * np.median(s_score):.1f})")

    share_r, gray_r = phase_stats(train)
    share_s, gray_s = phase_stats(synth)
    print("\nphase pixel share (%):   real " + "  ".join(f"{x:5.1f}" for x in share_r)
          + "   synthetic " + "  ".join(f"{x:5.1f}" for x in share_s))
    print("phase mean gray level:   real " + "  ".join(f"{x:5.1f}" for x in gray_r)
          + "   synthetic " + "  ".join(f"{x:5.1f}" for x in gray_s))

    scans = [quarter(img) for img, _ in train]
    # 384 px windows (96 px at quarter resolution) fit inside the smallest scan.
    pick = rng.choice(len(synth), min(args.copies, len(synth)), replace=False)
    c_synth = best_match(random_crops([synth[i] for i in pick], 384, rng), scans)
    c_valid = best_match(random_crops(valid, 384, rng), scans)
    c_train = best_match(random_crops([train[i] for i in rng.choice(len(train), 10, replace=False)], 384, rng), scans)
    print("\nhighest correlation with any training scan (median / max):")
    print(f"  training crops (copies)  {np.median(c_train):.2f} / {c_train.max():.2f}")
    print(f"  validation crops (new)   {np.median(c_valid):.2f} / {c_valid.max():.2f}")
    print(f"  synthetic images         {np.median(c_synth):.2f} / {c_synth.max():.2f}")

    out = {"judge_runs": args.runs,
           "agreement": {"valid_miou": 100 * float(v_score.mean()), "valid_class_iou": v_cls.tolist(),
                         "synthetic_miou": 100 * float(s_score.mean()), "synthetic_class_iou": s_cls.tolist(),
                         "synthetic_per_image": (100 * s_score).tolist()},
           "phase_share": {"real": share_r, "synthetic": share_s},
           "phase_gray": {"real": gray_r, "synthetic": gray_s},
           "copies": {"train": c_train.tolist(), "valid": c_valid.tolist(), "synthetic": c_synth.tolist()}}
    with open(os.path.join(args.synth_dir, "quality.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {os.path.join(args.synth_dir, 'quality.json')}")


if __name__ == "__main__":
    main()
