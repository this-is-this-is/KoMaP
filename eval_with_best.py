"""Does a group of models add to the best ensemble?

Scores, under the final inference setting (8-way TTA, 768-px native tiles, Si
particle vote), the six 200-epoch models that hold the best validation score,
the given group on its own, and the two together -- with equal weights and with
the newcomers at half weight. A model family that is weaker alone can still be
worth having if it errs elsewhere; this is where that shows.

--also names groups that have already earned their place (SE-ResNeXt-50, say):
the six plus those is then the ensemble to beat, and the group is added to it too.
Every row is compared with the ensemble to beat; the interval is a paired bootstrap
over the 20 validation scans, the only honest error bar a single ensemble has.
Writes runs/with_best_<group>.json.

    ./.conda/bin/python eval_with_best.py b_seresnext50_cs2_aug3
    ./.conda/bin/python eval_with_best.py b_dpn92_cs2_aug3 --also b_seresnext50_cs2_aug3
"""
import argparse
import glob
import json
import os
import re
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

BEST = [f"b_cs2_{recipe}_ep200_s{s}" for recipe in ("aug3", "plus") for s in range(3)]


def group_runs(group):
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"runs/{group}_s*/best.pth")
                  if re.search(rf"/{re.escape(group)}_s\d+/best.pth$", p))
    if not runs:
        raise SystemExit(f"no finished runs for group {group}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("group", help="runs/<group>_s<seed>")
    ap.add_argument("--also", nargs="+", default=[], metavar="GROUP",
                    help="groups counted as part of the ensemble to beat")
    args = ap.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new = group_runs(args.group)
    also = [r for g in args.also for r in group_runs(g)]

    data = []
    for p in sorted(glob.glob("valid/valid/images/*.png")):
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        data.append((cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB),
                     rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))))
    cache = {}
    for r in BEST + also + new:
        model, a = load_model(f"runs/{r}/best.pth", device)
        s = a.get("input_scale", 1.0)
        table = parse_scale_table(a.get("scale_by_height"))
        cache[r] = [predict_tiled(model, prepare(img, a).to(device), tile=scaled_tile(768, s), overlap=0.5,
                                  num_classes=NUM_CLASSES, flip_tta=True, transpose_tta=True,
                                  scale=scan_scale(img.shape[0], s, table)).half().cpu().numpy()
                    for img, _ in data]
        del model
        torch.cuda.empty_cache()

    def score(members, weights):
        scores, ious = [], []
        for i, (_, gt) in enumerate(data):
            prob = sum(w * cache[r][i].astype(np.float32) for w, r in zip(weights, members)) / sum(weights)
            miou, iou = per_image_miou(si_particle_vote(prob), gt)
            scores.append(miou)
            ious.append(iou)
        return np.array(scores), 100 * np.nanmean(ious, 0)

    best = "best 6 (200 epochs)"
    base = BEST + also                  # the ensemble to beat
    to_beat = "best 6 + " + " + ".join(f"{g} x{len(group_runs(g))}" for g in args.also) if also else best
    rows = [(best, BEST, [1.0] * 6)]
    if also:
        rows.append((to_beat, base, [1.0] * len(base)))
    rows.append((f"{args.group} x{len(new)}", new, [1.0] * len(new)))
    if also:
        rows.append((f"best 6 + {args.group} x{len(new)}", BEST + new, [1.0] * (6 + len(new))))
    rows += [(f"{to_beat.replace(' (200 epochs)', '')} + {args.group} x{len(new)}", base + new,
              [1.0] * (len(base) + len(new))),
             ("  the same, newcomers at half weight", base + new, [1.0] * len(base) + [0.5] * len(new))]

    ref = score(base, [1.0] * len(base))[0]
    resample = np.random.default_rng(0).integers(0, len(data), (10000, len(data)))
    out = {}
    width = max(len(r[0]) for r in rows) + 1
    print(f"{'8-way TTA, tile 768, Si vote':{width}s}  n    mIoU  vs to-beat    95% interval   img +/-    "
          + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
    for name, members, weights in rows:
        s, c = score(members, weights)
        d = s - ref
        lo, hi = np.percentile(100 * d[resample].mean(1), [2.5, 97.5])
        out[name.strip()] = {"miou": 100 * float(s.mean()), "vs_to_beat": 100 * float(d.mean()),
                             "interval95": [float(lo), float(hi)], "per_image": (100 * s).tolist(),
                             "class_iou": c.tolist(), "members": members, "weights": weights}
        versus = (f"{'   to beat':32s}" if members == base and weights == [1.0] * len(base) else
                  f"{100 * d.mean():+6.2f}   [{lo:+5.2f}, {hi:+5.2f}]    {np.sum(d > 1e-4):2d}/{np.sum(d < -1e-4):2d}")
        print(f"{name:{width}s} {len(members):2d}  {100 * s.mean():6.2f}      {versus}    "
              + "  ".join(f"{x:5.1f}" for x in c))
    print(f"to beat: {to_beat}. The interval is a paired bootstrap over the {len(data)} validation scans.")
    with open(f"runs/with_best_{args.group}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote runs/with_best_{args.group}.json")


if __name__ == "__main__":
    main()
