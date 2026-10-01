"""Compare ensembles on valid under the final inference setting.

Every model's probabilities (8-way TTA, 768-px native tiles, at the model's own
--input_scale) are computed once and cached in memory, so any number of
combinations can then be scored cheaply; each gets the Si particle vote.
Combinations whose runs do not exist yet are skipped.

    ./.conda/bin/python eval_ensembles.py
"""
import glob
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from komap import CLASSES, NUM_CLASSES, per_image_miou, rgb_to_index
from postprocess import si_particle_vote
from predict import MEAN, STD, load_model
from tiling import predict_tiled, scaled_tile

TILE = 768


def seeds(name):
    return [f"b_{name}_s{s}" for s in range(3)]


FULL, COLOR, SCALE2, BOTH, B4 = (seeds("fullres"), seeds("aug_color"), seeds("scale2"),
                                 seeds("color_scale2"), ["s4_micronet_b4"])
COMBOS = {
    "fullres x3 + B4 (reference)": FULL + B4,
    "aug_color x3": COLOR,
    "aug_color x3 + B4": COLOR + B4,
    "scale2 x3": SCALE2,
    "scale2 x3 + B4": SCALE2 + B4,
    "aug_color x3 + scale2 x3": COLOR + SCALE2,
    "aug_color x3 + scale2 x3 + B4": COLOR + SCALE2 + B4,
    "fullres x3 + aug_color x3 + scale2 x3 + B4": FULL + COLOR + SCALE2 + B4,
    "color_scale2 x3": BOTH,
    "color_scale2 x3 + B4": BOTH + B4,
    "color_scale2 x3 + aug_color x3 + scale2 x3": BOTH + COLOR + SCALE2,
    "color_scale2 x3 + aug_color x3 + scale2 x3 + B4": BOTH + COLOR + SCALE2 + B4,
}


def main():
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    have = lambda r: os.path.exists(f"runs/{r}/best.pth")
    combos = {k: v for k, v in COMBOS.items() if all(have(r) for r in v)}
    for k in COMBOS:
        if k not in combos:
            print(f"skip (runs missing): {k}")

    xs, gts = [], []
    for p in sorted(glob.glob("valid/valid/images/*.png")):
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = ((img / 255.0 - np.array(MEAN)) / np.array(STD)).transpose(2, 0, 1)
        xs.append(torch.from_numpy(x).float()[None])
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        gts.append(rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))

    cache = {}
    for r in sorted({r for v in combos.values() for r in v}):
        model, a = load_model(f"runs/{r}/best.pth", device)
        s = a.get("input_scale", 1.0)
        cache[r] = [predict_tiled(model, x.to(device), tile=scaled_tile(TILE, s), overlap=0.5,
                                  num_classes=NUM_CLASSES, flip_tta=True, transpose_tta=True,
                                  scale=s).half().cpu().numpy() for x in xs]
        print(f"cached {r} (scale {s:g})", flush=True)
        del model
        torch.cuda.empty_cache()

    def score(members):
        s, c = [], []
        for i, gt in enumerate(gts):
            prob = sum(cache[r][i].astype(np.float32) for r in members) / len(members)
            miou, ious = per_image_miou(si_particle_vote(prob), gt)
            s.append(miou)
            c.append(ious)
        return np.array(s), 100 * np.nanmean(c, 0)

    ref = score(next(iter(combos.values())))[0]
    print(f"\n{'ensemble (8-way TTA, tile 768, Si vote)':48s}  n   mIoU    vs ref  img +/-   "
          + "  ".join(f"{c[:5]:>5s}" for c in CLASSES))
    for k, v in combos.items():
        s, c = score(v)
        d = s - ref
        print(f"{k:48s} {len(v):2d}  {100 * s.mean():6.2f}%  {100 * d.mean():+5.2f}  "
              f"{np.sum(d > 1e-4):2d}/{np.sum(d < -1e-4):2d}    " + "  ".join(f"{x:5.1f}" for x in c))


if __name__ == "__main__":
    main()
