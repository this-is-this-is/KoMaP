"""Did --al3ni_darken_p fix the error it was built for?

Scores cs2_aug3 (reference) and cs2_aug3 + Al3Ni darkening on valid under the
final inference setting (8-way TTA, 768-px native tiles, Si particle vote),
single models and 3-seed ensembles, and splits out the errors the
augmentation targets:

    Al3Ni -> EutSi   Al3Ni pixels predicted Eutectic Si, by distance to the
                     nearest Eutectic Si in the ground truth (2-5 px, >5 px),
                     as a share of all Al3Ni pixels
    EutSi -> Al3Ni   the reverse, the side effect to watch, >1 px from Al3Ni

plus the mIoU of the four scans that hold most of the Al3Ni -> EutSi error.
Writes runs/eval_al3ni_darken.json.

    ./.conda/bin/python eval_al3ni_darken.py
"""
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

TILE = 768
AL3NI, EUT = 1, 2
GROUPS = {"cs2_aug3": [f"b_cs2_aug3_s{s}" for s in range(3)],
          "cs2_aug3_dark": [f"b_cs2_aug3_dark_s{s}" for s in range(3)]}
FOCUS = ["A181_step1_sample1", "A191_step4_sample3", "A71_step2_sample1", "A125_step1_sample4"]


def errors(pred, gt):
    """Pixel counts: Al3Ni->EutSi at 2-5 and >5 px, EutSi->Al3Ni at >1 px, and the class totals."""
    to_eut = cv2.distanceTransform((gt != EUT).astype(np.uint8), cv2.DIST_L2, 5)
    to_al3 = cv2.distanceTransform((gt != AL3NI).astype(np.uint8), cv2.DIST_L2, 5)
    a2e = (gt == AL3NI) & (pred == EUT)
    e2a = (gt == EUT) & (pred == AL3NI)
    return np.array([(a2e & (to_eut > 1.5) & (to_eut <= 5)).sum(), (a2e & (to_eut > 5)).sum(),
                     (e2a & (to_al3 > 1.5)).sum(), (gt == AL3NI).sum(), (gt == EUT).sum()], float)


def main():
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    missing = [r for v in GROUPS.values() for r in v if not os.path.exists(f"runs/{r}/best.pth")]
    if missing:
        raise SystemExit(f"runs missing: {missing}")

    names, images, gts = [], [], []
    for p in sorted(glob.glob("valid/valid/images/*.png")):
        names.append(os.path.basename(p).replace("_image.png", ""))
        images.append(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        gts.append(rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))
    focus = [names.index(n) for n in FOCUS]

    cache = {}
    for r in [r for v in GROUPS.values() for r in v]:
        model, a = load_model(f"runs/{r}/best.pth", device)
        s = a.get("input_scale", 1.0)
        table = parse_scale_table(a.get("scale_by_height"))
        cache[r] = [predict_tiled(model, prepare(img, a).to(device), tile=scaled_tile(TILE, s),
                                  overlap=0.5, num_classes=NUM_CLASSES, flip_tta=True,
                                  transpose_tta=True, scale=scan_scale(img.shape[0], s, table)
                                  ).half().cpu().numpy() for img in images]
        print(f"cached {r}", flush=True)
        del model
        torch.cuda.empty_cache()

    def score(members):
        m, c, e = [], [], np.zeros(5)
        for i, gt in enumerate(gts):
            prob = sum(cache[r][i].astype(np.float32) for r in members) / len(members)
            pred = si_particle_vote(prob)
            miou, ious = per_image_miou(pred, gt)
            m.append(miou)
            c.append(ious)
            e += errors(pred, gt)
        m = np.array(m)
        return {"miou": 100 * m.mean(), "focus_miou": 100 * m[focus].mean(),
                "class_iou": dict(zip(CLASSES, (100 * np.nanmean(c, 0)).round(2).tolist())),
                "al3ni_to_eut_2_5px": 100 * e[0] / e[3], "al3ni_to_eut_gt5px": 100 * e[1] / e[3],
                "eut_to_al3ni_gt1px": 100 * e[2] / e[4], "per_image": dict(zip(names, (100 * m).tolist()))}

    rows = {r: score([r]) for v in GROUPS.values() for r in v}
    rows.update({f"{g} x3": score(v) for g, v in GROUPS.items()})
    rows["cs2_aug3 x3 + cs2_aug3_dark x3"] = score(GROUPS["cs2_aug3"] + GROUPS["cs2_aug3_dark"])

    print(f"\n{'final inference (8-way TTA, tile 768, Si vote)':34s}  mIoU   focus4  "
          + "  ".join(f"{c[:5]:>5s}" for c in CLASSES)
          + "   A->E 2-5px  A->E >5px  E->A >1px  (% of GT class px)")
    for k, v in rows.items():
        print(f"{k:34s} {v['miou']:6.2f}  {v['focus_miou']:6.2f}  "
              + "  ".join(f"{x:5.1f}" for x in v["class_iou"].values())
              + f"   {v['al3ni_to_eut_2_5px']:9.2f}  {v['al3ni_to_eut_gt5px']:9.2f}  {v['eut_to_al3ni_gt1px']:9.2f}")
    for g in GROUPS:
        single = [rows[r]["miou"] for r in GROUPS[g]]
        print(f"{g}: single models {np.mean(single):.2f} +/- {np.std(single, ddof=1):.2f}")
    ref, new = rows["cs2_aug3 x3"]["per_image"], rows["cs2_aug3_dark x3"]["per_image"]
    d = np.array([new[n] - ref[n] for n in names])
    print(f"cs2_aug3_dark x3 vs cs2_aug3 x3: {d.mean():+.2f}pp, images better/worse {np.sum(d > 1e-4)}/{np.sum(d < -1e-4)}")
    with open("runs/eval_al3ni_darken.json", "w") as f:
        json.dump(rows, f, indent=1)
    print("wrote runs/eval_al3ni_darken.json")


if __name__ == "__main__":
    main()
