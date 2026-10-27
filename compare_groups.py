"""Compare two training recipes on valid under the final inference setting.

A group is runs/<group>_s<seed>. Each model's probabilities (8-way TTA,
768-px native tiles, the model's own scale) are computed once; single models,
each group's seed ensemble and the two groups together are then scored with
the Si particle vote, alongside where the errors are:

    <=1px / 2-5px / >5px   error pixels by distance to the nearest GT pixel of
                           the predicted class (src/diagnostics.py): boundary
                           placement versus whole-structure mislabelling
    A->E >5px, E->A >1px   the main phase confusion, Al3Ni called Eutectic Si
                           far from any Eutectic Si and the reverse, as a share
                           of the true class's pixels

Writes runs/compare_<ref>_vs_<new>.json (with a _tile<N> suffix for a tile other than 768).

    ./.conda/bin/python compare_groups.py b_cs2_aug3 b_cs2_aug3_edge
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
from scipy import stats

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from diagnostics import BUCKETS, error_breakdown
from komap import CLASSES, NUM_CLASSES, per_image_miou, rgb_to_index
from postprocess import si_particle_vote
from predict import load_model
from preprocess import parse_scale_table, prepare, scan_scale
from tiling import predict_tiled, scaled_tile

AL3NI, EUT = 1, 2


def group_runs(group, seeds=None):
    runs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"runs/{group}_s*/best.pth")
                  if re.search(rf"/{re.escape(group)}_s\d+/best.pth$", p))
    if not runs:
        raise SystemExit(f"no finished runs for group {group}")
    if seeds is not None:
        requested = [f"{group}_s{s}" for s in seeds]
        missing = set(requested) - set(runs)
        if missing:
            raise SystemExit(f"missing requested runs: {sorted(missing)}")
        runs = requested
    return runs


def confusion(pred, gt):
    """Al3Ni->EutSi pixels >5 px from any EutSi, EutSi->Al3Ni >1 px from any Al3Ni, class totals."""
    to_eut = cv2.distanceTransform((gt != EUT).astype(np.uint8), cv2.DIST_L2, 5)
    to_al3 = cv2.distanceTransform((gt != AL3NI).astype(np.uint8), cv2.DIST_L2, 5)
    return np.array([((gt == AL3NI) & (pred == EUT) & (to_eut > 5)).sum(),
                     ((gt == EUT) & (pred == AL3NI) & (to_al3 > 1.5)).sum(),
                     (gt == AL3NI).sum(), (gt == EUT).sum()], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="reference group, e.g. b_cs2_aug3")
    ap.add_argument("new", help="group to test, e.g. b_cs2_aug3_edge")
    ap.add_argument("--tile", type=int, default=768, help="tile in native pixels")
    ap.add_argument("--seeds", type=int, nargs="+", help="compare the same explicit seeds in both groups")
    args = ap.parse_args()
    os.chdir(ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {args.ref: group_runs(args.ref, args.seeds), args.new: group_runs(args.new, args.seeds)}

    names, images, gts = [], [], []
    for p in sorted(glob.glob("valid/valid/images/*.png")):
        names.append(os.path.basename(p).replace("_image.png", ""))
        images.append(cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        gts.append(rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)))

    cache = {}
    for r in [r for v in groups.values() for r in v]:
        model, a = load_model(f"runs/{r}/best.pth", device)
        s = a.get("input_scale", 1.0)
        table = parse_scale_table(a.get("scale_by_height"))
        cache[r] = [predict_tiled(model, prepare(img, a).to(device), tile=scaled_tile(args.tile, s),
                                  overlap=0.5, num_classes=NUM_CLASSES, flip_tta=True,
                                  transpose_tta=True, scale=scan_scale(img.shape[0], s, table)
                                  ).half().cpu().numpy() for img in images]
        print(f"cached {r}", flush=True)
        del model
        torch.cuda.empty_cache()

    def score(members):
        m, c, conf = [], [], np.zeros(4)
        buckets = {name: 0 for name, _, _ in BUCKETS}
        for i, gt in enumerate(gts):
            prob = sum(cache[r][i].astype(np.float32) for r in members) / len(members)
            pred = si_particle_vote(prob)
            miou, ious = per_image_miou(pred, gt)
            m.append(miou)
            c.append(ious)
            for name, (n, _) in error_breakdown(pred, gt).items():
                buckets[name] += n
            conf += confusion(pred, gt)
        return {"miou": 100 * float(np.mean(m)),
                "class_iou": dict(zip(CLASSES, (100 * np.nanmean(c, 0)).round(2).tolist())),
                "error_px": buckets,
                "al3ni_to_eut_gt5px": 100 * conf[0] / conf[2], "eut_to_al3ni_gt1px": 100 * conf[1] / conf[3],
                "per_image": dict(zip(names, (100 * np.array(m)).tolist()))}

    rows = {r: score([r]) for v in groups.values() for r in v}
    for g, v in groups.items():
        rows[f"{g} x{len(v)}"] = score(v)
    rows[f"{args.ref} + {args.new}"] = score(groups[args.ref] + groups[args.new])

    width = max(len(k) for k in rows) + 1
    print(f"\n{'8-way TTA, tile ' + str(args.tile) + ', Si vote':{width}s}  mIoU   "
          + "  ".join(f"{c[:5]:>5s}" for c in CLASSES)
          + "  | error px (k): <=1px  2-5px   >5px | A->E >5px  E->A >1px (% of class)")
    for k, v in rows.items():
        e = v["error_px"]
        print(f"{k:{width}s} {v['miou']:6.2f}  " + "  ".join(f"{x:5.1f}" for x in v["class_iou"].values())
              + f"  |            {e['<=1px'] / 1e3:5.0f}  {e['2-5px'] / 1e3:5.0f}  {e['>5px'] / 1e3:5.0f}"
              + f" |   {v['al3ni_to_eut_gt5px']:7.2f}   {v['eut_to_al3ni_gt1px']:7.2f}")

    single = {g: [rows[r]["miou"] for r in v] for g, v in groups.items()}
    a, b = single[args.ref], single[args.new]
    p = stats.ttest_ind(b, a, equal_var=False).pvalue if len(a) > 1 and len(b) > 1 else float("nan")
    print(f"\nsingle models: {args.ref} {np.mean(a):.2f} +/- {np.std(a, ddof=1):.2f}, "
          f"{args.new} {np.mean(b):.2f} +/- {np.std(b, ddof=1):.2f}  "
          f"(diff {np.mean(b) - np.mean(a):+.2f}pp, Welch p={p:.3f})")
    ref = rows[f"{args.ref} x{len(groups[args.ref])}"]["per_image"]
    new = rows[f"{args.new} x{len(groups[args.new])}"]["per_image"]
    d = np.array([new[n] - ref[n] for n in names])
    print(f"ensembles: {args.new} vs {args.ref}: {d.mean():+.2f}pp, images better/worse "
          f"{np.sum(d > 1e-4)}/{np.sum(d < -1e-4)}")
    out = f"runs/compare_{args.ref}_vs_{args.new}" + ("" if args.tile == 768 else f"_tile{args.tile}") + ".json"
    if args.seeds is not None:
        out = out[:-5] + "_seeds" + "_".join(map(str, args.seeds)) + ".json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
