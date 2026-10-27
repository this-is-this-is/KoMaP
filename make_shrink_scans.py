"""Small-scan-style training scans made from the large ones, with their real annotation.

The 480 x 675 scans are where the models score lowest, and only 17 of the 70
training scans are of that kind (half of the validation scans are). Shrinking
a large scan to 480 px high reproduces the small scans' structure statistics
(Eutectic Si mean depth 1.2-1.3 px against 1.2 px, a boundary on 15-22% of the
pixels against 21%), so the small scans look like the same microstructure at
about half the magnification. What shrinking does not reproduce is their tone:
the Si phases are much darker there. So each shrunk scan is also given the
phase gray levels of a real small scan, by a monotone tone curve through the
phase medians.

Written as extra "scans" for train_v2.py --synth_dir (no diffusion involved):

    synth/shrink        2 tone variants of each large scan (shrunk + tone curve)
    synth/shrink_sharp  the same, sharpened until the set is as crisp as the real small
                        scans (their edges are steeper than area-averaging leaves them)
    synth/shrink_plain  each large scan shrunk, tone untouched
    synth/small_real    the 17 real small scans again: the control that only
                        raises their share of the crops

    ./.conda/bin/python make_shrink_scans.py
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from komap import CLASSES, NUM_CLASSES, index_to_rgb, rgb_to_index

SMALL = 480      # height of the small scans
MIN_PIXELS = 300  # a phase needs this many pixels for its median to be used


def load(split_dir):
    scans = []
    for p in sorted(glob.glob(os.path.join(split_dir, "images", "*.png"))):
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        scans.append((os.path.basename(p).replace("_image.png", ""), cv2.imread(p, cv2.IMREAD_GRAYSCALE),
                      rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))))
    return scans


def shrink(image, mask, height=SMALL):
    """Lower magnification: area-average the image; each output pixel takes the phase
    that covers most of it, so no in-between labels appear."""
    f = height / mask.shape[0]
    size = (int(round(mask.shape[1] * f)), height)
    planes = np.stack([cv2.resize((mask == c).astype(np.float32), size, interpolation=cv2.INTER_AREA)
                       for c in range(NUM_CLASSES)])
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA), planes.argmax(0).astype(np.uint8)


def phase_medians(image, mask):
    return {c: float(np.median(image[mask == c])) for c in range(NUM_CLASSES) if (mask == c).sum() >= MIN_PIXELS}


def tone_curve(source, target):
    """Monotone gray-level curve sending the phase medians of one scan onto another's."""
    shared = sorted((source[c], target[c]) for c in source if c in target)
    xs = np.array([0.0] + [s for s, _ in shared] + [255.0])
    ys = np.array([0.0] + [t for _, t in shared] + [255.0])
    # Two phases can share a gray level in one scan and not in the other; keep the curve
    # single-valued and non-decreasing.
    keep = np.concatenate([[True], np.diff(xs) > 0])
    return np.interp(np.arange(256), xs[keep], np.maximum.accumulate(ys[keep])).round().astype(np.uint8)


def sharpen(image, amount, sigma=1.0):
    """Unsharp mask: the image plus `amount` times what a Gaussian blur removes."""
    x = image.astype(np.float32)
    return np.clip(x + amount * (x - cv2.GaussianBlur(x, (0, 0), sigma)), 0, 255).round().astype(np.uint8)


def sharpness(image):
    return float(cv2.Laplacian(image.astype(np.float32), cv2.CV_32F).var())


def write(folder, name, image, mask):
    os.makedirs(os.path.join(folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
    cv2.imwrite(os.path.join(folder, "images", f"{name}_image.png"), np.repeat(image[..., None], 3, 2))
    cv2.imwrite(os.path.join(folder, "masks", f"{name}_mask.png"), cv2.cvtColor(index_to_rgb(mask), cv2.COLOR_RGB2BGR))


def describe(pairs):
    """Median sharpness, Eutectic Si depth, boundary share and phase grays of a set of scans."""
    sharp, depth, edge, gray = [], [], [], {c: [] for c in range(NUM_CLASSES)}
    for image, mask in pairs:
        sharp.append(sharpness(image))
        e = (mask == 2).astype(np.uint8)
        if e.sum() >= MIN_PIXELS:
            depth.append(cv2.distanceTransform(e, cv2.DIST_L2, 5)[e > 0].mean())
        b = np.zeros(mask.shape, bool)
        b[:-1] |= mask[1:] != mask[:-1]
        b[:, :-1] |= mask[:, 1:] != mask[:, :-1]
        edge.append(100 * b.mean())
        for c, v in phase_medians(image, mask).items():
            gray[c].append(v)
    return {"sharpness": float(np.median(sharp)), "eutectic_depth": float(np.median(depth)),
            "boundary_share": float(np.median(edge)),
            "gray": {CLASSES[c]: float(np.mean(v)) for c, v in gray.items()},
            "gray_sd": {CLASSES[c]: float(np.std(v)) for c, v in gray.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=ROOT)
    ap.add_argument("--out", default=os.path.join(ROOT, "synth"))
    ap.add_argument("--variants", type=int, default=2, help="tone variants per large scan")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for sub in ("shrink", "shrink_sharp", "shrink_plain", "small_real"):
        if os.path.exists(os.path.join(args.out, sub)):
            raise SystemExit(f"{os.path.join(args.out, sub)} exists; remove it to regenerate")

    rng = np.random.default_rng(args.seed)
    scans = load(os.path.join(args.data_root, "train/train"))
    small = [s for s in scans if s[2].shape[0] <= SMALL + 20]
    large = [s for s in scans if s[2].shape[0] > SMALL + 20]
    targets = [phase_medians(image, mask) for _, image, mask in small]
    made = {"shrink_plain": []}
    plan = []  # name, shrunk image, mask, phase grays of the small scan it imitates
    for name, image, mask in large:
        img, msk = shrink(image, mask)
        write(os.path.join(args.out, "shrink_plain"), f"{name}_shrink", img, msk)
        made["shrink_plain"].append((img, msk))
        for k in rng.choice(len(targets), args.variants, replace=False):
            plan.append((f"{name}_as_{small[int(k)][0]}", img, msk, targets[int(k)]))

    def render(amount):
        """Sharpen first, then set the tone, so the phase grays land on the target either way."""
        out = []
        for _, img, msk, target in plan:
            x = sharpen(img, amount) if amount > 0 else img
            out.append(tone_curve(phase_medians(x, msk), target)[x])
        return out

    # One sharpening amount for the whole set, found by bisection on the median sharpness.
    goal = np.median([sharpness(image) for _, image, _ in small])
    lo, hi = 0.0, 4.0
    for _ in range(16):
        amount = (lo + hi) / 2
        if np.median([sharpness(x) for x in render(amount)]) < goal:
            lo = amount
        else:
            hi = amount
    for sub, images in (("shrink", render(0.0)), ("shrink_sharp", render(amount))):
        made[sub] = [(x, msk) for x, (_, _, msk, _) in zip(images, plan)]
        for x, (name, _, msk, _) in zip(images, plan):
            write(os.path.join(args.out, sub), name, x, msk)
    for name, image, mask in small:
        write(os.path.join(args.out, "small_real"), name, image, mask)

    stats = {"real small scans": describe([(i, m) for _, i, m in small]),
             "shrink_plain": describe(made["shrink_plain"]), "shrink": describe(made["shrink"]),
             "shrink_sharp": describe(made["shrink_sharp"])}
    stats["sharpen_amount"] = amount
    print(f"{len(large)} large scans -> {len(made['shrink_plain'])} shrunk, {len(made['shrink'])} shrunk + toned "
          f"(also sharpened, unsharp amount {amount:.2f}); {len(small)} real small scans copied")
    print(f"{'':18s} sharpness  EutSi depth  boundary %   gray: " + " / ".join(CLASSES) + "  (sd across scans)")
    for k, v in stats.items():
        if not isinstance(v, dict):
            continue
        print(f"{k:18s} {v['sharpness']:8.0f}  {v['eutectic_depth']:10.2f}  {v['boundary_share']:9.1f}    "
              + " / ".join(f"{v['gray'][c]:.0f}" for c in CLASSES)
              + "   (" + " / ".join(f"{v['gray_sd'][c]:.0f}" for c in CLASSES) + ")")
    with open(os.path.join(args.out, "shrink", "stats.json"), "w") as f:
        json.dump(stats, f, indent=1)


if __name__ == "__main__":
    main()
