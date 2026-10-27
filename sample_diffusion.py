"""Draw synthetic image-mask pairs from the joint diffusion model.

Each sample is one 512 x 512 micrograph with its mask, written the way the
dataset is laid out (images/*_image.png, masks/*_mask.png), so train_v2.py can
read the folder with --synth_dir.

The phase fractions each sample is asked for follow PF-DiffSeg: most are the
fractions of random windows of the training scans, lightly jittered, so the
synthetic set has the real composition; a share is drawn from the windows
richest in Eutectic Si, the phase the segmentation models find hardest.

    ./.conda/bin/python sample_diffusion.py --ckpt runs/diffusion_d1/best.pt --out synth/d1
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from diffusion import NUM_CLASSES, UNet, from_planes, sample
from komap import index_to_rgb
from train_diffusion import load_scans

EUTECTIC = 2


def window_fractions(scans, size, per_scan, rng):
    """Phase fractions of random windows, `per_scan` from every training scan."""
    out = []
    for _, mask in scans:
        h, w = min(size, mask.shape[0]), min(size, mask.shape[1])
        for _ in range(per_scan):
            y = int(rng.integers(0, mask.shape[0] - h + 1))
            x = int(rng.integers(0, mask.shape[1] - w + 1))
            out.append(np.bincount(mask[y:y + h, x:x + w].ravel(), minlength=NUM_CLASSES) / (h * w))
    return np.array(out)


def requested_fractions(scans, size, count, boost, rng):
    """`count` fraction vectors: real ones, a `boost` share from the Eutectic-Si-richest quarter."""
    real = window_fractions(scans, size, 40, rng)
    rich = real[real[:, EUTECTIC] >= np.quantile(real[:, EUTECTIC], 0.75)]
    pick = np.where(rng.random(count)[:, None] < boost,
                    rich[rng.integers(len(rich), size=count)], real[rng.integers(len(real), size=count)])
    # PF-DiffSeg's jitter: noise that grows with the fraction, then back onto the simplex.
    pick = np.clip(pick + 0.02 * (pick + 0.01) * rng.standard_normal(pick.shape), 0, None)
    return pick / pick.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default=ROOT)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32, help="Heun steps (2 network calls each)")
    ap.add_argument("--boost", type=float, default=0.3,
                    help="share of samples asked to be rich in Eutectic Si")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True  # fixed input sizes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    net = UNet(base=ckpt["args"]["base"], dropout=ckpt["args"]["dropout"])
    net.load_state_dict(ckpt["model"])
    net = net.to(device).to(memory_format=torch.channels_last).eval()

    rng = np.random.default_rng(args.seed)
    scans = load_scans(os.path.join(args.data_root, "train/train"))
    wanted = requested_fractions(scans, args.size, args.count, args.boost, rng)
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "masks"), exist_ok=True)
    g = torch.Generator(device).manual_seed(args.seed)
    got = []
    for start in range(0, args.count, args.batch_size):
        fraction = torch.tensor(wanted[start:start + args.batch_size], dtype=torch.float32, device=device)
        image, mask = from_planes(sample(net, fraction, (args.size, args.size), steps=args.steps, generator=g))
        for k, (im, mk) in enumerate(zip(image.cpu().numpy(), mask.cpu().numpy())):
            name = f"S{start + k:05d}"
            cv2.imwrite(os.path.join(args.out, "images", f"{name}_image.png"), np.repeat(im[..., None], 3, 2))
            cv2.imwrite(os.path.join(args.out, "masks", f"{name}_mask.png"),
                        cv2.cvtColor(index_to_rgb(mk), cv2.COLOR_RGB2BGR))
            got.append((np.bincount(mk.ravel(), minlength=NUM_CLASSES) / mk.size).tolist())
        print(f"sampled {min(start + args.batch_size, args.count)}/{args.count}", flush=True)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"ckpt": args.ckpt, "ckpt_step": ckpt["step"], "held_out_loss": ckpt["held_out_loss"],
                   "args": vars(args), "requested_fractions": wanted.tolist(), "sampled_fractions": got}, f)
    err = np.abs(np.array(got) - wanted).mean(0)
    print(f"wrote {args.count} pairs to {args.out}; mean |sampled - requested| fraction per phase: "
          + ", ".join(f"{e:.3f}" for e in err))


if __name__ == "__main__":
    main()
