"""Train the joint image-mask diffusion model on the segmentation training scans.

The model only ever sees train/train -- the scans the segmentation models are
trained on. Random 256 px patches at native resolution, each with one of the 8
rotations/reflections (micrographs have no orientation, so this is a true
symmetry and does not leak into the samples).

valid/valid is used for one thing: the denoising loss of the EMA weights on
fixed patches, noise levels and noise draws, measured every --eval_every steps.
It is the held-out counterpart of the training loss, so it shows when the model
starts to memorise 70 scans, and the checkpoint with the lowest value is kept
as best.pt (what sample_diffusion.py uses). A small sample sheet is written at
every evaluation.

    ./.conda/bin/python train_diffusion.py --out runs/diffusion_d1
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
from diffusion import NUM_CLASSES, UNet, diffusion_loss, from_planes, sample, to_planes
from komap import index_to_rgb, rgb_to_index


def load_scans(split_dir):
    scans = []
    for p in sorted(glob.glob(os.path.join(split_dir, "images", "*.png"))):
        mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
        scans.append((cv2.imread(p, cv2.IMREAD_GRAYSCALE),
                      rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))))
    if not scans:
        raise FileNotFoundError(f"no images under {split_dir}/images")
    return scans


def fractions(planes):
    """Phase fractions of a patch from its one-hot planes: (..., 5, H, W) -> (..., 4)."""
    return (planes[..., 1:, :, :] > 0).float().mean((-2, -1))


class Patches(IterableDataset):
    """Endless random patches: a scan chosen uniformly (as the segmentation loader
    weighs them), a random window, one of the 8 orientations."""

    def __init__(self, scans, size, seed):
        self.scans, self.size, self.seed = scans, size, seed

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng(self.seed + (worker.id if worker else 0))
        while True:
            image, mask = self.scans[int(rng.integers(len(self.scans)))]
            y = int(rng.integers(0, mask.shape[0] - self.size + 1))
            x = int(rng.integers(0, mask.shape[1] - self.size + 1))
            image, mask = image[y:y + self.size, x:x + self.size], mask[y:y + self.size, x:x + self.size]
            k = int(rng.integers(8))
            if k >= 4:
                image, mask = image[:, ::-1], mask[:, ::-1]
            image, mask = np.rot90(image, k % 4), np.rot90(mask, k % 4)
            yield to_planes(np.ascontiguousarray(image), np.ascontiguousarray(mask))


def fixed_patches(scans, size, per_scan=4):
    """The same held-out patches every time: a 2 x 2 grid of windows per scan."""
    out = []
    for image, mask in scans:
        h, w = mask.shape
        for y in (0, h - size):
            for x in (0, w - size):
                out.append(to_planes(image[y:y + size, x:x + size], mask[y:y + size, x:x + size]))
    return torch.stack(out[:per_scan * len(scans)])


@torch.no_grad()
def held_out_loss(net, clean, sigmas, noise, batch=40):
    """Mean training-style loss over fixed patches x fixed noise levels, with fixed noise."""
    total = 0.0
    for sigma in sigmas:
        for i in range(0, clean.shape[0], batch):
            x = clean[i:i + batch]
            s = torch.full((x.shape[0],), float(sigma), device=x.device)
            total += diffusion_loss(net, x, fractions(x), sigma=s, noise=noise[i:i + batch]).item() * x.shape[0]
    return total / (len(sigmas) * clean.shape[0])


def sample_sheet(net, fraction, path, size, steps=18):
    """Images over their masks, for a look at what the model draws."""
    g = torch.Generator(fraction.device).manual_seed(0)
    image, mask = from_planes(sample(net, fraction, (size, size), steps=steps, generator=g))
    top = np.concatenate([np.repeat(i[..., None], 3, 2) for i in image.cpu().numpy()], 1)
    bottom = np.concatenate([index_to_rgb(m) for m in mask.cpu().numpy()], 1)
    cv2.imwrite(path, cv2.cvtColor(np.concatenate([top, bottom], 0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=ROOT)
    ap.add_argument("--out", default="runs/diffusion_d1")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--hours", type=float, default=None,
                    help="stop after this much training time if --steps is not reached first")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--fraction_drop", type=float, default=0.1,
                    help="share of patches trained without their phase fractions (zeros)")
    ap.add_argument("--p_mean", type=float, default=-0.4)
    ap.add_argument("--p_std", type=float, default=1.4)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True  # fixed input sizes
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_scans(os.path.join(args.data_root, "train/train"))
    valid = load_scans(os.path.join(args.data_root, "valid/valid"))
    loader = iter(DataLoader(Patches(train, args.patch, args.seed), batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=True))

    net = UNet(base=args.base, dropout=args.dropout).to(device).to(memory_format=torch.channels_last)
    ema = UNet(base=args.base, dropout=args.dropout).to(device).to(memory_format=torch.channels_last)
    ema.load_state_dict(net.state_dict())
    ema.requires_grad_(False).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    held_out = fixed_patches(valid, args.patch).to(device)
    g = torch.Generator(device).manual_seed(1234)
    held_noise = torch.randn(held_out.shape, device=device, generator=g)
    held_sigmas = np.exp(np.linspace(np.log(0.05), np.log(20), 8))
    sheet_fraction = fractions(held_out[::max(1, held_out.shape[0] // 6)][:6])

    print(f"train scans {len(train)}, held-out patches {held_out.shape[0]} from {len(valid)} valid scans")
    print(f"U-Net {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M parameters, patch {args.patch}, "
          f"batch {args.batch_size}, {args.steps} steps, lr {args.lr:g}, EMA {args.ema}")
    print(f"noise levels: ln sigma ~ N({args.p_mean}, {args.p_std}^2)", flush=True)

    history, best, running, seen, t0, start = [], float("inf"), 0.0, 0, time.time(), time.time()
    for step in range(1, args.steps + 1):
        clean = next(loader).to(device, non_blocking=True)
        fraction = fractions(clean)
        fraction[torch.rand(fraction.shape[0], device=device) < args.fraction_drop] = 0
        for group in opt.param_groups:
            group["lr"] = args.lr * min(1.0, step / args.warmup)
        net.train()
        loss = diffusion_loss(net, clean, fraction, args.p_mean, args.p_std)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        seen += 1
        # EMA with a warm-up, so the early average is not stuck at the initialisation.
        decay = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for e, p in zip(ema.parameters(), net.parameters()):
                e.lerp_(p, 1 - decay)

        out_of_time = args.hours is not None and time.time() - start > args.hours * 3600
        if step % args.eval_every == 0 or step == args.steps or out_of_time:
            val = held_out_loss(ema, held_out, held_sigmas, held_noise)
            history.append({"step": step, "train_loss": running / seen, "held_out_loss": val})
            line = (f"step {step:6d}/{args.steps}  train {running / seen:.4f}  "
                    f"held-out {val:.4f}  ({time.time() - t0:.0f}s)")
            ckpt = {"model": ema.state_dict(), "args": vars(args), "step": step, "held_out_loss": val}
            torch.save(ckpt, os.path.join(args.out, "last.pt"))
            if val < best:
                best = val
                torch.save(ckpt, os.path.join(args.out, "best.pt"))
                line += "  <- best"
            print(line, flush=True)
            sample_sheet(ema, sheet_fraction, os.path.join(args.out, f"samples_{step:06d}.png"), args.patch)
            with open(os.path.join(args.out, "history.json"), "w") as f:
                json.dump({"history": history, "args": vars(args)}, f, indent=1)
            running, seen, t0 = 0.0, 0, time.time()
            if out_of_time:
                print(f"stopping at step {step}: {args.hours:g} h of training reached")
                break
    print(f"\nbest held-out loss {best:.4f}")


if __name__ == "__main__":
    main()
