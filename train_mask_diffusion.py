"""Train mask-conditioned image synthesis using only train/train.

Select a step on an internal, grouped train holdout; then refit all 70 training
scans to that step. The official validation/test images are never opened here.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from komap import index_to_rgb
from mask_diffusion import MaskUNet, loss, sample
from mask_diffusion_data import PatchStream, crop, read_train, split_records, tensors

MODEL_KEYS = ("base", "geometry", "style", "dropout")
RECIPE_KEYS = MODEL_KEYS + ("patch", "lr", "warmup", "ema", "p_mean", "p_std",
                            "boundary_weight", "thin_weight", "batch_size", "accum")


def atomic_save(value, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, examples, device):
    total = 0.0
    for i, batch in enumerate(examples):
        batch = {k: v[None].to(device) for k, v in batch.items()}
        rng = torch.Generator(device).manual_seed(8123 + i)
        noise = torch.randn(batch["image"].shape, device=device, generator=rng)
        for s in (0.05, 0.2, 1.0, 5.0, 20.0):
            # Compare checkpoint denoising with the same unweighted objective.
            total += loss(model, **batch, boundary_weight=0, thin_weight=0,
                          sigma=torch.tensor([s], device=device), noise=noise).item()
    return total / (len(examples) * 5)


@torch.no_grad()
def preview(model, examples, path, device):
    batch = {k: torch.stack([x[k] for x in examples[:2]]).to(device) for k in examples[0]}
    images = sample(model, batch["condition"], batch["style"], steps=24,
                    generator=torch.Generator(device).manual_seed(313))
    columns = []
    for i in range(len(images)):
        real = ((batch["image"][i, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        generated = images[i].cpu().numpy()
        mask = batch["condition"][i, :4].argmax(0).cpu().numpy()
        columns.append(np.concatenate([np.repeat(real[..., None], 3, 2),
                                       index_to_rgb(mask), np.repeat(generated[..., None], 3, 2)], 0))
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(columns, 1), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", default=str(ROOT))
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["select", "refit"], default="select")
    ap.add_argument("--selection", help="select/best.pt: imports recipe and selected step for refit")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--patch", type=int, default=512)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--geometry", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--style", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.9999)
    ap.add_argument("--p_mean", type=float, default=-0.4)
    ap.add_argument("--p_std", type=float, default=1.4)
    ap.add_argument("--boundary_weight", type=float, default=0.5)
    ap.add_argument("--thin_weight", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--preview_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=17)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = ap.parse_args()
    if args.mode == "refit":
        if not args.selection:
            ap.error("refit requires --selection")
        selected = torch.load(args.selection, map_location="cpu", weights_only=False)
        if selected["args"]["mode"] != "select":
            ap.error("selection must come from an internal-holdout selection run")
        for key in RECIPE_KEYS:
            setattr(args, key, selected["args"][key])
        args.steps = selected["step"]
    if (args.patch < 32 or args.patch % 8 or min(args.steps, args.batch_size, args.accum,
            args.eval_every, args.log_every, args.warmup) < 1 or args.preview_every < 0):
        ap.error("invalid patch size / step / interval / batch configuration")
    if (args.hours is not None and args.hours <= 0) or args.workers < 0:
        ap.error("hours must be positive and workers must be nonnegative")
    if not 0 <= args.ema < 1 or min(args.boundary_weight, args.thin_weight) < 0 or args.lr <= 0:
        ap.error("invalid EMA / loss weights / learning rate")
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA unavailable; use --device cpu only for a small smoke test")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        ap.error("output already has a run; use --resume or a new --out")
    if args.resume and not (out / "last.pt").exists():
        ap.error("--resume requires out/last.pt")
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    records = read_train(args.data_root)
    training, heldout = split_records(records, args.split_seed)
    if args.mode == "refit":
        expected = {(r["name"], r["sha256"]) for r in selected["sources"]}
        if expected != {(r["name"], r["sha256"]) for r in records}:
            raise ValueError("train data differs from selection run")
        training, heldout = records, []
    sources = [dict(name=r["name"], sha256=r["sha256"], source_height=r["mask"].shape[0]) for r in records]
    split = dict(train=[r["name"] for r in training], holdout=[r["name"] for r in heldout],
                 group_rule="A<number> filename prefix; not verified specimen identity")
    with open(out / "split.json", "w") as f:
        json.dump(dict(sources=sources, **split), f, indent=2)
    net = MaskUNet(**{k: getattr(args, k) for k in MODEL_KEYS}).to(device)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0)
    start_step, best, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        for key in RECIPE_KEYS + ("seed", "split_seed", "mode", "steps"):
            if state["args"][key] != getattr(args, key):
                raise ValueError(f"resume recipe differs: {key}")
        if state["sources"] != sources:
            raise ValueError("resume data differs")
        net.load_state_dict(state["net"])
        ema.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best, history = state["step"], state["best_loss"], state["history"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    # Worker sampling restarts with a different seed on resume; optimizer/EMA
    # and torch RNG are restored, but bit-exact data-order replay is not claimed.
    loader = iter(DataLoader(PatchStream(training, args.patch, args.seed + start_step * 101),
                             batch_size=args.batch_size, num_workers=args.workers,
                             pin_memory=device.type == "cuda"))
    rng = np.random.default_rng(121)
    examples = [tensors(*crop(r, args.patch, rng, orient=False)[:2], r["style"])
                for r in (heldout or training[:2])]
    print(f"mode={args.mode} train={len(training)} inner_holdout={len(heldout)} "
          f"patch={args.patch} params={sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"effective_batch={args.batch_size * args.accum} steps={args.steps}", flush=True)
    started, running, tick = time.monotonic(), 0.0, time.monotonic()
    for step in range(start_step + 1, args.steps + 1):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * min(1, step / args.warmup)
        step_loss = 0.0
        for _ in range(args.accum):
            batch = {k: v.to(device, non_blocking=True) for k, v in next(loader).items()}
            value = loss(net, **batch, p_mean=args.p_mean, p_std=args.p_std,
                         boundary_weight=args.boundary_weight, thin_weight=args.thin_weight)
            if not torch.isfinite(value):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            (value / args.accum).backward()
            step_loss += value.detach().item() / args.accum
        torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            decay = min(args.ema, (1 + step) / (10 + step))
            for e, p in zip(ema.parameters(), net.parameters()):
                e.lerp_(p, 1 - decay)
        running += step_loss
        if step % args.log_every == 0:
            print(f"step={step}/{args.steps} train={running / args.log_every:.4f} "
                  f"seconds={time.monotonic() - tick:.1f}", flush=True)
            running, tick = 0.0, time.monotonic()
        timed_out = args.hours is not None and time.monotonic() - started >= args.hours * 3600
        if step % args.eval_every == 0 or step == args.steps or timed_out:
            val = evaluate(ema, examples, device) if heldout else None
            history.append(dict(step=step, train_loss=step_loss, inner_loss=val))
            improved = val is not None and val < best
            if improved:
                best = val
            snapshot = dict(model=ema.state_dict(), args=vars(args), step=step,
                            inner_loss=val, sources=sources, split=split)
            if improved or not heldout:
                atomic_save(snapshot, out / "best.pt")
            atomic_save(dict(**snapshot, net=net.state_dict(), optimizer=optimizer.state_dict(),
                             best_loss=best, history=history, rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else []),
                        out / "last.pt")
            with open(out / "history.json", "w") as f:
                json.dump(dict(args=vars(args), history=history), f, indent=2)
            print(f"checkpoint step={step} inner_loss={val} best={improved}", flush=True)
            if args.preview_every and (step % args.preview_every == 0 or step == args.steps or timed_out):
                preview(ema, examples, out / f"preview_{step:06d}.png", device)
            if timed_out and step < args.steps:
                print("time budget reached; checkpoint saved, resume supported", flush=True)
                if args.mode == "select":
                    (out / "complete.json").write_text(json.dumps(dict(step=step, mode=args.mode,
                                                                         stop_reason="time_budget")))
                break
    else:
        (out / "complete.json").write_text(json.dumps(dict(step=args.steps, mode=args.mode)))


if __name__ == "__main__":
    main()
