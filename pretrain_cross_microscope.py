"""Adapt MicroNet EfficientNet-B3 with same-ROI cross-microscope contrastive learning."""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from external_pretraining import (CrossMicroscopePairs, ProjectionHead, file_sha256,
                                  supervised_contrastive_loss)
from models import build


class ContrastiveEncoder(nn.Module):
    def __init__(self, encoder, channels, hidden=512, projection=128):
        super().__init__()
        self.encoder = encoder
        self.projection = ProjectionHead(channels, hidden, projection)

    def forward(self, image):
        return self.projection(self.encoder(image)[-1])


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/external_cross_ssl")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--pretrain", default="micronet", choices=["micronet", "imagenet", "none"])
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=24,
                    help="ROI draws per epoch; default gives about 5k optimizer steps in 50 epochs")
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--target_ratio", type=float, default=0.25,
                    help="share of sampled SSL groups drawn from KoMaP train images")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="CPU smoke-test escape hatch")
    args = ap.parse_args()
    if not 0 < args.target_ratio < 1:
        ap.error("--target_ratio must be in (0, 1)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    komap = sorted(glob.glob(str(args.data_root / "train/train/images/*.png")))
    if not komap:
        raise FileNotFoundError(f"no KoMaP training images under {args.data_root}")
    train_ds = CrossMicroscopePairs(args.manifest, "train", args.size, args.repeats,
                                    args.seed, komap_images=komap)
    valid_ds = CrossMicroscopePairs(args.manifest, "valid", args.size, 1, args.seed + 1)

    # Weighted replacement fixes the domain mix independently of how many
    # acquisition variants the external release contains.
    n_groups = len(train_ds.groups)
    komap_groups = np.array([name.startswith("komap:") for name, _ in train_ds.groups])
    n_target, n_external = int(komap_groups.sum()), int((~komap_groups).sum())
    if not n_target or not n_external:
        raise ValueError("SSL training needs both external ROI groups and KoMaP train images")
    group_weight = np.where(komap_groups, args.target_ratio / n_target,
                            (1 - args.target_ratio) / n_external)
    weights = np.array([group_weight[i % n_groups] for i in range(len(train_ds))])
    sampler = WeightedRandomSampler(torch.from_numpy(weights), len(train_ds), replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                          num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=device.type == "cuda")
    if not len(train_dl):
        raise ValueError("SSL batch is larger than the training dataset")

    segmentation_model = build("unetpp", args.encoder, args.pretrain)
    encoder = segmentation_model.encoder
    channels = encoder.out_channels[-1]
    del segmentation_model
    model = ContrastiveEncoder(encoder, channels).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": args.encoder_lr},
        {"params": model.projection.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dl)
    warmup = min(5 * len(train_dl), max(1, total_steps // 4))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best, step = [], float("inf"), 0

    def batch_loss(batch):
        a = batch["view1"].to(device, non_blocking=True)
        b = batch["view2"].to(device, non_blocking=True)
        labels = batch["group"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            embeddings = torch.cat((model(a), model(b)))
            return supervised_contrastive_loss(embeddings, torch.cat((labels, labels)),
                                               args.temperature)

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        losses, start = [], time.time()
        for batch in train_dl:
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
            step += 1
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        validation = []
        with torch.no_grad():
            for batch in valid_dl:
                validation.append(float(batch_loss(batch)))
        train_loss, valid_loss = float(np.mean(losses)), float(np.mean(validation))
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        print(f"epoch {epoch:3d}/{args.epochs} train {train_loss:.4f} valid {valid_loss:.4f} "
              f"lr {optimizer.param_groups[0]['lr']:.2g} ({time.time()-start:.0f}s)", flush=True)
        checkpoint = {"format": "komap_encoder_v1", "encoder": model.encoder.state_dict(),
                      "projection": model.projection.state_dict(), "args": vars(args),
                      "manifest_sha256": file_sha256(args.manifest),
                      "external_train_groups": n_external, "komap_train_groups": n_target,
                      "valid_groups": len(valid_ds.groups), "epoch": epoch,
                      "valid_loss": valid_loss}
        atomic_save(checkpoint, args.out / "last_encoder.pth")
        if valid_loss < best:
            best = valid_loss
            atomic_save(checkpoint, args.out / "best_encoder.pth")
        if args.max_steps and step >= args.max_steps:
            break

    serialisable = vars(args).copy()
    serialisable.update(manifest=str(args.manifest), data_root=str(args.data_root), out=str(args.out))
    (args.out / "history.json").write_text(json.dumps(
        {"best_valid_loss": best, "history": history, "args": serialisable,
         "external_train_groups": n_external, "komap_train_groups": n_target,
         "valid_groups": len(valid_ds.groups)}, indent=2))


if __name__ == "__main__":
    main()
