"""Pretrain the complete KoMaP decoder on NASA's pixel-labelled Ni superalloy data."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from external_pretraining import NasaSuperDataset
from models import build


class MacroDice(nn.Module):
    def forward(self, logits, target):
        probability = logits.softmax(1)
        onehot = F.one_hot(target, 3).permute(0, 3, 1, 2).float()
        intersection = (probability * onehot).sum((0, 2, 3))
        cardinality = probability.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        return (1 - (2 * intersection + 1) / (cardinality + 1)).mean()


def mean_iou(prediction, target):
    values = []
    for label in range(3):
        a, b = prediction == label, target == label
        union = np.logical_or(a, b).sum()
        values.append(np.logical_and(a, b).sum() / union if union else np.nan)
    return float(np.nanmean(values)), values


def atomic_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nasa_root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/external_nasa_super")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--pretrain", default="micronet", choices=["micronet", "imagenet", "none"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--mode", choices=["select", "refit", "all"], default="all",
                    help="select on five OOD images, refit all 23 for selected epochs, or both")
    ap.add_argument("--samples_per_image", type=int, default=32)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="CPU smoke-test escape hatch")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ce, dice = nn.CrossEntropyLoss(), MacroDice()

    def new_model():
        # Refit starts again from the same MicroNet initialisation rather than
        # leaking the OOD epoch-selection images through a continued run.
        torch.manual_seed(args.seed)
        return build("unetpp", args.encoder, args.pretrain, num_classes=3,
                     fullres_skip=True, fullres_ch=32).to(device)

    def fit(split, epochs, validation=None, save_best=None, label="selection"):
        dataset = NasaSuperDataset(args.nasa_root, split, args.samples_per_image, args.seed,
                                   size=args.size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
        valid_loader = None
        if validation:
            valid_ds = NasaSuperDataset(args.nasa_root, validation, 1, args.seed,
                                        augment=False, size=args.size)
            valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False,
                                      num_workers=args.workers, pin_memory=device.type == "cuda")
        model = new_model()
        encoder_parameters = list(model.encoder.parameters())
        encoder_ids = {id(p) for p in encoder_parameters}
        optimizer = torch.optim.AdamW([
            {"params": encoder_parameters, "lr": args.encoder_lr},
            {"params": [p for p in model.parameters() if id(p) not in encoder_ids], "lr": args.lr},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        best, best_epoch, history, step = -1.0, epochs, [], 0
        for epoch in range(1, epochs + 1):
            dataset.set_epoch(epoch)
            model.train()
            losses, start = [], time.time()
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    logits = model(image)
                    loss = 0.5 * ce(logits, target) + 0.5 * dice(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach()))
                step += 1
                if args.max_steps and step >= args.max_steps:
                    break
            scheduler.step()
            row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
            validate = valid_loader is not None and (epoch % args.val_every == 0 or epoch == epochs
                                                      or (args.max_steps and step >= args.max_steps))
            detail = ""
            if validate:
                model.eval()
                scores, classes = [], []
                with torch.no_grad():
                    for batch in valid_loader:
                        target = batch["mask"].numpy()[0]
                        prediction = model(batch["image"].to(device)).argmax(1).cpu().numpy()[0]
                        score, per_class = mean_iou(prediction, target)
                        scores.append(score)
                        classes.append(per_class)
                score = float(np.mean(scores))
                row.update(valid_miou=score, valid_class_iou=np.nanmean(classes, axis=0).tolist())
                detail = f" OOD_mIoU {100*score:.2f}%"
                if score > best:
                    best, best_epoch = score, epoch
                    if save_best:
                        atomic_save({"model": model.state_dict(), "epoch": epoch,
                                     "valid_miou": score}, save_best)
            history.append(row)
            print(f"{label} epoch {epoch:3d}/{epochs} loss {row['train_loss']:.4f}{detail} "
                  f"({time.time()-start:.0f}s)", flush=True)
            if args.max_steps and step >= args.max_steps:
                if valid_loader is None:
                    best_epoch = epoch
                break
        return model, history, best, best_epoch

    history_path = args.out / "selection_history.json"
    if args.mode in ("select", "all"):
        _, selection_history, best, selected_epoch = fit(
            "select_train", args.epochs, validation="ood",
            save_best=args.out / "selection_best.pth", label="select")
        history_path.write_text(json.dumps({"best_ood_miou": best,
                                            "selected_epoch": selected_epoch,
                                            "history": selection_history}, indent=2))
    else:
        if not history_path.is_file():
            raise FileNotFoundError(f"run --mode select first: {history_path}")
        selected_epoch = int(json.loads(history_path.read_text())["selected_epoch"])

    if args.mode in ("refit", "all"):
        model, refit_history, _, _ = fit("all", selected_epoch, label="refit")
        payload = {"format": "komap_segmentation_transfer_v1", "model": model.state_dict(),
                   "epoch": selected_epoch,
                   "args": {"arch": "unetpp", "encoder": args.encoder,
                            "pretrain": args.pretrain, "fullres_skip": True,
                            "fullres_ch": 32, "decoder_attention": None,
                            "num_classes": 3, "source": "NASA Super unique 23-pair refit",
                            "selection": "18 source images / 5 OOD images",
                            "nasa_commit": "9b7c4abc1321e81eca7a68d548e5371676fa74fa"}}
        atomic_save(payload, args.out / "best.pth")
        serialisable = vars(args).copy()
        serialisable.update(nasa_root=str(args.nasa_root), out=str(args.out))
        (args.out / "history.json").write_text(json.dumps(
            {"selected_epoch": selected_epoch, "history": refit_history,
             "args": serialisable}, indent=2))


if __name__ == "__main__":
    main()
