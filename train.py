"""Train a phase-segmentation model for the KoMaP competition.

Scoring target (from the competition brief): mean over the 10 evaluation
images of the 4-class mIoU. Reference baseline is ~75%.

Example:
    ./.conda/bin/python train.py --epochs 80 --arch unet --encoder resnet34
"""
import argparse
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from komap import CLASSES, NUM_CLASSES, KomapDataset, build_model, per_image_miou

# ImageNet statistics, matching the pretrained encoder.
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(size, train, native=False):
    """Augmentation pipeline.

    Only 70 training images are available, so augmentation carries real weight
    here. Micrographs have no canonical orientation, which makes flips and
    90-degree rotations label-preserving; brightness/contrast jitter covers
    variation in illumination between scans.
    """
    if train:
        if native:
            # Crop a window straight out of the full-resolution scan instead of
            # shrinking the whole frame. Downsampling to 512 alone caps the
            # achievable score near 67% -- the thin Eutectic Si lamellae do not
            # survive it -- so native-resolution tiles raise that ceiling.
            return A.Compose([
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
                A.RandomCrop(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ])
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.35, 1.0), ratio=(0.85, 1.18)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    # Validation keeps the whole frame: the score is computed on full images,
    # so cropping here would measure something other than the target metric.
    if native:
        # Feed the scan at its own resolution, padded to a multiple of 32 so the
        # encoder's downsampling stages divide evenly.
        return A.Compose([
            # position="top_left" keeps the original pixels at (0,0) so the
            # padding can simply be sliced back off before scoring.
            A.PadIfNeeded(None, None, pad_height_divisor=32, pad_width_divisor=32,
                          position="top_left", border_mode=cv2.BORDER_REFLECT_101),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


class DiceLoss(nn.Module):
    """Soft multiclass Dice, which counters the strong class imbalance.

    Al covers ~58% of pixels while Eutectic Si covers ~7%, so a pure
    cross-entropy objective is dominated by the majority phase while the score
    weights all four phases equally.
    """

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        prob = logits.softmax(1)
        onehot = torch.nn.functional.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        card = prob.sum(dims) + onehot.sum(dims)
        return (1 - (2 * inter + self.eps) / (card + self.eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device, size):
    """Validate at full resolution, the way the competition scores."""
    model.eval()
    scores, per_class = [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        target = batch["mask"][0].numpy()

        logits = model(image)
        if logits.shape[-2:] != target.shape:
            # Native mode pads up to a multiple of 32, so take the top-left
            # region back off rather than rescaling; resize mode interpolates
            # back to the original grid.
            if logits.shape[-2] >= target.shape[0] and logits.shape[-1] >= target.shape[1]:
                logits = logits[..., : target.shape[0], : target.shape[1]]
            else:
                logits = torch.nn.functional.interpolate(
                    logits, size=target.shape, mode="bilinear", align_corners=False
                )
        pred = logits.argmax(1)[0].cpu().numpy()

        miou, ious = per_image_miou(pred, target)
        scores.append(miou)
        per_class.append(ious)
    return float(np.mean(scores)), np.nanmean(np.stack(per_class), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--arch", default="unet", choices=["unet", "unetpp", "deeplabv3p"])
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--native", action="store_true",
                    help="train on full-resolution crops and validate at native "
                         "resolution instead of resizing (raises the score ceiling)")
    ap.add_argument("--out", default="runs/unet_r34")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = KomapDataset(
        os.path.join(args.data_root, "train/train"),
        build_transforms(args.size, True, args.native),
    )
    # batch_size=1 for validation: images come in five different resolutions,
    # and each is scored on its own grid.
    valid_ds = KomapDataset(
        os.path.join(args.data_root, "valid/valid"),
        build_transforms(args.size, False, args.native),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=args.workers)

    model = build_model(args.arch, args.encoder).to(device)
    ce = nn.CrossEntropyLoss()
    dice = DiceLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    print(f"train={len(train_ds)} valid={len(valid_ds)} device={device} "
          f"arch={args.arch}/{args.encoder} size={args.size}")

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for batch in train_dl:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device, enabled=(device == "cuda")):
                logits = model(image)
                loss = ce(logits, target) + dice(logits, target)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        miou, per_class = evaluate(model, valid_dl, device, args.size)
        flag = ""
        if miou > best:
            best = miou
            torch.save({"model": model.state_dict(), "args": vars(args), "miou": miou},
                       os.path.join(args.out, "best.pth"))
            flag = "  <- best"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {running/max(len(train_dl),1):.4f}  "
              f"val_mIoU {100*miou:.2f}%  ({time.time()-t0:.0f}s){flag}")
        if flag:
            print("            per-class IoU: " + "  ".join(
                f"{c}={100*v:.1f}%" for c, v in zip(CLASSES, per_class)))

    print(f"\nbest val mIoU: {100*best:.2f}%  (competition reference ~75%)")
    print(f"checkpoint: {os.path.join(args.out, 'best.pth')}")


if __name__ == "__main__":
    main()
