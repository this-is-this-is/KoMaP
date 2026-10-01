"""Train phase-segmentation models with native-resolution crops.

Strategy: learn on 512x512 crops taken at the scan's own pixel scale, then
score whole images with overlapping tiles. Shrinking a full frame to 512 alone
caps the achievable score near 67% on this data -- the thin Eutectic Si
lamellae do not survive the downsample -- so the crop/tile route is what makes
the 75% reference reachable.

Examples:
    ./.conda/bin/python train_v2.py --arch segformer_b2 --out runs/segformer_b2
    ./.conda/bin/python train_v2.py --arch unetpp --encoder efficientnet-b3 \
        --pretrain micronet --out runs/micronet_unetpp
    ./.conda/bin/python train_v2.py --arch dinov2 --enc_lr 1e-5 --lr 1e-4 \
        --out runs/dinov2_eomt

Boundary experiments (run_boundary_suite.sh runs all of them, 3 seeds each):
    --fullres_skip                    stride-1 branch before the head
    --ce_mode relax                   boundary label relaxation (Zhu+ CVPR19)
    --ce_mode interface               Al3Ni-Si interface weights (Strohmann+ 2019)
    --loss ce_jml [--bls_eps 0.5]     Jaccard metric loss [+ boundary label smoothing]
    --loss ce_bdou                    Boundary DoU (Sun+ MICCAI23)
    --aux_loss kervadec               boundary (surface) loss (Kervadec+ MIDL19)
    --loss ce_lovasz --aux_loss abl   Active Boundary Loss (Wang+ AAAI22)

Augmentation / sampling (round 3, on top of --fullres_skip):
    --aug_d4                          one of the 8 rotations/reflections, uniformly
    --color_aug oneof                 one of brightness +-10%, contrast/gamma 0.9-1.1
    --input_scale 1.5 | 2             enlarge scans for training and inference
    --thin_ratio 0.3                  30% of crops centred on thin, dense Eutectic Si
    --cutmix_p 0.2                    CutMix
"""
import argparse
import json
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import LovaszLoss
from torch.utils.data import DataLoader, Dataset, get_worker_info

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from komap import CLASSES, NUM_CLASSES, rgb_to_index, per_image_miou
from augmentations import (attenuate_connected_si, context_paste, fourier_style,
                           illumination_field, weak_blur)
from losses import (ActiveBoundaryLoss, BoundaryDoULoss, JaccardMetricLoss, boundary_distance,
                    boundary_smoothed_targets, interface_weights, relaxed_ce, signed_distance,
                    soft_ce, surface_loss, weighted_ce)
from models import build
from sampling import ClassAwareCropper, thin_density
from tiling import predict_tiled, scaled_tile

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class CropDataset(Dataset):
    """Full-resolution crops.

    Every image is held decoded in memory -- 70 scans is a few hundred MB and
    re-reading PNGs would dominate the epoch time.
    """

    def __init__(self, split_dir, tile=512, class_ratio=0.5, samples_per_image=8, seed=0,
                 scale_range=None, dist=None, input_scale=1.0, thin_ratio=0.0,
                 aug_d4=False, color_aug="legacy", color_p=0.5, cutmix_p=0.0,
                 illumination_p=0.0, blur_p=0.0, fourier_p=0.0,
                 context_paste_p=0.0, si_attenuate_p=0.0):
        import glob

        self.dist = dist
        self.cutmix_p = cutmix_p
        self.illumination_p = illumination_p
        self.blur_p = blur_p
        self.fourier_p = fourier_p
        self.context_paste_p = context_paste_p
        self.si_attenuate_p = si_attenuate_p
        self.input_scale = input_scale
        self.mix_rng = np.random.default_rng(seed + 1)
        self.aug_rng = np.random.default_rng(seed + 701)
        self.aug_worker_seed = None

        self.paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no images under {split_dir}/images")
        self.images, self.masks = [], []
        for p in self.paths:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
            msk = rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            if input_scale != 1:
                # Enlarge once at load -- the image linearly, the mask by
                # nearest neighbour so no in-between labels appear -- and a
                # 3 px lamella spans 3*s pixels inside the network.
                img = cv2.resize(img, None, fx=input_scale, fy=input_scale,
                                 interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self.images.append(img)
            self.masks.append(msk)
        self.samples_per_image = samples_per_image
        # tile is in native pixels: at input_scale the crop keeps the same
        # field of view and carries s times as many pixels per side.
        self.cropper = ClassAwareCropper(scaled_tile(tile, input_scale), NUM_CLASSES, class_ratio,
                                         np.random.default_rng(seed),
                                         scale_range=scale_range)
        # Thin-structure crops pick the scan in proportion to its share of thin
        # Eutectic Si (1-17% across scans) and centre where it packs densest.
        # Centring alone barely helps: a 512 crop covers most of a scan, and the
        # best 512 window holds only 6.8% thin pixels against 5.7% at random.
        self.thin_ratio = thin_ratio
        self.thin_maps, self.thin_p = [None] * len(self.masks), None
        if thin_ratio > 0:
            self.thin_maps = [thin_density(m, half_width=2.0 * input_scale,
                                           window=int(round(64 * input_scale)))
                              for m in self.masks]
            share = np.array([0.0 if t is None else float(t[0].mean()) for t in self.thin_maps])
            self.thin_p = share / share.sum()
        # Geometric ops only; anything that resamples the mask must stay
        # nearest-neighbour, so colour jitter is kept separate from warps.
        geometric = [A.D4(p=1.0)] if aug_d4 else [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ]
        if color_aug == "oneof":
            # One mild photometric change at a time: brightness +-10%,
            # contrast x0.9-1.1 or gamma 0.9-1.1.
            colour = [A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.0, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.1, p=1.0),
                A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            ], p=color_p)]
        else:
            colour = [
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3),
                A.GaussNoise(p=0.15),
            ]
        self.aug = A.Compose(geometric + colour + [A.Normalize(mean=MEAN, std=STD)])

    def __len__(self):
        return len(self.paths) * self.samples_per_image

    def _augmentation_rng(self):
        # DataLoader forks its dataset into workers. Give the new transforms
        # independent worker streams; otherwise every worker repeats choices.
        worker = get_worker_info()
        if worker is not None and self.aug_worker_seed != worker.seed:
            self.aug_rng = np.random.default_rng(worker.seed)
            self.aug_worker_seed = worker.seed
        return self.aug_rng

    def _donor_index(self, source_index, rng):
        if len(self.images) == 1:
            return 0
        k = int(rng.integers(len(self.images) - 1))
        return k + int(k >= source_index)

    def _cutmix(self, img, msk):
        """Paste a box from another random crop (Yun et al., ICCV 2019).

        The box covers a (1 - lam) share of the crop with lam ~ U(0, 1), and
        the mask is pasted with it, so labels are never blended.
        """
        k = int(self.mix_rng.integers(len(self.images)))
        img2, msk2 = self.cropper(self.images[k], self.masks[k])
        t = msk.shape[0]
        cut = int(t * np.sqrt(1.0 - self.mix_rng.uniform()))
        cy, cx = self.mix_rng.integers(0, t, size=2)
        y0, y1 = np.clip([cy - cut // 2, cy + cut // 2], 0, t)
        x0, x1 = np.clip([cx - cut // 2, cx + cut // 2], 0, t)
        # The cropper returns views into the cached scans; copy before pasting.
        img, msk = img.copy(), msk.copy()
        img[y0:y1, x0:x1] = img2[y0:y1, x0:x1]
        msk[y0:y1, x0:x1] = msk2[y0:y1, x0:x1]
        return img, msk

    def __getitem__(self, i):
        j = i // self.samples_per_image
        # The extra RNG draws happen only when an option is on, so default
        # training sees exactly the crop sequence it always did.
        if self.thin_ratio > 0 and self.mix_rng.random() < self.thin_ratio:
            j = int(self.mix_rng.choice(len(self.images), p=self.thin_p))
            img, msk = self.cropper(self.images[j], self.masks[j], self.thin_maps[j])
        else:
            img, msk = self.cropper(self.images[j], self.masks[j])
        if self.cutmix_p > 0 and self.mix_rng.random() < self.cutmix_p:
            img, msk = self._cutmix(img, msk)
        if self.context_paste_p > 0 and self._augmentation_rng().random() < self.context_paste_p:
            rng = self._augmentation_rng()
            k = self._donor_index(j, rng)
            donor_img, donor_msk = self.cropper(self.images[k], self.masks[k])
            img, msk = context_paste(img, msk, donor_img, donor_msk, rng,
                                     scale=self.input_scale)
        if self.illumination_p > 0 and self._augmentation_rng().random() < self.illumination_p:
            img = illumination_field(img, self._augmentation_rng(), grid=96 * self.input_scale)
        if self.blur_p > 0 and self._augmentation_rng().random() < self.blur_p:
            img = weak_blur(img, self._augmentation_rng(), scale=self.input_scale)
        if self.fourier_p > 0 and self._augmentation_rng().random() < self.fourier_p:
            rng = self._augmentation_rng()
            k = self._donor_index(j, rng)
            donor_img, _ = self.cropper(self.images[k], self.masks[k])
            img = fourier_style(img, donor_img, rng)
        if self.si_attenuate_p > 0 and self._augmentation_rng().random() < self.si_attenuate_p:
            img = attenuate_connected_si(img, msk, self._augmentation_rng(),
                                         scale=self.input_scale)
        out = self.aug(image=img, mask=msk)
        image = torch.from_numpy(out["image"].transpose(2, 0, 1).copy())
        item = {"image": image, "mask": torch.from_numpy(out["mask"].copy()).long()}
        # The distance-map losses need exact EDTs of the augmented mask; the
        # loader workers build them so the GPU step does not wait on the CPU.
        if self.dist == "sdf":
            item["dist"] = torch.from_numpy(signed_distance(out["mask"], NUM_CLASSES))
        elif self.dist == "boundary":
            item["dist"] = torch.from_numpy(boundary_distance(out["mask"]))
        return item


class FullImageSet(Dataset):
    """Whole scans, for scoring the way the competition does."""

    def __init__(self, split_dir, with_mask=True):
        import glob

        self.paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
        self.with_mask = with_mask

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = ((img / 255.0 - np.array(MEAN)) / np.array(STD)).transpose(2, 0, 1)
        item = {"image": torch.from_numpy(x).float(), "name": os.path.basename(p)}
        if self.with_mask:
            mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
            item["mask"] = torch.from_numpy(
                rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            ).long()
        return item


class DiceLoss(nn.Module):
    """Macro (class-averaged) soft Dice.

    Averaging over classes rather than pixels is what matches the metric: Al
    covers ~58% of pixels but only a quarter of the score.
    """

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        prob = logits.softmax(1)
        onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        card = prob.sum(dims) + onehot.sum(dims)
        return (1 - (2 * inter + self.eps) / (card + self.eps)).mean()


@torch.no_grad()
def validate(model, ds, device, tile, overlap, flip_tta=False, scale=1.0):
    """Per-image mIoU over full scans, averaged -- the competition's formula."""
    model.eval()
    scores, per_class, rows = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        image = item["image"][None].to(device)
        gt = item["mask"].numpy()
        prob = predict_tiled(model, image, tile=scaled_tile(tile, scale), overlap=overlap,
                             num_classes=NUM_CLASSES, flip_tta=flip_tta, scale=scale)
        pred = prob.argmax(0).cpu().numpy()
        miou, ious = per_image_miou(pred, gt)
        scores.append(miou)
        per_class.append(ious)
        rows.append({"name": item["name"], "miou": miou,
                     **{c: (None if np.isnan(v) else float(v)) for c, v in zip(CLASSES, ious)}})
    return float(np.mean(scores)), np.nanmean(np.stack(per_class), 0), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--arch", default="unetpp")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--pretrain", default="micronet",
                    choices=["micronet", "image-micronet", "imagenet", "none"])
    ap.add_argument("--freeze_encoder", action="store_true")
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--class_ratio", type=float, default=0.5)
    ap.add_argument("--loss", default="ce_dice",
                    choices=["ce_dice", "ce_lovasz", "ce_dice_lovasz", "ce_jml", "ce_bdou", "bdou"],
                    help="ce_dice is the baseline; the lovasz variants target thin phases; "
                         "ce_jml = 0.25 CE + 0.75 Jaccard metric loss (the JML paper's weights); "
                         "ce_bdou swaps Dice for Boundary DoU; bdou is B-DoU alone, as its authors train")
    ap.add_argument("--lovasz_per_image", action="store_true",
                    help="average Lovasz per image, as the competition metric does")
    ap.add_argument("--ce_mode", default="hard", choices=["hard", "relax", "interface"],
                    help="how the CE term treats boundaries: relax = label relaxation "
                         "(Zhu+ CVPR19); interface = Al3Ni-Si interface weights (Strohmann+ 2019)")
    ap.add_argument("--relax_k", type=int, default=3, help="label relaxation window")
    ap.add_argument("--interface_w", type=float, default=1.5,
                    help="weight of Al3Ni-Si interface pixels (paper: 1.5)")
    ap.add_argument("--si_w", type=float, default=1.25, help="weight of Si pixels (paper: 1.25)")
    ap.add_argument("--bls_eps", type=float, default=0.0,
                    help="boundary label smoothing of the CE/JML targets (paper best: 0.5)")
    ap.add_argument("--bls_k", type=int, default=3, help="BLS boundary window (paper best: 3)")
    ap.add_argument("--aux_loss", default="none", choices=["none", "kervadec", "abl"],
                    help="extra boundary term: kervadec = boundary (surface) loss; "
                         "abl = Active Boundary Loss")
    ap.add_argument("--aux_w", type=float, default=None,
                    help="abl: weight (default 0.5, the paper's 1.0 against CE+IoU at our 0.5 "
                         "scale); kervadec: per-epoch weight step (default 0.01, as the authors)")
    ap.add_argument("--fullres_skip", action="store_true",
                    help="stride-1 branch into the head (unet / unetpp)")
    ap.add_argument("--fullres_ch", type=int, default=32)
    ap.add_argument("--aug_d4", action="store_true",
                    help="one of the 8 rotations/reflections, uniformly, instead of "
                         "independent flips + rot90")
    ap.add_argument("--color_aug", default="legacy", choices=["legacy", "oneof"],
                    help="legacy: brightness/contrast +-15%%, gamma, noise; oneof: one of "
                         "brightness +-10%%, contrast 0.9-1.1, gamma 0.9-1.1")
    ap.add_argument("--color_p", type=float, default=0.5, help="probability of the oneof step")
    ap.add_argument("--input_scale", type=float, default=1.0,
                    help="enlarge scans by this factor for training and inference (1, 1.5, 2); "
                         "--tile stays in native pixels, so the field of view is unchanged")
    ap.add_argument("--thin_ratio", type=float, default=0.0,
                    help="share of crops centred on thin, dense Eutectic Si (e.g. 0.3)")
    ap.add_argument("--cutmix_p", type=float, default=0.0, help="CutMix probability (e.g. 0.2)")
    ap.add_argument("--illumination_p", type=float, default=0.0,
                    help="smooth image-only illumination field (suggested 0.2)")
    ap.add_argument("--blur_p", type=float, default=0.0,
                    help="weak image-only Gaussian PSF blur (suggested 0.15)")
    ap.add_argument("--fourier_p", type=float, default=0.0,
                    help="train-to-train low-frequency Fourier amplitude mixing (suggested 0.1)")
    ap.add_argument("--context_paste_p", type=float, default=0.0,
                    help="context-matched Si-rich image+mask box copy (suggested 0.1)")
    ap.add_argument("--si_attenuate_p", type=float, default=0.0,
                    help="conditional local contrast attenuation on connected Eutectic Si (suggested 0.05)")
    ap.add_argument("--scale_aug", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="random crop scale, e.g. --scale_aug 0.7 1.4. The scans come at "
                         "four magnifications; this spreads each image over that range.")
    ap.add_argument("--samples_per_image", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4, help="head / new parameters")
    ap.add_argument("--enc_lr", type=float, default=None,
                    help="encoder LR; defaults to --lr. Use ~1e-5 for ViT backbones.")
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=2)
    ap.add_argument("--out", default="runs/v2")
    args = ap.parse_args()
    if args.aux_w is None:
        args.aux_w = {"kervadec": 0.01, "abl": 0.5}.get(args.aux_loss, 0.0)
    if args.bls_eps > 0 and args.ce_mode != "hard":
        ap.error("--bls_eps and --ce_mode relax/interface both rewrite the CE term; pick one")
    for name in ("illumination_p", "blur_p", "fourier_p", "context_paste_p", "si_attenuate_p"):
        if not 0 <= getattr(args, name) <= 1:
            ap.error(f"--{name} must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = CropDataset(os.path.join(args.data_root, "train/train"), args.tile,
                           args.class_ratio, args.samples_per_image, args.seed,
                           scale_range=tuple(args.scale_aug) if args.scale_aug else None,
                           dist={"kervadec": "sdf", "abl": "boundary"}.get(args.aux_loss),
                           input_scale=args.input_scale, thin_ratio=args.thin_ratio,
                           aug_d4=args.aug_d4, color_aug=args.color_aug, color_p=args.color_p,
                           cutmix_p=args.cutmix_p, illumination_p=args.illumination_p,
                           blur_p=args.blur_p, fourier_p=args.fourier_p,
                           context_paste_p=args.context_paste_p,
                           si_attenuate_p=args.si_attenuate_p)
    valid_ds = FullImageSet(os.path.join(args.data_root, "valid/valid"))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)

    model = build(args.arch, args.encoder, args.pretrain, freeze_encoder=args.freeze_encoder,
                  fullres_skip=args.fullres_skip, fullres_ch=args.fullres_ch).to(device)

    # Pretrained features are worth protecting: give the backbone a smaller step
    # than the freshly initialised head, as EoMT does with its split warm-up.
    enc_lr = args.enc_lr if args.enc_lr is not None else args.lr
    enc_keys = ("backbone", "encoder", "net.segformer")
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(enc_keys)]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith(enc_keys)]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": enc_lr},
         {"params": head_params, "lr": args.lr}], weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce, dice = nn.CrossEntropyLoss(), DiceLoss()
    # Lovasz optimises the IoU surrogate directly rather than a pixel average,
    # which is what the thin phases need: Eutectic Si and the thinner Al3Ni
    # lamellae lose their score to a few boundary pixels, and Dice on a 7%-of-
    # pixels class barely moves. per_image matches the competition's averaging.
    lovasz = LovaszLoss(mode="multiclass", per_image=args.lovasz_per_image) \
        if "lovasz" in args.loss else None
    jml, bdou = JaccardMetricLoss(), BoundaryDoULoss()
    abl = ActiveBoundaryLoss() if args.aux_loss == "abl" else None

    def ce_term(logits, target, soft):
        if soft is not None:
            return soft_ce(logits, soft)
        if args.ce_mode == "relax":
            return relaxed_ce(logits, target, NUM_CLASSES, args.relax_k)
        if args.ce_mode == "interface":
            return weighted_ce(logits, target, interface_weights(
                target, NUM_CLASSES, args.interface_w, args.si_w))
        return ce(logits, target)

    def region_loss(logits, target):
        # BLS rewrites the targets of the CE and JML terms; Dice, Lovasz and
        # B-DoU cannot take soft labels and keep the hard mask.
        soft = boundary_smoothed_targets(target, NUM_CLASSES, args.bls_eps, args.bls_k) \
            if args.bls_eps > 0 else None
        if args.loss == "ce_dice":
            return 0.5 * ce_term(logits, target, soft) + 0.5 * dice(logits, target)
        if args.loss == "ce_lovasz":
            return 0.5 * ce_term(logits, target, soft) + 0.5 * lovasz(logits, target)
        if args.loss == "ce_dice_lovasz":
            return (0.34 * ce_term(logits, target, soft) + 0.33 * dice(logits, target)
                    + 0.33 * lovasz(logits, target))
        if args.loss == "ce_jml":
            y = soft if soft is not None else F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
            return 0.25 * ce_term(logits, target, soft) + 0.75 * jml(logits, y)
        if args.loss == "ce_bdou":
            return 0.5 * ce_term(logits, target, soft) + 0.5 * bdou(logits, target)
        return bdou(logits, target)

    def total_loss(logits, target, dist, epoch):
        loss = region_loss(logits, target)
        if args.aux_loss == "kervadec":
            # The authors' rebalancing: the boundary term's weight grows by
            # aux_w every epoch while the region terms give way (floor 0.1).
            return (max(0.1, 1 - args.aux_w * (epoch - 1)) * loss
                    + args.aux_w * epoch * surface_loss(logits, dist))
        if args.aux_loss == "abl":
            return loss + args.aux_w * abl(logits, dist)
        return loss

    print(f"arch={args.arch} encoder={args.encoder} pretrain={args.pretrain} "
          f"tile={args.tile} crops/epoch={len(train_ds)}")
    print(f"loss={args.loss} ce_mode={args.ce_mode} bls_eps={args.bls_eps} "
          f"aux={args.aux_loss}(w={args.aux_w}) fullres_skip={args.fullres_skip}")
    print(f"aug: d4={args.aug_d4} color={args.color_aug} input_scale={args.input_scale} "
          f"(crop {scaled_tile(args.tile, args.input_scale)} px) thin_ratio={args.thin_ratio} "
          f"cutmix_p={args.cutmix_p}")
    print(f"new aug p: illumination={args.illumination_p} blur={args.blur_p} "
          f"fourier={args.fourier_p} context_paste={args.context_paste_p} "
          f"si_attenuate={args.si_attenuate_p}")
    print(f"encoder params {sum(p.numel() for p in enc_params)/1e6:.1f}M @ lr {enc_lr:g} | "
          f"head {sum(p.numel() for p in head_params)/1e6:.1f}M @ lr {args.lr:g}")

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for batch in train_dl:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True) if "dist" in batch else None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
                loss = total_loss(logits, target, dist, epoch)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
        sched.step()

        line = f"epoch {epoch:3d}/{args.epochs}  loss {running/len(train_dl):.4f}"
        if epoch % args.val_every == 0 or epoch == args.epochs:
            miou, pc, rows = validate(model, valid_ds, device, args.tile, args.overlap,
                                      scale=args.input_scale)
            history.append({"epoch": epoch, "miou": miou})
            line += f"  val_mIoU {100*miou:.2f}%"
            if miou > best:
                best = miou
                torch.save({"model": model.state_dict(), "args": vars(args), "miou": miou},
                           os.path.join(args.out, "best.pth"))
                # Keep the full 20x4 table: it shows whether a change helped the
                # thin phases or only the easy bulk ones.
                with open(os.path.join(args.out, "val_per_image.json"), "w") as f:
                    json.dump({"mean_miou": miou, "per_image": rows}, f, indent=2)
                line += "  <- best"
                line += "\n            per-class IoU: " + "  ".join(
                    f"{c}={100*v:.1f}%" for c, v in zip(CLASSES, pc))
        print(line + f"   ({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"best_miou": best, "history": history, "args": vars(args)}, f, indent=2)
    print(f"\nbest val mIoU: {100*best:.2f}%  (competition reference ~75%)")


if __name__ == "__main__":
    main()
