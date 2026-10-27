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

Per-scan standardisation and instance paste (round 5):
    --tone_norm 5 70                  same gray levels for Si and Al in every scan
    --scale_by_height 800:2           small scans enlarged more than the rest
    --prisi_paste_p 0.25              move a Primary Si particle within its scan

Phase-conditional intensity (round 7, on top of cs2_aug3):
    --al3ni_darken_p 0.3              darken some Al3Ni regions to the Eutectic Si level

One augmentation at a time (run_aug_table.sh, team protocol: batch 4, 12000 steps, seed 42):
    --max_steps 12000                 stop after this many optimizer steps
    --geo_aug none|hflip|vflip|rot90|d4      one geometric transform, or none
    --color_aug none|brightness|contrast|gamma|noise     one photometric transform, or none
    --crop_mode grid                  fixed tiles instead of random crops
    --class_ratio 0                   plain random crops
    --focus_ratio 0.5                 crops centred on one phase (--focus_class, Eutectic Si)

Synthetic pairs from the joint diffusion model (round 10, run_diffusion.sh):
    --synth_dir synth/d1 --synth_ratio 0.3    a share of the crops comes from synthetic pairs
    --synth_until 0.67                        ... only in the first two thirds of the epochs
    --synth_ignore_edge 1                     ... and without labels within 1 px of a phase edge

Decoder attention (round 9):
    --decoder_attention scse          scSE in every U-Net++ decoder block (Roy+ MICCAI18)

Auxiliary head (round 8, team experiment E3, needs --fullres_skip):
    --edge_w 0.05                     binary class-boundary head on the shared feature,
                                      class-balanced BCE; training only
"""
import argparse
import hashlib
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
from augmentations import (attenuate_connected_si, context_paste, darken_al3ni, fourier_style,
                           illumination_field, particle_bank, paste_particle, phase_levels,
                           weak_blur)
from losses import (ActiveBoundaryLoss, BoundaryDoULoss, JaccardMetricLoss, balanced_bce,
                    boundary_distance, boundary_smoothed_targets, class_boundary,
                    interface_weights, relaxed_ce, signed_distance, soft_ce, surface_loss,
                    weighted_ce)
from models import build, load_encoder_checkpoint, load_transfer_checkpoint
from group_cv import read_manifest
from preprocess import parse_scale_table, scan_scale, tone_normalize, tone_percentiles
from sampling import ClassAwareCropper, grid_crop, thin_density
from tiling import predict_tiled, scaled_tile

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IGNORE = 255  # mask value of pixels that carry no label (CE and Dice skip them)


class CropDataset(Dataset):
    """Full-resolution crops.

    Every image is held decoded in memory -- 70 scans is a few hundred MB and
    re-reading PNGs would dominate the epoch time.
    """

    def __init__(self, split_dir, tile=512, class_ratio=0.5, samples_per_image=8, seed=0,
                 scale_range=None, dist=None, input_scale=1.0, thin_ratio=0.0,
                 aug_d4=False, color_aug="legacy", color_p=0.5, cutmix_p=0.0,
                 illumination_p=0.0, blur_p=0.0, fourier_p=0.0,
                 context_paste_p=0.0, si_attenuate_p=0.0,
                 tone_norm=None, tone_target=None, scale_table=None, prisi_paste_p=0.0,
                 al3ni_darken_p=0.0, edge_width=None, geo_aug="legacy", crop_mode="random",
                 focus_class=2, focus_ratio=0.0, synth_dir=None, synth_ratio=0.0,
                 synth_ignore_edge=0.0, image_paths=None, teacher_prob_dir=None):
        import glob

        self.dist = dist
        self.cutmix_p = cutmix_p
        self.illumination_p = illumination_p
        self.blur_p = blur_p
        self.fourier_p = fourier_p
        self.context_paste_p = context_paste_p
        self.si_attenuate_p = si_attenuate_p
        self.prisi_paste_p = prisi_paste_p
        self.al3ni_darken_p = al3ni_darken_p
        self.edge_width = edge_width
        self.crop_mode = crop_mode
        self.synth_ratio = synth_ratio
        self.synth_ignore_edge = synth_ignore_edge
        self.input_scale = input_scale
        self.teacher_prob_dir = teacher_prob_dir
        self.mix_rng = np.random.default_rng(seed + 1)
        self.aug_rng = np.random.default_rng(seed + 701)
        self.aug_worker_seed = None

        self.paths = sorted(image_paths) if image_paths is not None else \
            sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no training images under {split_dir or 'manifest'}")
        raw = []
        for p in self.paths:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
            msk = rgb_to_index(cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
            teacher = None
            if teacher_prob_dir:
                name = os.path.basename(p).replace("_image.png", "_prob.npz")
                probability_path = os.path.join(teacher_prob_dir, name)
                if not os.path.isfile(probability_path):
                    raise FileNotFoundError(f"teacher probability missing for {p}: {probability_path}")
                with np.load(probability_path) as value:
                    probability = value["prob"].astype(np.float32)
                    source = str(value["image"].item()) if "image" in value.files else ""
                if probability.shape != (NUM_CLASSES, *msk.shape):
                    raise ValueError(f"teacher probability shape mismatch for {p}: "
                                     f"{probability.shape} vs {(NUM_CLASSES, *msk.shape)}")
                if source and os.path.basename(source) != os.path.basename(p):
                    raise ValueError(f"teacher probability source mismatch: {source} vs {p}")
                teacher = probability.transpose(1, 2, 0)
            raw.append((img, msk, teacher))
        # Tone standardisation sends every scan's two percentiles to the same
        # gray levels: the training set's own medians, so a typical scan is
        # left as it was. The levels travel with the checkpoint to inference.
        self.tone_target = tone_target
        if tone_norm and tone_target is None:
            points = np.array([tone_percentiles(img, tone_norm) for img, _, _ in raw])
            self.tone_target = [float(v) for v in np.median(points, 0)]
        # Synthetic pairs join the scans after the real ones: an epoch is still
        # samples_per_image crops per real scan, and synth_ratio of those crops
        # are redirected to a random synthetic pair (see __getitem__).
        self.n_real = len(raw)
        source_heights = [mask.shape[0] for _, mask, _ in raw]
        if synth_dir:
            synth = sorted(glob.glob(os.path.join(synth_dir, "images", "*.png")))
            if not synth:
                raise FileNotFoundError(f"no images under {synth_dir}/images")
            # Conditional samples retain their source scan's height. In cs2_plus,
            # a 512px patch from a large scan must not be mistaken for a small
            # scan and enlarged x4 solely because of the exported patch size.
            manifest_path = os.path.join(synth_dir, "manifest.jsonl")
            provenance = {}
            if os.path.isfile(manifest_path):
                with open(manifest_path) as f:
                    for line in f:
                        row = json.loads(line)
                        if row["name"] in provenance:
                            raise ValueError(f"duplicate synthetic manifest name: {row['name']}")
                        provenance[row["name"]] = int(row["source_height"])
            for p in synth:
                img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                mp = p.replace("/images/", "/masks/").replace("_image.png", "_mask.png")
                raw.append((img, rgb_to_index(cv2.cvtColor(
                    cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)), None))
                if os.path.isfile(manifest_path) and os.path.basename(p) not in provenance:
                    raise ValueError(f"synthetic image missing from manifest: {p}")
                height = provenance.get(os.path.basename(p), img.shape[0])
                if height <= 0:
                    raise ValueError(f"invalid source height for {p}: {height}")
                source_heights.append(height)
        self.images, self.masks, self.teachers, self.scales = [], [], [], []
        for (img, msk, teacher), source_height in zip(raw, source_heights):
            if tone_norm:
                img = tone_normalize(img, tone_norm, self.tone_target)
            # input_scale, or this scan's own scale when a height table is given.
            s = scan_scale(source_height, input_scale, scale_table)
            if s != 1:
                # Enlarge once at load -- the image linearly, the mask by
                # nearest neighbour so no in-between labels appear -- and a
                # 3 px lamella spans 3*s pixels inside the network.
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                msk = cv2.resize(msk, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                if teacher is not None:
                    teacher = cv2.resize(teacher, (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_LINEAR)
            self.images.append(img)
            self.masks.append(msk)
            # Float16 keeps 70 full-scan probability maps manageable in host
            # memory. They are promoted to float32 only for the sampled crop.
            self.teachers.append(None if teacher is None else teacher.astype(np.float16))
            self.scales.append(s)
        self.samples_per_image = samples_per_image
        # tile is in native pixels: at input_scale the crop keeps the same
        # field of view and carries s times as many pixels per side. With a
        # height table the crop stays this size in model pixels for every
        # scan, which is the point: structures then share one pixel size.
        self.cropper = ClassAwareCropper(scaled_tile(tile, input_scale), NUM_CLASSES, class_ratio,
                                         np.random.default_rng(seed),
                                         scale_range=scale_range,
                                         focus_class=focus_class, focus_ratio=focus_ratio)
        # Thin-structure crops pick the scan in proportion to its share of thin
        # Eutectic Si (1-17% across scans) and centre where it packs densest.
        # Centring alone barely helps: a 512 crop covers most of a scan, and the
        # best 512 window holds only 6.8% thin pixels against 5.7% at random.
        self.thin_ratio = thin_ratio
        self.thin_maps, self.thin_p = [None] * len(self.masks), None
        if thin_ratio > 0:
            self.thin_maps = [thin_density(m, half_width=2.0 * s, window=int(round(64 * s)))
                              for m, s in zip(self.masks, self.scales)]
            share = np.array([0.0 if t is None else float(t[0].mean()) for t in self.thin_maps])
            self.thin_p = share / share.sum()
        self.banks = None
        if prisi_paste_p > 0:
            self.banks = [particle_bank(m, scale=s) for m, s in zip(self.masks, self.scales)]
        self.levels = None
        if al3ni_darken_p > 0:
            self.levels = [phase_levels(img, m) for img, m in zip(self.images, self.masks)]
        # Geometric ops only; anything that resamples the mask must stay
        # nearest-neighbour, so colour jitter is kept separate from warps.
        geometric = {
            "legacy": [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)],
            "none": [],
            "hflip": [A.HorizontalFlip(p=0.5)],
            "vflip": [A.VerticalFlip(p=0.5)],
            "rot90": [A.RandomRotate90(p=1.0)],  # 0 / 90 / 180 / 270 degrees, uniformly
            "d4": [A.D4(p=1.0)],
        }["d4" if aug_d4 else geo_aug]
        # One photometric transform on its own, at the legacy strength and with
        # probability color_p. Noise is the exception: the legacy GaussNoise()
        # default is sigma 20-44% of the gray range, per channel, so the
        # single-transform test uses a small, channel-shared sigma of 2-5%.
        single = {
            "brightness": lambda: A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.0, p=color_p),
            "contrast": lambda: A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.15, p=color_p),
            "gamma": lambda: A.RandomGamma(gamma_limit=(90, 110), p=color_p),
            "noise": lambda: A.GaussNoise(std_range=(0.02, 0.05), per_channel=False, p=color_p),
        }
        if color_aug == "none":
            colour = []
        elif color_aug in single:
            colour = [single[color_aug]()]
        elif color_aug == "oneof":
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
        additional = ({"teacher": "mask", "teacher_valid": "mask"}
                      if teacher_prob_dir else None)
        self.aug = A.Compose(geometric + colour + [A.Normalize(mean=MEAN, std=STD)],
                             additional_targets=additional)

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
        if self.synth_ratio > 0 and self._augmentation_rng().random() < self.synth_ratio:
            j = self.n_real + int(self._augmentation_rng().integers(len(self.images) - self.n_real))
        # The extra RNG draws happen only when an option is on, so default
        # training sees exactly the crop sequence it always did.
        teacher = self.teachers[j]
        if self.crop_mode == "grid":
            cropped = grid_crop(self.images[j], self.masks[j], self.cropper.tile,
                                i % self.samples_per_image, teacher=teacher)
        elif self.thin_ratio > 0 and self.mix_rng.random() < self.thin_ratio:
            j = int(self.mix_rng.choice(len(self.images), p=self.thin_p))
            teacher = self.teachers[j]
            cropped = self.cropper(self.images[j], self.masks[j], self.thin_maps[j],
                                   teacher=teacher)
        else:
            cropped = self.cropper(self.images[j], self.masks[j], teacher=teacher)
        if teacher is None:
            img, msk = cropped
            teacher_valid = None
        else:
            img, msk, teacher = cropped
            teacher = teacher.astype(np.float32)
            teacher_valid = np.ones(msk.shape, np.uint8)
        if (self.prisi_paste_p > 0 and self.banks[j]
                and self._augmentation_rng().random() < self.prisi_paste_p):
            before = img.copy() if teacher is not None else None
            img, msk = paste_particle(img, msk, self.images[j], self.masks[j], self.banks[j],
                                      self._augmentation_rng())
            if teacher is not None:
                # The cached teacher saw the clean scan, not the pasted
                # particle. Keep supervised learning there but exclude those
                # changed pixels from KL distillation.
                teacher_valid[np.any(img != before, axis=2)] = 0
        if (self.al3ni_darken_p > 0 and self.levels[j] is not None
                and self._augmentation_rng().random() < self.al3ni_darken_p):
            img = darken_al3ni(img, msk, self.levels[j], self._augmentation_rng(),
                               scale=self.scales[j])
        if self.cutmix_p > 0 and self.mix_rng.random() < self.cutmix_p:
            img, msk = self._cutmix(img, msk)
        if self.context_paste_p > 0 and self._augmentation_rng().random() < self.context_paste_p:
            rng = self._augmentation_rng()
            k = self._donor_index(j, rng)
            donor_img, donor_msk = self.cropper(self.images[k], self.masks[k])
            img, msk = context_paste(img, msk, donor_img, donor_msk, rng,
                                     scale=self.scales[j])
        if self.illumination_p > 0 and self._augmentation_rng().random() < self.illumination_p:
            img = illumination_field(img, self._augmentation_rng(), grid=96 * self.scales[j])
        if self.blur_p > 0 and self._augmentation_rng().random() < self.blur_p:
            img = weak_blur(img, self._augmentation_rng(), scale=self.scales[j])
        if self.fourier_p > 0 and self._augmentation_rng().random() < self.fourier_p:
            rng = self._augmentation_rng()
            k = self._donor_index(j, rng)
            donor_img, _ = self.cropper(self.images[k], self.masks[k])
            img = fourier_style(img, donor_img, rng)
        if self.si_attenuate_p > 0 and self._augmentation_rng().random() < self.si_attenuate_p:
            img = attenuate_connected_si(img, msk, self._augmentation_rng(),
                                         scale=self.scales[j])
        if teacher is None:
            out = self.aug(image=img, mask=msk)
        else:
            out = self.aug(image=img, mask=msk, teacher=teacher,
                           teacher_valid=teacher_valid)
        image = torch.from_numpy(out["image"].transpose(2, 0, 1).copy())
        label = out["mask"].copy()
        if self.synth_ignore_edge > 0 and j >= self.n_real:
            # A generated mask sits less exactly on its image than an annotated one
            # (the real-trained models disagree with it within 1 px 35% more often),
            # so the edge band of a synthetic crop carries no label: the pair still
            # teaches which phase is where, not where exactly it ends.
            r = max(1, int(round(self.synth_ignore_edge * self.scales[j])))
            label[class_boundary(label, r) > 0] = IGNORE
        item = {"image": image, "mask": torch.from_numpy(label).long()}
        if teacher is not None:
            item["teacher"] = torch.from_numpy(
                out["teacher"].transpose(2, 0, 1).copy()).float()
            item["teacher_valid"] = torch.from_numpy(out["teacher_valid"].copy()).float()
            item["teacher_boundary"] = torch.from_numpy(class_boundary(out["mask"], 1))
        # The distance-map losses need exact EDTs of the augmented mask; the
        # loader workers build them so the GPU step does not wait on the CPU.
        if self.dist == "sdf":
            item["dist"] = torch.from_numpy(signed_distance(out["mask"], NUM_CLASSES))
        elif self.dist == "boundary":
            item["dist"] = torch.from_numpy(boundary_distance(out["mask"]))
        if self.edge_width:
            # edge_width is in native pixels; the band is r model pixels to each side.
            r = max(1, int(round(self.edge_width * self.scales[j] / 2)))
            item["edge"] = torch.from_numpy(class_boundary(out["mask"], r))
        return item


class FullImageSet(Dataset):
    """Whole scans, for scoring the way the competition does."""

    def __init__(self, split_dir, with_mask=True, tone_norm=None, tone_target=None,
                 image_paths=None):
        import glob

        self.paths = sorted(image_paths) if image_paths is not None else \
            sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
        if not self.paths:
            raise FileNotFoundError(f"no validation images under {split_dir or 'manifest'}")
        self.with_mask = with_mask
        self.tone_norm, self.tone_target = tone_norm, tone_target

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        if self.tone_norm:
            img = tone_normalize(img, self.tone_norm, self.tone_target)
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
        # IGNORE pixels drop out of both the intersection and the cardinality.
        keep = (target != IGNORE)[:, None]
        prob = logits.softmax(1) * keep
        onehot = F.one_hot(target.clamp(max=NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2).float() * keep
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        card = prob.sum(dims) + onehot.sum(dims)
        return (1 - (2 * inter + self.eps) / (card + self.eps)).mean()


def distillation_kl(logits, teacher, valid, boundary, temperature=2.0, confidence=0.5):
    """Return full-crop and boundary KL from cached teacher probabilities."""
    teacher = teacher.float().clamp_min(1e-7)
    teacher = teacher / teacher.sum(1, keepdim=True).clamp_min(1e-7)
    certainty = teacher.max(1).values
    # A cached ensemble probability has no logits. Raising p to 1/T and
    # renormalising is exactly softmax(log(p) / T).
    softened = teacher.pow(1.0 / temperature)
    softened = softened / softened.sum(1, keepdim=True).clamp_min(1e-7)
    per_pixel = F.kl_div(
        F.log_softmax(logits.float() / temperature, 1), softened,
        reduction="none").sum(1) * temperature ** 2
    confidence_weight = ((certainty - confidence) / (1.0 - confidence)).clamp(0, 1)
    weight = valid.float() * confidence_weight
    full = (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)
    boundary_weight = weight * boundary.float()
    on_boundary = ((per_pixel * boundary_weight).sum()
                   / boundary_weight.sum().clamp_min(1.0))
    return full, on_boundary


@torch.no_grad()
def validate(model, ds, device, tile, overlap, flip_tta=False, scale=1.0, scale_table=None):
    """Per-image mIoU over full scans, averaged -- the competition's formula."""
    model.eval()
    scores, per_class, rows = [], [], []
    for i in range(len(ds)):
        item = ds[i]
        image = item["image"][None].to(device)
        gt = item["mask"].numpy()
        prob = predict_tiled(model, image, tile=scaled_tile(tile, scale), overlap=overlap,
                             num_classes=NUM_CLASSES, flip_tta=flip_tta,
                             scale=scan_scale(image.shape[-2], scale, scale_table))
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
    ap.add_argument("--train_manifest", default=None,
                    help="JSON/text image manifest; overrides data_root/train/train")
    ap.add_argument("--valid_manifest", default=None,
                    help="JSON/text image manifest; overrides data_root/valid/valid")
    ap.add_argument("--no_validation", action="store_true",
                    help="train for the fixed epoch budget on all manifest images and save final.pth")
    ap.add_argument("--teacher_prob_dir", default=None,
                    help="cached full-scan teacher probabilities (*_prob.npz) for offline KD")
    ap.add_argument("--kd_weight", type=float, default=0.0,
                    help="weight of full-crop KL distillation (suggested 0.3)")
    ap.add_argument("--kd_boundary_w", type=float, default=0.0,
                    help="additional KL weight on the 2-pixel GT boundary band (suggested 0.1)")
    ap.add_argument("--kd_temperature", type=float, default=2.0)
    ap.add_argument("--kd_confidence", type=float, default=0.5,
                    help="ignore teacher pixels whose maximum class probability is below this")
    ap.add_argument("--arch", default="unetpp")
    ap.add_argument("--encoder", default="efficientnet-b3")
    ap.add_argument("--pretrain", default="micronet",
                    choices=["micronet", "image-micronet", "matssl", "imagenet", "none"])
    ap.add_argument("--encoder_checkpoint", default=None,
                    help="strict encoder-only checkpoint from cross-microscope SSL")
    ap.add_argument("--init_checkpoint", default=None,
                    help="segmentation pretrain; load its body and reset the class heads")
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
    ap.add_argument("--decoder_attention", default=None, choices=["scse"],
                    help="scSE: concurrent spatial and channel squeeze-and-excitation in every "
                         "decoder block, on the skip-concatenated input and on the output "
                         "(unet / unetpp)")
    ap.add_argument("--aug_d4", action="store_true",
                    help="one of the 8 rotations/reflections, uniformly, instead of "
                         "independent flips + rot90")
    ap.add_argument("--geo_aug", default="legacy",
                    choices=["legacy", "none", "hflip", "vflip", "rot90", "d4"],
                    help="legacy: flips + rot90, each with p=0.5; or a single transform / none "
                         "(--aug_d4 is the same as d4)")
    ap.add_argument("--color_aug", default="legacy",
                    choices=["legacy", "oneof", "none", "brightness", "contrast", "gamma", "noise"],
                    help="legacy: brightness/contrast +-15%%, gamma, noise; oneof: one of "
                         "brightness +-10%%, contrast 0.9-1.1, gamma 0.9-1.1; none; or a single "
                         "transform with probability --color_p (noise: sigma 2-5%%)")
    ap.add_argument("--crop_mode", default="random", choices=["random", "grid"],
                    help="grid: the same fixed tiles of each scan every epoch, i.e. no crop "
                         "augmentation")
    ap.add_argument("--focus_ratio", type=float, default=0.0,
                    help="share of crops centred on a pixel of --focus_class (e.g. 0.5)")
    ap.add_argument("--focus_class", type=int, default=2,
                    help="phase the focus crops centre on: 0 Al, 1 Al3Ni, 2 Eutectic Si, "
                         "3 Primary Si")
    ap.add_argument("--synth_dir", default=None,
                    help="folder of synthetic pairs (images/, masks/) from sample_diffusion.py")
    ap.add_argument("--synth_ratio", type=float, default=0.0,
                    help="share of training crops taken from --synth_dir instead of a real scan "
                         "(e.g. 0.3); the number of crops per epoch does not change")
    ap.add_argument("--synth_until", type=float, default=1.0,
                    help="synthetic crops only in this first share of the epochs (e.g. 0.67); the "
                         "remaining epochs train on real scans alone")
    ap.add_argument("--synth_ignore_edge", type=float, default=0.0,
                    help="leave the pixels of synthetic crops within this many native pixels of a "
                         "phase edge unlabelled (e.g. 1); needs --loss ce_dice with plain CE")
    ap.add_argument("--max_steps", type=int, default=None,
                    help="stop after this many optimizer steps instead of after --epochs: the "
                         "epoch count becomes ceil(max_steps / steps per epoch) and the last "
                         "epoch is cut short")
    ap.add_argument("--color_p", type=float, default=0.5, help="probability of the oneof step")
    ap.add_argument("--input_scale", type=float, default=1.0,
                    help="enlarge scans by this factor for training and inference (1, 1.5, 2); "
                         "--tile stays in native pixels, so the field of view is unchanged")
    ap.add_argument("--tone_norm", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                    help="map these two percentiles of every scan onto fixed gray levels, in "
                         "training and inference (e.g. 5 70: the Si tail and the Al matrix)")
    ap.add_argument("--scale_by_height", nargs="+", default=None, metavar="H:S",
                    help="per-scan scale: scans up to H native pixels tall use S instead of "
                         "--input_scale, e.g. 800:2 (whole numbers only; the crop stays "
                         "tile x input_scale model pixels)")
    ap.add_argument("--prisi_paste_p", type=float, default=0.0,
                    help="move one Primary Si particle with its matrix ring inside the same "
                         "scan (e.g. 0.25)")
    ap.add_argument("--al3ni_darken_p", type=float, default=0.0,
                    help="darken half of the Al3Ni regions of a crop toward the scan's Eutectic "
                         "Si gray level, image only (e.g. 0.3)")
    ap.add_argument("--edge_w", type=float, default=0.0,
                    help="weight of an auxiliary class-boundary BCE from a 1x1 edge head on the "
                         "fused full-resolution feature (needs --fullres_skip). The team's E3 used "
                         "0.1 next to CE + Dice summed; next to this script's 0.5 CE + 0.5 Dice "
                         "the same balance is 0.05")
    ap.add_argument("--edge_width", type=float, default=2.0,
                    help="width of the edge target band in native pixels (E3: about 2)")
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
    if args.encoder_checkpoint and args.init_checkpoint:
        ap.error("--encoder_checkpoint and --init_checkpoint are alternative initialisations")
    if args.no_validation and args.valid_manifest:
        ap.error("--no_validation and --valid_manifest cannot be used together")
    if bool(args.teacher_prob_dir) != bool(args.kd_weight > 0 or args.kd_boundary_w > 0):
        ap.error("--teacher_prob_dir and a positive --kd_weight/--kd_boundary_w go together")
    if args.kd_weight < 0 or args.kd_boundary_w < 0:
        ap.error("KD weights must be non-negative")
    if args.kd_temperature <= 0:
        ap.error("--kd_temperature must be positive")
    if not 0 <= args.kd_confidence < 1:
        ap.error("--kd_confidence must be in [0, 1)")
    if args.teacher_prob_dir and (args.synth_dir or args.cutmix_p > 0 or args.context_paste_p > 0):
        ap.error("offline KD does not support synth/cutmix/context-paste because cached teacher "
                 "maps do not contain the inserted content")
    if (args.encoder_checkpoint or args.init_checkpoint) and args.pretrain != "none":
        ap.error("external checkpoints replace the initialisation; use --pretrain none")
    if args.aux_w is None:
        args.aux_w = {"kervadec": 0.01, "abl": 0.5}.get(args.aux_loss, 0.0)
    if (args.synth_ratio > 0) != bool(args.synth_dir):
        ap.error("--synth_dir and --synth_ratio go together")
    if not 0 < args.synth_until <= 1:
        ap.error("--synth_until is a share of the epochs, in (0, 1]")
    if args.synth_ignore_edge > 0 and (args.loss != "ce_dice" or args.ce_mode != "hard" or args.bls_eps > 0
                                       or args.aux_loss != "none" or args.edge_w > 0):
        ap.error("--synth_ignore_edge works with --loss ce_dice, plain CE and no auxiliary loss only")
    if args.edge_w > 0 and not args.fullres_skip:
        ap.error("--edge_w puts its head on the fullres_skip fused feature; add --fullres_skip")
    if args.bls_eps > 0 and args.ce_mode != "hard":
        ap.error("--bls_eps and --ce_mode relax/interface both rewrite the CE term; pick one")
    scale_table = parse_scale_table(args.scale_by_height)
    for name in ("illumination_p", "blur_p", "fourier_p", "context_paste_p", "si_attenuate_p",
                 "prisi_paste_p", "al3ni_darken_p", "focus_ratio", "synth_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            ap.error(f"--{name} must be between 0 and 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_paths = read_manifest(args.train_manifest, args.data_root) if args.train_manifest else None
    valid_paths = read_manifest(args.valid_manifest, args.data_root) if args.valid_manifest else None
    for name in ("train_manifest", "valid_manifest"):
        path = getattr(args, name)
        if path:
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            setattr(args, f"{name}_sha256", digest.hexdigest())
    train_ds = CropDataset(os.path.join(args.data_root, "train/train"), args.tile,
                           args.class_ratio, args.samples_per_image, args.seed,
                           scale_range=tuple(args.scale_aug) if args.scale_aug else None,
                           dist={"kervadec": "sdf", "abl": "boundary"}.get(args.aux_loss),
                           input_scale=args.input_scale, thin_ratio=args.thin_ratio,
                           aug_d4=args.aug_d4, color_aug=args.color_aug, color_p=args.color_p,
                           cutmix_p=args.cutmix_p, illumination_p=args.illumination_p,
                           blur_p=args.blur_p, fourier_p=args.fourier_p,
                           context_paste_p=args.context_paste_p,
                           si_attenuate_p=args.si_attenuate_p,
                           tone_norm=args.tone_norm, scale_table=scale_table,
                           prisi_paste_p=args.prisi_paste_p,
                           al3ni_darken_p=args.al3ni_darken_p,
                           edge_width=args.edge_width if args.edge_w > 0 else None,
                           geo_aug=args.geo_aug, crop_mode=args.crop_mode,
                           focus_class=args.focus_class, focus_ratio=args.focus_ratio,
                           synth_dir=args.synth_dir, synth_ratio=args.synth_ratio,
                           synth_ignore_edge=args.synth_ignore_edge, image_paths=train_paths,
                           teacher_prob_dir=args.teacher_prob_dir)
    # Saved with the checkpoint, so validation and predict.py standardise alike.
    args.tone_target = train_ds.tone_target
    valid_ds = None if args.no_validation else FullImageSet(
        os.path.join(args.data_root, "valid/valid"), tone_norm=args.tone_norm,
        tone_target=args.tone_target, image_paths=valid_paths)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    if args.max_steps:
        # The cosine schedule and the validation cadence stay per epoch.
        args.epochs = -(-args.max_steps // len(train_dl))

    model = build(args.arch, args.encoder, args.pretrain, freeze_encoder=args.freeze_encoder,
                  fullres_skip=args.fullres_skip, fullres_ch=args.fullres_ch,
                  edge_head=args.edge_w > 0,
                  decoder_attention=args.decoder_attention)
    initialisation = None
    if args.encoder_checkpoint:
        got = load_encoder_checkpoint(model.encoder, args.encoder_checkpoint)
        print(f"  external SSL checkpoint: {got} encoder tensors loaded")
        initialisation = {"kind": "encoder", "path": os.path.abspath(args.encoder_checkpoint),
                          "loaded_tensors": got}
    if args.init_checkpoint:
        got, total = load_transfer_checkpoint(model, args.init_checkpoint)
        print(f"  segmentation transfer checkpoint: {got}/{total} body tensors loaded; heads reset")
        initialisation = {"kind": "segmentation_body", "path": os.path.abspath(args.init_checkpoint),
                          "loaded_tensors": got, "target_tensors": total,
                          "reset_prefixes": ["segmentation_head.", "edge_head."]}
    if initialisation:
        digest = hashlib.sha256()
        with open(initialisation["path"], "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        initialisation["sha256"] = digest.hexdigest()
        args.initialization_sha256 = initialisation["sha256"]
        with open(os.path.join(args.out, "initialization.json"), "w") as f:
            json.dump(initialisation, f, indent=2)
    model = model.to(device)

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
    ce, dice = nn.CrossEntropyLoss(ignore_index=IGNORE), DiceLoss()
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
          f"decoder_attention={args.decoder_attention} tile={args.tile} "
          f"crops/epoch={len(train_ds)}")
    print(f"data: train={len(train_ds.paths)} scans"
          + (" | validation=disabled (fixed-budget final fit)" if valid_ds is None
             else f" | valid={len(valid_ds.paths)} scans"))
    print(f"loss={args.loss} ce_mode={args.ce_mode} bls_eps={args.bls_eps} "
          f"aux={args.aux_loss}(w={args.aux_w}) fullres_skip={args.fullres_skip} "
          f"edge_w={args.edge_w} (band {args.edge_width:g} native px)")
    print(f"budget: {args.epochs} epochs x {len(train_dl)} steps"
          + (f", stopping at step {args.max_steps}" if args.max_steps else "")
          + f" | batch {args.batch_size} | crop_mode={args.crop_mode} class_ratio={args.class_ratio} "
          f"focus_ratio={args.focus_ratio} (class {args.focus_class})")
    if args.synth_dir:
        print(f"synthetic: {len(train_ds.images) - train_ds.n_real} pairs from {args.synth_dir}, "
              f"{100 * args.synth_ratio:.0f}% of the crops for the first {100 * args.synth_until:.0f}% of the epochs, "
              f"edge band without labels: {args.synth_ignore_edge:g} px")
    print(f"aug: geo={'d4' if args.aug_d4 else args.geo_aug} color={args.color_aug} input_scale={args.input_scale} "
          f"(crop {scaled_tile(args.tile, args.input_scale)} px) thin_ratio={args.thin_ratio} "
          f"cutmix_p={args.cutmix_p}")
    print(f"standardise: tone_norm={args.tone_norm} -> {args.tone_target} "
          f"scale_by_height={args.scale_by_height} | prisi_paste_p={args.prisi_paste_p}")
    print(f"new aug p: illumination={args.illumination_p} blur={args.blur_p} "
          f"fourier={args.fourier_p} context_paste={args.context_paste_p} "
          f"si_attenuate={args.si_attenuate_p}")
    if args.teacher_prob_dir:
        print(f"distillation: teacher={args.teacher_prob_dir} kd_w={args.kd_weight} "
              f"boundary_w={args.kd_boundary_w} T={args.kd_temperature} "
              f"confidence>={args.kd_confidence}")
    print(f"encoder params {sum(p.numel() for p in enc_params)/1e6:.1f}M @ lr {enc_lr:g} | "
          f"head {sum(p.numel() for p in head_params)/1e6:.1f}M @ lr {args.lr:g}")

    best, history, step = -1.0, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        if args.synth_ratio > 0:
            # The loader's workers are started anew every epoch, so they see this.
            train_ds.synth_ratio = args.synth_ratio if epoch <= args.synth_until * args.epochs else 0.0
        edge_stats = np.zeros(3)  # edge loss sum, intersection, union on training crops
        kd_stats = np.zeros(2)  # unweighted full-crop and boundary KL sums
        for batch in train_dl:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            dist = batch["dist"].to(device, non_blocking=True) if "dist" in batch else None
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(image)
                if isinstance(logits, tuple):
                    logits, edge_logits = logits
                loss = total_loss(logits, target, dist, epoch)
                if "teacher" in batch:
                    kd_full, kd_boundary = distillation_kl(
                        logits, batch["teacher"].to(device, non_blocking=True),
                        batch["teacher_valid"].to(device, non_blocking=True),
                        batch["teacher_boundary"].to(device, non_blocking=True),
                        args.kd_temperature, args.kd_confidence)
                    loss = (loss + args.kd_weight * kd_full
                            + args.kd_boundary_w * kd_boundary)
                    kd_stats += [kd_full.item(), kd_boundary.item()]
                if args.edge_w > 0:
                    edge = batch["edge"].to(device, non_blocking=True)
                    edge_loss = balanced_bce(edge_logits[:, 0], edge)
                    loss = loss + args.edge_w * edge_loss
                    with torch.no_grad():
                        hit, gt_edge = edge_logits[:, 0] > 0, edge > 0.5
                        edge_stats += [edge_loss.item(), (hit & gt_edge).sum().item(),
                                       (hit | gt_edge).sum().item()]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item()
            seen += 1
            step += 1
            if args.max_steps and step >= args.max_steps:
                break
        sched.step()

        line = f"epoch {epoch:3d}/{args.epochs}  loss {running/seen:.4f}"
        if args.synth_ratio > 0:
            line += f"  synth {100 * train_ds.synth_ratio:.0f}%"
        if args.edge_w > 0:
            line += (f"  edge_bce {edge_stats[0]/seen:.3f}"
                     f" edge_IoU(train) {100*edge_stats[1]/max(edge_stats[2], 1):.1f}%")
        if args.teacher_prob_dir:
            line += f"  kd {kd_stats[0]/seen:.4f} boundary_kd {kd_stats[1]/seen:.4f}"
        if valid_ds is not None and (epoch % args.val_every == 0 or epoch == args.epochs):
            miou, pc, rows = validate(model, valid_ds, device, args.tile, args.overlap,
                                      scale=args.input_scale, scale_table=scale_table)
            # Per-image scores at every evaluation, not only the best one: the
            # best checkpoint is picked on these same 20 scans, and
            # split_half_check.py needs them to pick on half and score the rest.
            history.append({"epoch": epoch, "miou": miou,
                            "per_image": {r["name"]: r["miou"] for r in rows}})
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

    if valid_ds is None:
        final_path = os.path.join(args.out, "final.pth")
        torch.save({"model": model.state_dict(), "args": vars(args), "miou": None,
                    "epoch": args.epochs}, final_path)
        best_value = None
        print(f"\nfixed-budget final checkpoint: {final_path}")
    else:
        best_value = best
        print(f"\nbest val mIoU: {100*best:.2f}%  (competition reference ~75%)")
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"best_miou": best_value, "history": history, "args": vars(args)}, f, indent=2)


if __name__ == "__main__":
    main()
