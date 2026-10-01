"""Shared pieces for the KoMaP phase-segmentation task: data, metric, model.

The competition scores each evaluation image by mIoU over the 4 phases and
averages that over the 10 images, so the metric here is deliberately
per-image-then-average rather than dataset-pooled.
"""
import glob
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Phase palette, read off the provided masks. Index order is the class index
# used everywhere downstream.
CLASSES = ["Al", "Al3Ni", "EutecticSi", "PrimarySi"]
COLORS = np.array(
    [
        [153, 127, 76],   # 0 Al            (brown)
        [25, 76, 153],    # 1 Al3Ni         (blue)
        [204, 204, 204],  # 2 Eutectic Si   (white)
        [76, 178, 76],    # 3 Primary Si    (green)
    ],
    dtype=np.uint8,
)
NUM_CLASSES = len(CLASSES)


def rgb_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Map an RGB mask to class indices by nearest palette colour.

    The masks are saved as PNG and come back with exactly the four palette
    colours, but nearest-colour keeps this correct if a file is ever re-encoded
    with slight interpolation instead of failing silently on an exact match.
    """
    flat = mask_rgb.reshape(-1, 3).astype(np.int16)
    d = ((flat[:, None, :] - COLORS[None, :, :].astype(np.int16)) ** 2).sum(-1)
    return d.argmin(1).reshape(mask_rgb.shape[:2]).astype(np.uint8)


def index_to_rgb(idx: np.ndarray) -> np.ndarray:
    return COLORS[idx]


class KomapDataset(Dataset):
    """Image/mask pairs. `split_dir` is the folder holding images/ and masks/."""

    def __init__(self, split_dir, transform=None, with_mask=True):
        self.image_paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.png")))
        if not self.image_paths:
            raise FileNotFoundError(f"no images under {split_dir}/images")
        self.with_mask = with_mask
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def mask_path(self, image_path):
        return image_path.replace("/images/", "/masks/").replace("_image.png", "_mask.png")

    def __getitem__(self, i):
        ip = self.image_paths[i]
        # The scans are stored RGBA but are visually grayscale; IMREAD_COLOR
        # drops the alpha channel and gives the 3 channels the
        # ImageNet-pretrained encoder expects.
        image = cv2.imread(ip, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {ip}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.with_mask:
            mp = self.mask_path(ip)
            mask_rgb = cv2.cvtColor(cv2.imread(mp, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = rgb_to_index(mask_rgb)
        else:
            mask = np.zeros(image.shape[:2], np.uint8)

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]

        # With a ToTensorV2 transform these are already tensors; without one
        # (handy for inspecting raw data) fall back to numpy conversion.
        if not torch.is_tensor(mask):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        if not torch.is_tensor(image):
            image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return {
            "image": image,
            "mask": mask.long(),
            "name": os.path.basename(ip),
        }


def per_image_miou(pred: np.ndarray, target: np.ndarray, num_classes=NUM_CLASSES):
    """mIoU for one image, plus the per-class IoUs.

    A class that is absent from both prediction and ground truth is undefined
    (0/0) and is excluded from the mean rather than counted as 0 -- two of the
    provided masks contain no Primary Si at all, and scoring those as zero
    would understate the result by ~25 points on that image.
    """
    ious = np.full(num_classes, np.nan)
    for c in range(num_classes):
        p, t = pred == c, target == c
        union = np.logical_or(p, t).sum()
        if union == 0:
            continue
        ious[c] = np.logical_and(p, t).sum() / union
    return float(np.nanmean(ious)), ious


def build_model(arch="unet", encoder="resnet34", weights="imagenet"):
    import segmentation_models_pytorch as smp

    factory = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3p": smp.DeepLabV3Plus}
    if arch not in factory:
        raise ValueError(f"unknown arch {arch}; pick from {sorted(factory)}")
    return factory[arch](
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=NUM_CLASSES,
    )
