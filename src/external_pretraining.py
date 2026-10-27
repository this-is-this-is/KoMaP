"""Datasets and losses for microscopy-specific transfer pretraining."""
from collections import defaultdict
import glob
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info


MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD = np.array((0.229, 0.224, 0.225), np.float32)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_fold(value, folds=10):
    raw = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big") % folds


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _d4(image, code):
    # Four rotations and their reflected versions, all exact index operations.
    if code >= 4:
        image = np.flip(image, axis=1)
    return np.ascontiguousarray(np.rot90(image, code % 4))


def _normalise(image):
    value = image.astype(np.float32) / 255.0
    return torch.from_numpy(((value - MEAN) / STD).transpose(2, 0, 1).copy())


def _shared_geometry(a, b, rng, size):
    # Identical ROIs stay aligned: the two acquisition variants receive the
    # same crop and D4 transform.  Every input is square in the released set.
    side = size + max(16, size // 8)
    a = cv2.resize(a, (side, side), interpolation=cv2.INTER_AREA if a.shape[0] > side else cv2.INTER_CUBIC)
    b = cv2.resize(b, (side, side), interpolation=cv2.INTER_AREA if b.shape[0] > side else cv2.INTER_CUBIC)
    top = int(rng.integers(side - size + 1))
    left = int(rng.integers(side - size + 1))
    code = int(rng.integers(8))
    return (_d4(a[top:top + size, left:left + size], code),
            _d4(b[top:top + size, left:left + size], code))


def _appearance(image, rng):
    # Operate in gray so microscope-specific colour casts cannot become an
    # easy shortcut.  The ranges match KoMaP's successful mild colour jitter.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    contrast = float(rng.uniform(0.9, 1.1))
    brightness = float(rng.uniform(-0.1, 0.1))
    gamma = float(rng.uniform(0.9, 1.1))
    gray = np.clip((gray - 0.5) * contrast + 0.5 + brightness, 0, 1) ** gamma
    if rng.random() < 0.2:
        gray = cv2.GaussianBlur(gray, (0, 0), float(rng.uniform(0.3, 0.7)))
    if rng.random() < 0.2:
        gray = gray + rng.normal(0, rng.uniform(0.005, 0.02), gray.shape)
    out = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(out[..., None], 3, axis=2)


class CrossMicroscopePairs(Dataset):
    """Same-ROI acquisition pairs plus target-domain KoMaP image pairs."""

    def __init__(self, manifest, split="train", size=320, repeats=4, seed=0,
                 komap_images=None):
        self.manifest = Path(manifest)
        root = self.manifest.parent
        rows = read_jsonl(self.manifest)
        groups = defaultdict(list)
        for row in rows:
            if row["split"] != split:
                continue
            path = Path(row["path"])
            if not path.is_absolute():
                path = root / path
            groups[str(row["roi_id"])].append(
                (path, str(row.get("microscope", "unknown")), str(row.get("magnification", "unknown"))))
        if split == "train" and komap_images:
            paths = (sorted(glob.glob(str(komap_images)))
                     if not isinstance(komap_images, (list, tuple)) else komap_images)
            for path in paths:
                path = Path(path)
                groups[f"komap:{path.stem}"].append((path, "komap", "komap"))
        self.groups = [(name, values) for name, values in sorted(groups.items()) if values]
        if not self.groups:
            raise ValueError(f"no {split} pairs in {manifest}")
        self.size, self.repeats, self.seed, self.epoch = size, repeats, seed, 0

    def __len__(self):
        return len(self.groups) * self.repeats

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, index):
        worker = get_worker_info()
        worker_seed = 0 if worker is None else worker.id * 1000003
        return np.random.default_rng(self.seed + self.epoch * 10000019 + index + worker_seed)

    def __getitem__(self, index):
        group_index = index % len(self.groups)
        name, variants = self.groups[group_index]
        rng = self._rng(index)
        first = int(rng.integers(len(variants)))
        # Prefer another microscope/configuration for a real cross-domain pair.
        # Prefer a different microscope at the same magnification.  These
        # released patches have the same pixel geometry and need no semantic
        # assumption about what a 20x-to-100x resize should preserve.
        choices = [i for i, (_, microscope, magnification) in enumerate(variants)
                   if microscope != variants[first][1] and magnification == variants[first][2]]
        if not choices:
            choices = [i for i, (_, _, magnification) in enumerate(variants)
                       if i != first and magnification == variants[first][2]]
        if not choices:
            choices = [i for i, (_, microscope, _) in enumerate(variants)
                       if microscope != variants[first][1]]
        if not choices:
            choices = [i for i in range(len(variants)) if i != first]
        second = int(rng.choice(choices)) if choices else first
        a, b = _read_rgb(variants[first][0]), _read_rgb(variants[second][0])
        a, b = _shared_geometry(a, b, rng, self.size)
        a, b = _appearance(a, rng), _appearance(b, rng)
        return {"view1": _normalise(a), "view2": _normalise(b),
                "group": group_index, "roi_id": name,
                "microscope1": variants[first][1], "microscope2": variants[second][1],
                "magnification1": variants[first][2], "magnification2": variants[second][2]}


def supervised_contrastive_loss(features, labels, temperature=0.1):
    """Multi-positive supervised contrastive loss using ROI identity as label."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be NxD and labels must be N")
    if len(features) < 2:
        raise ValueError("contrastive loss needs at least two embeddings")
    features = F.normalize(features.float(), dim=1)
    logits = features @ features.T / temperature
    eye = torch.eye(len(features), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    if not positives.any(1).all():
        raise ValueError("every embedding needs another view of the same ROI")
    logits = logits - logits.max(1, keepdim=True).values.detach()
    exp = torch.exp(logits) * ~eye
    log_prob = logits - torch.log(exp.sum(1, keepdim=True).clamp_min(1e-12))
    return -(log_prob * positives).sum(1).div(positives.sum(1)).mean()


class ProjectionHead(nn.Module):
    def __init__(self, channels, hidden=512, output=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, output))

    def forward(self, feature):
        return self.net(F.adaptive_avg_pool2d(feature, 1).flatten(1))


def nasa_mask(path):
    """Decode NASA's coloured annotation overlay into matrix/secondary/tertiary."""
    overlay = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if overlay is None:
        raise ValueError(f"cannot decode NASA mask: {path}")
    secondary = np.all(overlay == (255, 0, 0), axis=2)
    tertiary = np.all(overlay == (0, 0, 255), axis=2)
    if np.any(secondary & tertiary):
        raise ValueError(f"overlapping NASA classes in {path}")
    result = np.zeros(overlay.shape[:2], np.uint8)
    result[secondary] = 1
    result[tertiary] = 2
    return result


class NasaSuperDataset(Dataset):
    """Unique Super images for selection (18/5 OOD) or all-data refitting."""

    def __init__(self, root, split="select_train", samples_per_image=32, seed=0, augment=True,
                 size=512):
        base = Path(root) / "benchmark_segmentation_data"
        super1 = base / "Super1"
        directories = {
            "select_train": [(super1 / name, super1 / f"{name}_annot")
                             for name in ("train", "val", "test")],
            "ood": [(base / "Super2" / "different_test",
                     base / "Super2" / "different_test_annot")],
            "all": ([(super1 / name, super1 / f"{name}_annot")
                     for name in ("train", "val", "test")]
                    + [(base / "Super2" / "different_test",
                        base / "Super2" / "different_test_annot")]),
        }
        if split not in directories:
            raise ValueError(f"unknown NASA split {split}")
        pairs = []
        for image_dir, mask_dir in directories[split]:
            for path in sorted(image_dir.glob("*.tif")):
                pairs.append((path, mask_dir / f"{path.stem}_mask.tif"))
        # The selected directories are the 23-pair union, but keep this guard
        # if a future upstream archive happens to repeat one by basename.
        unique = {image.name: (image, mask) for image, mask in pairs}
        self.images = [unique[name][0] for name in sorted(unique)]
        self.masks = [unique[name][1] for name in sorted(unique)]
        missing = [str(p) for p in self.masks if not p.is_file()]
        if not self.images or missing:
            raise FileNotFoundError(f"invalid NASA Super {split} split; missing={missing[:3]}")
        expected = {"select_train": 18, "ood": 5, "all": 23}[split]
        if len(self.images) != expected:
            raise ValueError(f"NASA {split} has {len(self.images)} unique pairs, expected {expected}")
        self.samples_per_image = samples_per_image if augment else 1
        self.seed, self.epoch, self.augment, self.size = seed, 0, augment, int(size)

    def __len__(self):
        return len(self.images) * self.samples_per_image

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        source = index % len(self.images)
        image = _read_rgb(self.images[source])
        mask = nasa_mask(self.masks[source])
        if image.shape[:2] != (self.size, self.size):
            image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            rng = np.random.default_rng(self.seed + self.epoch * 10000019 + index)
            code = int(rng.integers(8))
            image, mask = _d4(image, code), _d4(mask, code)
            image = _appearance(image, rng)
        return {"image": _normalise(image), "mask": torch.from_numpy(mask.copy()).long(),
                "name": self.images[source].name}
