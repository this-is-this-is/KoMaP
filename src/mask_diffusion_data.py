"""Train-only data and auditable source selection for conditional generation."""
import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from komap import COLORS
from mask_diffusion import mask_condition, phase_style


def read_train(data_root):
    records = []
    for path in sorted((Path(data_root) / "train/train/images").glob("*_image.png")):
        mask_path = path.parent.parent / "masks" / path.name.replace("_image", "_mask")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if image is None or rgb is None:
            raise ValueError(f"unreadable pair: {path}")
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        matches = (rgb[..., None, :] == COLORS).all(-1)
        if not matches.any(-1).all() or rgb.shape[:2] != image.shape:
            raise ValueError(f"invalid mask palette/shape: {mask_path}")
        mask = matches.argmax(-1).astype(np.uint8)
        records.append(dict(name=path.name, image=image, mask=mask,
                            style=phase_style(image, mask),
                            sha256=hashlib.sha256(path.read_bytes() + mask_path.read_bytes()).hexdigest()))
    if not records:
        raise FileNotFoundError(f"no training pairs in {data_root}/train/train")
    return records


def split_records(records, seed=17, fraction=0.15):
    """Keep equal A<number> file prefixes together; a conservative filename proxy."""
    def group(record):
        match = re.match(r"(A\d+)_", record["name"])
        return match.group(1) if match else record["name"]
    groups = sorted(set(map(group, records)))
    if len(groups) < 2:
        raise ValueError("need two source groups for internal validation")
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(groups)[:max(1, min(len(groups) - 1, round(len(groups) * fraction)))])
    return ([r for r in records if group(r) not in held],
            [r for r in records if group(r) in held])


def crop(record, size, rng, orient=True):
    image, mask = record["image"], record["mask"]
    h, w = mask.shape
    ph, pw = max(0, size - h), max(0, size - w)
    if ph or pw:
        image = np.pad(image, ((0, ph), (0, pw)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
    y = int(rng.integers(image.shape[0] - size + 1))
    x = int(rng.integers(image.shape[1] - size + 1))
    image, mask = image[y:y + size, x:x + size], mask[y:y + size, x:x + size]
    k = int(rng.integers(8)) if orient else 0
    if k >= 4:
        image, mask = image[:, ::-1], mask[:, ::-1]
    image, mask = np.rot90(image, k % 4).copy(), np.rot90(mask, k % 4).copy()
    return image, mask, dict(y=y, x=x, size=size, d4=k, pad_bottom=ph, pad_right=pw)


def tensors(image, mask, style):
    return dict(image=torch.from_numpy(image.astype(np.float32)[None] / 127.5 - 1),
                condition=torch.from_numpy(mask_condition(mask)), style=torch.from_numpy(style.copy()))


class PatchStream(IterableDataset):
    def __init__(self, records, size, seed):
        self.records, self.size, self.seed = records, size, seed

    def __iter__(self):
        worker = get_worker_info()
        rng = np.random.default_rng(self.seed + (worker.id if worker else 0))
        while True:
            record = self.records[int(rng.integers(len(self.records)))]
            image, mask, _ = crop(record, self.size, rng)
            yield tensors(image, mask, record["style"])


def remodel(mask, rng, displacement=0.75):
    """Small smooth warp, accepted only if class topology/area stays close.

    This changes a CONDITION mask; a new image must subsequently be generated.
    It is never paired with the unwarped source image as a training label.
    """
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    offsets = [cv2.resize(rng.uniform(-displacement, displacement, (5, 5)).astype(np.float32),
                          (w, h), interpolation=cv2.INTER_CUBIC).clip(-displacement, displacement)
               for _ in range(2)]
    warped = cv2.remap(mask, xx + offsets[0], yy + offsets[1], cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_REFLECT_101)
    for c in range(4):
        before, after = (mask == c).astype(np.uint8), (warped == c).astype(np.uint8)
        area = int(before.sum())
        if abs(int(after.sum()) - area) > max(2, 0.05 * area):
            return mask.copy(), False
        for binary in (True, False):
            a, b = (before, after) if binary else (1 - before, 1 - after)
            if cv2.connectedComponents(a, connectivity=8)[0] != cv2.connectedComponents(b, connectivity=8)[0]:
                return mask.copy(), False
    return warped, bool(np.any(warped != mask))
