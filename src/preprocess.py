"""Per-scan standardisation shared by training, validation and inference.

Two things differ from scan to scan and are cheap to remove before the network
sees the image:

tone  -- exposure and contrast. Raw, 24 of the 90 labelled scans show Eutectic
         Si as bright as Al3Ni is elsewhere, although within any one scan the
         order Si < Al3Ni < Al always holds. Mapping two percentiles of each
         scan onto fixed gray levels makes the tones comparable (0 of 90).
scale -- magnification. Eutectic Si is ~1.7 px across on the 675x480 scans and
         ~3.5 px on the 1440x1024 ones, so a per-height scale brings structures
         to a common pixel size, as nnU-Net's resampling or Cellpose's diameter
         rescaling do. Only whole-number factors: x1.5 hurt on this data.

Both are functions of the scan alone, so they apply identically at test time.
"""
import numpy as np
import torch

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def tone_percentiles(image, percentiles):
    """The scan's gray level at the two percentiles (dark tail, matrix)."""
    lo, hi = np.percentile(image[..., 0], percentiles)
    return float(lo), float(hi)


def tone_normalize(image, percentiles, target):
    """Map the scan's two percentiles onto the gray levels in `target`.

    With (5, 70) the low point sits inside Si (at least 5.9% of every scan) and
    the high point inside the Al matrix (at least 41%), so the map is linear and
    independent of how the scan was exposed. The result stays uint8, so the
    colour augmentations and ImageNet statistics downstream are unchanged.
    """
    lo, hi = tone_percentiles(image, percentiles)
    if hi - lo < 1:
        return image
    out = (image.astype(np.float32) - lo) * ((target[1] - target[0]) / (hi - lo)) + target[0]
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def parse_scale_table(items):
    """['800:4', '900:3'] -> [(800, 4.0), (900, 3.0)], ascending by height."""
    table = []
    for item in items or []:
        height, scale = item.split(":")
        table.append((int(height), float(scale)))
    return sorted(table)


def scan_scale(height, base=1.0, table=None):
    """Scale for a scan of this native height: the first table row it fits, else `base`."""
    for max_height, scale in table or []:
        if height <= max_height:
            return scale
    return base


def prepare(image, args):
    """uint8 RGB scan -> normalised (1, 3, H, W) tensor, as the run in `args` was trained.

    `args` is the checkpoint's argument dict; runs from before these options
    existed have neither key and pass through untouched.
    """
    if args.get("tone_norm"):
        image = tone_normalize(image, args["tone_norm"], args["tone_target"])
    x = ((image / 255.0 - np.array(MEAN)) / np.array(STD)).transpose(2, 0, 1)
    return torch.from_numpy(x).float()[None]
