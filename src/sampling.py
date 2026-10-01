"""Class-aware cropping.

The score averages the four phases equally, but Eutectic Si covers ~7% of
pixels, so uniformly random crops under-sample exactly the classes that cost
the most points. This mixes plain random crops with crops centred on a chosen
class, keeping the natural distribution represented while giving rare phases
more exposure.
"""
import cv2
import numpy as np

# Pixels are subsampled before choosing a centre; scanning every pixel of a
# rare class in a 1920x1280 mask per sample is wasteful.
_MAX_CANDIDATES = 4096


def thin_density(mask, cls=2, half_width=2.0, window=64, stride=8):
    """Where thin, densely packed lamellae of one class sit, on a coarse grid.

    A pixel counts as thin when it lies within `half_width` px of the class's
    own edge (all of a ~3 px lamella, only the rim of a thick blob), and the
    box-filtered share of such pixels over `window` px measures how densely
    they pack. Returns (density[::stride, ::stride], stride) for the cropper.
    """
    m = (mask == cls).astype(np.uint8)
    if not m.any():
        return None
    inside = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    thin = ((m > 0) & (inside <= half_width)).astype(np.float32)
    dens = cv2.boxFilter(thin, -1, (window, window), normalize=True)
    return dens[::stride, ::stride].copy(), stride


class ClassAwareCropper:
    """Crops at the tile size, optionally at a randomised scale.

    The scans arrive at four magnifications -- Eutectic Si lamellae measure
    ~2.5px across on the 675x480 frames and ~4.0px on the 1440x1024 ones -- so
    a model trained only at native scale has to learn each magnification from
    its own handful of images. Sampling a window of `tile/s` and resampling it
    to `tile` spreads every image across that range instead. The window is cut
    at the source resolution and resampled once, rather than stretching an
    already-cut crop, so nothing is resampled twice.
    """

    def __init__(self, tile, num_classes=4, class_ratio=0.5, rng=None,
                 scale_range=None):
        self.tile = tile
        self.num_classes = num_classes
        self.class_ratio = class_ratio
        self.rng = rng or np.random.default_rng()
        self.scale_range = scale_range

    def __call__(self, image, mask, thin_map=None):
        """Crop at the tile size; with a thin_density() map, centre the crop
        where thin lamellae pack densest instead of using the policy below."""
        h, w = mask.shape

        t = self.tile
        if self.scale_range:
            lo, hi = self.scale_range
            # Sample in log space so up- and down-scaling are equally likely.
            s = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
            t = max(8, int(round(self.tile / s)))
            # Zooming out past the frame would fill the window with reflected
            # microstructure -- a mirrored seam is not a real grain boundary --
            # so cap the window at what this image can actually supply. The
            # 675x480 frames therefore zoom out less than the 1440x1024 ones.
            t = min(t, max(self.tile, min(h, w)))

        # Reflect-pad anything smaller than one tile so every crop is full size.
        if h < t or w < t:
            ph, pw = max(0, t - h), max(0, t - w)
            image = np.pad(image, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="reflect")
            h, w = mask.shape

        y = x = None
        if thin_map is not None:
            dens, stride = thin_map
            p = dens.ravel().astype(np.float64)
            k = int(self.rng.choice(p.size, p=p / p.sum()))
            cy = (k // dens.shape[1]) * stride + int(self.rng.integers(0, stride))
            cx = (k % dens.shape[1]) * stride + int(self.rng.integers(0, stride))
            y = int(np.clip(cy - t // 2, 0, h - t))
            x = int(np.clip(cx - t // 2, 0, w - t))

        if y is None and self.rng.random() < self.class_ratio:
            # Aim at a class that is actually present, preferring rarer ones.
            present = [c for c in range(self.num_classes) if (mask == c).any()]
            if present:
                counts = np.array([(mask == c).sum() for c in present], dtype=np.float64)
                weights = 1.0 / np.sqrt(counts)
                target = present[int(self.rng.choice(len(present), p=weights / weights.sum()))]
                ys, xs = np.nonzero(mask == target)
                if len(ys):
                    i = self.rng.integers(0, len(ys)) if len(ys) <= _MAX_CANDIDATES \
                        else self.rng.integers(0, len(ys))
                    cy, cx = int(ys[i]), int(xs[i])
                    # Centre the crop on the hit, then clamp inside the image.
                    y = int(np.clip(cy - t // 2, 0, h - t))
                    x = int(np.clip(cx - t // 2, 0, w - t))

        if y is None:
            y = int(self.rng.integers(0, h - t + 1))
            x = int(self.rng.integers(0, w - t + 1))

        image, mask = image[y:y + t, x:x + t], mask[y:y + t, x:x + t]
        if t != self.tile:
            size = (self.tile, self.tile)
            # AREA when shrinking keeps thin lamellae from aliasing away; the
            # mask is always NEAREST, since interpolating label indices would
            # invent phases that do not exist.
            interp = cv2.INTER_AREA if t > self.tile else cv2.INTER_LINEAR
            image = cv2.resize(image, size, interpolation=interp)
            mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return image, mask
