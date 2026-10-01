"""Overlapping-tile inference at native resolution.

The competition scores whole images, but training happens on crops, so
inference has to put the crops back together. Probabilities are accumulated
per class and the argmax is taken only at the very end: taking argmax per tile
first and stitching the labels would throw away the confidence needed to
resolve disagreement in the overlap.
"""
import numpy as np
import torch
import torch.nn.functional as F


def _cosine_window(h, w, device):
    """Raised-cosine weights, so tiles fade out toward their edges.

    A flat window makes seams visible where adjacent tiles disagree; tapering
    the contribution means the centre of a tile (which has the most context)
    dominates its own region.
    """
    wy = torch.hann_window(h, periodic=False, device=device).clamp_min(1e-3)
    wx = torch.hann_window(w, periodic=False, device=device).clamp_min(1e-3)
    return (wy[:, None] * wx[None, :])[None, None]


def _starts(total, tile, overlap):
    """Tile origins along one axis, always including a flush-right last tile."""
    if total <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    pos = list(range(0, max(total - tile, 0) + 1, step))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def _reflect_pad_to(image, size):
    """Reflect-pad (bottom/right) until both sides reach `size`.

    Reflection cannot add more than side-1 pixels in one go, which a tile of
    1024 over a 480-px scan needs, so it is applied repeatedly.
    """
    while image.shape[-2] < size or image.shape[-1] < size:
        h, w = image.shape[-2:]
        image = F.pad(image, (0, min(max(0, size - w), w - 1), 0, min(max(0, size - h), h - 1)),
                      mode="reflect")
    return image


def scaled_tile(tile, scale):
    """A tile of `tile` native pixels at `scale`, rounded to the encoder's stride of 32."""
    return tile if scale == 1 else max(32, int(round(tile * scale / 32)) * 32)


@torch.no_grad()
def predict_tiled(model, image, tile=512, overlap=0.5, num_classes=4,
                  flip_tta=False, amp=True, transpose_tta=False, scale=1.0):
    """Class probabilities for one full-resolution image.

    Args:
        image: normalised float tensor, (1, 3, H, W), already on the device.
        tile: tile size in the model's input pixels (after `scale`).
        flip_tta: average over horizontal / vertical / both flips.
        transpose_tta: also average the transposed tile; with flip_tta this
            covers all 8 rotations/reflections of the square tile.
        scale: the model was trained on scans enlarged this much (--input_scale);
            the image is enlarged the same way and the probabilities are brought
            back to native size, where the competition scores.
    Returns:
        (num_classes, H, W) probability tensor on the same device.
    """
    device = image.device
    _, _, H0, W0 = image.shape
    if scale != 1:
        image = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
    _, _, H, W = image.shape

    # A tile larger than the image would need padding on every call; reflect-pad
    # once up front instead so every tile is a real crop.
    image = _reflect_pad_to(image, tile)
    _, _, Hp, Wp = image.shape

    prob = torch.zeros(1, num_classes, Hp, Wp, device=device, dtype=torch.float32)
    norm = torch.zeros(1, 1, Hp, Wp, device=device, dtype=torch.float32)
    window = _cosine_window(tile, tile, device)

    variants = [(False, False)]
    if flip_tta:
        variants += [(True, False), (False, True), (True, True)]
    variants = [(fh, fv, tr) for tr in ((False, True) if transpose_tta else (False,))
                for fh, fv in variants]

    for y in _starts(Hp, tile, overlap):
        for x in _starts(Wp, tile, overlap):
            patch = image[:, :, y:y + tile, x:x + tile]

            acc = None
            for fh, fv, tr in variants:
                t = patch.transpose(-1, -2) if tr else patch
                if fh:
                    t = torch.flip(t, [-1])
                if fv:
                    t = torch.flip(t, [-2])
                with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
                    logits = model(t)
                if logits.shape[-2:] != t.shape[-2:]:
                    # SegFormer and other strided heads return a reduced map.
                    logits = F.interpolate(logits, size=t.shape[-2:],
                                           mode="bilinear", align_corners=False)
                if fh:
                    logits = torch.flip(logits, [-1])
                if fv:
                    logits = torch.flip(logits, [-2])
                if tr:
                    logits = logits.transpose(-1, -2)
                p = logits.float().softmax(1)
                acc = p if acc is None else acc + p
            acc = acc / len(variants)

            prob[:, :, y:y + tile, x:x + tile] += acc * window
            norm[:, :, y:y + tile, x:x + tile] += window

    prob = prob / norm.clamp_min(1e-6)
    prob = prob[:, :, :H, :W]
    if scale != 1:
        # Antialiased resampling averages the enlarged map back down; each
        # output pixel stays a convex mix, so the classes still sum to one.
        prob = F.interpolate(prob, size=(H0, W0), mode="bilinear", align_corners=False,
                             antialias=True)
    return prob[0]
