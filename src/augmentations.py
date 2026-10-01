"""Conservative, mask-aware augmentations for optical microstructure images.

All image-formation changes leave the semantic mask untouched. Context paste
copies image and all four mask classes together. Intensities are shared across
RGB channels because the source microscope images are grayscale RGB files.
"""

import cv2
import numpy as np


def _gray_rgb(gray):
    return np.repeat(np.clip(gray, 0, 255).astype(np.uint8)[..., None], 3, axis=2)


def illumination_field(image, rng, strength=0.05, grid=96):
    """Slowly varying multiplicative illumination, shared by all classes."""
    h, w = image.shape[:2]
    gh, gw = max(2, int(np.ceil(h / grid)) + 1), max(2, int(np.ceil(w / grid)) + 1)
    knots = rng.uniform(-strength, strength, (gh, gw)).astype(np.float32)
    field = cv2.resize(knots, (w, h), interpolation=cv2.INTER_CUBIC)
    field = np.clip(field, -strength, strength)
    return _gray_rgb(image[..., 0].astype(np.float32) * (1.0 + field))


def weak_blur(image, rng, scale=1.0, sigma_range=(0.3, 0.65)):
    """Gaussian point-spread approximation; sigma is in native image pixels."""
    sigma = float(rng.uniform(*sigma_range)) * scale
    gray = cv2.GaussianBlur(image[..., 0], (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.repeat(gray[..., None], 3, axis=2)


def fourier_style(image, donor, rng, low_frequency=0.02, mix_range=(0.1, 0.2)):
    """Mix low-frequency amplitude while retaining the source image phase."""
    source = image[..., 0].astype(np.float32)
    reference = donor[..., 0].astype(np.float32)
    if reference.shape != source.shape:
        reference = cv2.resize(reference, (source.shape[1], source.shape[0]))
    a = np.fft.fftshift(np.fft.fft2(source))
    b = np.fft.fftshift(np.fft.fft2(reference))
    h, w = source.shape
    ry = max(1, int(round(h * low_frequency)))
    rx = max(1, int(round(w * low_frequency)))
    cy, cx = h // 2, w // 2
    region = np.s_[cy - ry:cy + ry + 1, cx - rx:cx + rx + 1]
    alpha = float(rng.uniform(*mix_range))
    # Blending the complex coefficient would also change phase and geometry.
    amplitude = np.abs(a)
    amplitude[region] = ((1 - alpha) * amplitude[region]
                         + alpha * np.abs(b[region]))
    result = np.fft.ifft2(np.fft.ifftshift(amplitude * np.exp(1j * np.angle(a)))).real
    return _gray_rgb(result)


def context_paste(image, mask, donor_image, donor_mask, rng, scale=1.0,
                  native_size=(48, 96), candidates=48):
    """Paste a small Si-rich box where its edge matches the receiving context.

    This is a cheap, context-aware experiment inspired by NucleiMix, not its
    diffusion inpainting method. A rectangular seam is rejected if too many
    semantic labels disagree along the border.
    """
    h, w = mask.shape
    lo = max(12, int(round(native_size[0] * scale)))
    hi = max(lo, int(round(native_size[1] * scale)))
    size = min(int(rng.integers(lo, hi + 1)), h - 2, w - 2)
    if size < 12 or donor_mask.shape[0] < size or donor_mask.shape[1] < size:
        return image, mask
    ys, xs = np.nonzero(donor_mask == 2)
    if not len(ys):
        return image, mask
    for _ in range(8):
        i = int(rng.integers(len(ys)))
        sy = int(np.clip(ys[i] - size // 2, 0, donor_mask.shape[0] - size))
        sx = int(np.clip(xs[i] - size // 2, 0, donor_mask.shape[1] - size))
        dm = donor_mask[sy:sy + size, sx:sx + size]
        if np.mean(dm == 2) >= 0.015:
            break
    else:
        return image, mask

    di = donor_image[sy:sy + size, sx:sx + size]
    source_edge = np.concatenate((dm[0], dm[-1], dm[1:-1, 0], dm[1:-1, -1]))
    source_gray_edge = np.concatenate((di[0, :, 0], di[-1, :, 0],
                                       di[1:-1, 0, 0], di[1:-1, -1, 0]))
    best = None
    for _ in range(candidates):
        y = int(rng.integers(1, h - size))
        x = int(rng.integers(1, w - size))
        # The receiving pixels immediately outside the replacement box must
        # resemble the donor's inner boundary in both label and intensity.
        outside = np.concatenate((mask[y - 1, x:x + size],
                                  mask[y + size, x:x + size],
                                  mask[y + 1:y + size - 1, x - 1],
                                  mask[y + 1:y + size - 1, x + size]))
        outer_gray = np.concatenate((image[y - 1, x:x + size, 0],
                                     image[y + size, x:x + size, 0],
                                     image[y + 1:y + size - 1, x - 1, 0],
                                     image[y + 1:y + size - 1, x + size, 0]))
        mismatch = float(np.mean(outside != source_edge))
        intensity_gap = float(np.mean(np.abs(outer_gray.astype(np.float32)
                                              - source_gray_edge.astype(np.float32))) / 255)
        score = mismatch + 0.25 * intensity_gap
        if best is None or score < best[0]:
            best = (score, mismatch, y, x)
    if best is None or best[1] > 0.45:
        return image, mask
    _, _, y, x = best
    image, mask = image.copy(), mask.copy()
    target_image = image[y:y + size, x:x + size]
    target_mask = mask[y:y + size, x:x + size]
    # Soften the box seam only where the semantic label agrees. Blending
    # across differing labels would teach the model a wrong color-to-mask edge.
    edge_distance = np.minimum.reduce(np.broadcast_arrays(
        np.arange(size)[:, None], np.arange(size)[None, :],
        (size - 1 - np.arange(size))[:, None],
        (size - 1 - np.arange(size))[None, :]))
    alpha = np.where(dm == target_mask, np.clip((edge_distance + 1) / 3, 1 / 3, 1), 1)
    pasted = alpha[..., None] * di.astype(np.float32) + (1 - alpha[..., None]) * target_image
    image[y:y + size, x:x + size] = np.clip(pasted, 0, 255).astype(np.uint8)
    mask[y:y + size, x:x + size] = dm
    return image, mask


def attenuate_connected_si(image, mask, rng, scale=1.0):
    """Reduce local contrast on an existing Si component; never edit its GT."""
    binary = (mask == 2).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(8, int(round(10 * scale * scale)))
    choices = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= minimum_area) + 1
    if count <= 1 or not len(choices):
        return image
    component = int(rng.choice(choices))
    ys, xs = np.nonzero(labels == component)
    i = int(rng.integers(len(ys)))
    cy, cx = int(ys[i]), int(xs[i])
    sigma = float(rng.uniform(2.0, 3.5)) * scale
    radius = max(2, int(np.ceil(3 * sigma)))
    y0, y1 = max(0, cy - radius), min(mask.shape[0], cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(mask.shape[1], cx + radius + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    amount = float(rng.uniform(0.25, 0.45))
    alpha = amount * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    # A local surrounding estimate makes a short part of the component less
    # visible, without inventing a foreground structure in another class.
    gray = image[..., 0]
    context = gray[max(0, cy - 2 * radius):min(mask.shape[0], cy + 2 * radius + 1),
                   max(0, cx - 2 * radius):min(mask.shape[1], cx + 2 * radius + 1)]
    background = float(np.median(context))
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    out = image.copy()
    out[y0:y1, x0:x1] = _gray_rgb(patch * (1 - alpha) + background * alpha)
    return out
