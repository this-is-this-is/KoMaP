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


def particle_bank(mask, cls=3, scale=1.0, halo=3, min_area=150, max_side=160, min_matrix=0.8):
    """Primary Si particles of one scan that can be moved elsewhere in it.

    A particle qualifies if it is whole (its halo stays inside the scan), not
    larger than max_side native pixels, and at least min_matrix of its
    halo-pixel ring is Al, so that pasting particle + ring puts a real
    particle edge next to matrix rather than a slice of a neighbouring phase.
    Returns [(y0, x0, footprint)], the footprint being particle + ring.
    """
    h, w = mask.shape
    r = max(1, int(round(halo * scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), 8)
    bank = []
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if area < min_area * scale * scale or max(bw, bh) > max_side * scale:
            continue
        y0, x0, y1, x1 = y - r, x - r, y + bh + r, x + bw + r
        if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
            continue
        particle = labels[y0:y1, x0:x1] == i
        footprint = cv2.dilate(particle.astype(np.uint8), kernel) > 0
        ring = footprint & ~particle
        if np.mean(mask[y0:y1, x0:x1][ring] == 0) < min_matrix:
            continue
        bank.append((int(y0), int(x0), footprint))
    return bank


def paste_particle(image, mask, scan_image, scan_mask, bank, rng, candidates=16):
    """Move one Primary Si particle, with its matrix ring, to a new place in the crop.

    Instance copy-paste in the spirit of Simple Copy-Paste / NucleiMix, cut to
    this data: the particle comes from the same scan (same exposure and
    magnification), keeps its true outline and edge profile, and is turned
    only by the 8 rotations/reflections -- never rescaled, because size is
    what separates Primary from Eutectic Si. Among a few random positions the
    one whose footprint covers the most matrix is used, so the ring lands on
    Al where it can.
    """
    y0, x0, footprint = bank[int(rng.integers(len(bank)))]
    fh, fw = footprint.shape
    src_image = scan_image[y0:y0 + fh, x0:x0 + fw]
    src_mask = scan_mask[y0:y0 + fh, x0:x0 + fw]
    k = int(rng.integers(8))
    turn = lambda a: np.rot90(a[:, ::-1] if k >= 4 else a, k % 4)
    footprint, src_image, src_mask = turn(footprint), turn(src_image), turn(src_mask)
    fh, fw = footprint.shape
    h, w = mask.shape
    if fh >= h or fw >= w:
        return image, mask
    best = None
    for _ in range(candidates):
        y = int(rng.integers(0, h - fh + 1))
        x = int(rng.integers(0, w - fw + 1))
        matrix = float(np.mean(mask[y:y + fh, x:x + fw][footprint] == 0))
        if best is None or matrix > best[0]:
            best = (matrix, y, x)
    _, y, x = best
    image, mask = image.copy(), mask.copy()
    image[y:y + fh, x:x + fw][footprint] = src_image[footprint]
    mask[y:y + fh, x:x + fw][footprint] = src_mask[footprint]
    return image, mask


def phase_levels(image, mask, si=2, matrix=0):
    """(Eutectic Si median gray, Al median gray) of one scan, or None if either is absent."""
    gray = image[..., 0]
    if not (mask == si).any() or not (mask == matrix).any():
        return None
    return float(np.median(gray[mask == si])), float(np.median(gray[mask == matrix]))


def darken_al3ni(image, mask, levels, rng, scale=1.0, cls=1, share=0.5, alpha_range=(0.5, 1.0),
                 min_area=100, skip_below=0.15):
    """Pull some Al3Ni regions down toward the scan's Eutectic Si gray level.

    The models tell Al3Ni from Eutectic Si mostly by brightness: the Al3Ni they
    call Eutectic Si sits at the Eutectic Si gray level of its scan, while the
    Al3Ni they get right is 20-50 levels brighter. That dark kind is about 5%
    of Al3Ni pixels in train and valid alike. Darkening whole regions -- shape,
    texture and mask kept -- leaves shape as the cue. It is SynthSeg's
    per-label intensity idea cut down to one class and one direction, since
    brightness is still the main cue between the other phases.

    Each Al3Ni region of at least min_area native pixels is picked with
    probability share and shifted down by alpha x (its median - Eutectic Si
    median). Regions already within skip_below of the Eutectic Si level, on
    the Eutectic Si -> Al gray axis, are left alone. The shift fades out over
    the outer native pixel, so annotated edge pixels keep their gray and no
    new edge appears; pixels of other classes are never changed.
    """
    eut, matrix = levels
    span = max(matrix - eut, 1.0)
    gray = image[..., 0]
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == cls).astype(np.uint8), 8)
    shift = np.zeros(mask.shape, np.float32)
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < min_area * scale * scale or rng.random() >= share:
            continue
        region = labels[y:y + h, x:x + w] == i
        level = float(np.median(gray[y:y + h, x:x + w][region]))
        if (level - eut) / span < skip_below:
            continue
        shift[y:y + h, x:x + w][region] = float(rng.uniform(*alpha_range)) * (level - eut)
    if not shift.any():
        return image
    r = max(1, int(round(scale)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    shift = cv2.GaussianBlur(cv2.erode(shift, kernel), (0, 0), sigmaX=scale) * (mask == cls)
    return _gray_rgb(gray.astype(np.float32) - shift)
