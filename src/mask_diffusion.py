"""Pixel-space mask-to-image EDM, inspired by NeuroDiff/MSDM, not a reproduction.

Only the grayscale image is diffused. The semantic mask is a clean condition at
every resolution. Four signed-distance channels retain local boundary geometry;
per-phase gray statistics describe appearance without passing a source image.
"""
import math

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from diffusion import ResBlock, SIGMA_DATA


def mask_condition(mask, distance_clip=8.0):
    """4 one-hot + 4 signed distances, clipped in native pixels (inside > 0)."""
    if mask.ndim != 2 or not np.isin(mask, np.arange(4)).all():
        raise ValueError("expected a 2D mask with class IDs 0..3")
    planes = np.eye(4, dtype=np.float32)[mask].transpose(2, 0, 1)
    sdf = []
    for c in range(4):
        inside = (mask == c).astype(np.uint8)
        if not inside.any():
            signed = np.full(mask.shape, -distance_clip, np.float32)
        elif inside.all():
            signed = np.full(mask.shape, distance_clip, np.float32)
        else:
            signed = (cv2.distanceTransform(inside, cv2.DIST_L2, 5)
                      - cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5))
        sdf.append(np.clip(signed / distance_clip, -1, 1))
    return np.concatenate([planes, np.stack(sdf)]).astype(np.float32)


def phase_style(image, mask):
    """Per-class mean [-1,1], std [0,1], presence; absent classes remain zero."""
    style = np.zeros(12, np.float32)
    for c in range(4):
        values = image[mask == c].astype(np.float32)
        if len(values):
            style[c] = values.mean() / 127.5 - 1
            style[4 + c] = values.std() / 127.5
            style[8 + c] = 1
    return style


class MaskUNet(nn.Module):
    def __init__(self, base=64, geometry=True, style=True, dropout=0.05):
        super().__init__()
        if base < 32 or base % 32:
            raise ValueError("base must be a positive multiple of 32")
        self.base, self.geometry, self.use_style = base, geometry, style
        channels = [base * m for m in (1, 2, 3, 4)]
        emb = base * 4
        self.noise = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.style = nn.Sequential(nn.Linear(12, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.input = nn.Conv2d(1, base, 3, padding=1)
        self.conditions = nn.ModuleList([
            nn.Conv2d(8 if geometry else 4, c, 3, padding=1) for c in channels])
        self.encoder = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, c in enumerate(channels):
            self.encoder.append(nn.ModuleList([ResBlock(c, c, emb, dropout if i else 0),
                                               ResBlock(c, c, emb, dropout if i else 0)]))
            if i < 3:
                self.down.append(nn.Conv2d(c, channels[i + 1], 3, stride=2, padding=1))
        self.middle = ResBlock(channels[-1], channels[-1], emb, dropout)
        self.decoder = nn.ModuleList()
        for i in (2, 1, 0):
            self.decoder.append(nn.ModuleList([
                ResBlock(channels[i + 1] + channels[i], channels[i], emb, dropout if i else 0),
                ResBlock(channels[i], channels[i], emb, dropout if i else 0)]))
        self.output = nn.Sequential(nn.GroupNorm(32, base), nn.SiLU(), nn.Conv2d(base, 1, 3, padding=1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, image, c_noise, condition, style):
        if image.shape[-2] % 8 or image.shape[-1] % 8:
            raise ValueError("spatial size must be divisible by 8")
        half = self.base // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=image.device) / half)
        t = c_noise.float()[:, None] * 250 * freq
        emb = self.noise(torch.cat([t.cos(), t.sin()], dim=1))
        if self.use_style:
            emb = emb + self.style(style.float())
        h = self.input(image)
        cond = condition if self.geometry else condition[:, :4]
        skips, guidance = [], []
        for i, blocks in enumerate(self.encoder):
            # Area pooling keeps the occupancy of subpixel-thin phases visible.
            g = self.conditions[i](F.interpolate(cond, size=h.shape[-2:], mode="area"))
            h = h + g
            for block in blocks:
                h = block(h, emb)
            skips.append(h)
            guidance.append(g)
            if i < 3:
                h = self.down[i](h)
        h = self.middle(h, emb)
        for i, blocks in zip((2, 1, 0), self.decoder):
            h = F.interpolate(h, size=skips[i].shape[-2:], mode="nearest")
            h = blocks[0](torch.cat([h, skips[i]], dim=1), emb) + guidance[i]
            h = blocks[1](h, emb)
        return self.output(h)


def denoise(net, noisy, sigma, condition, style):
    s = sigma.float().reshape(-1, 1, 1, 1)
    total = s.square() + SIGMA_DATA ** 2
    out = net(noisy / total.sqrt(), sigma.log() / 4, condition, style)
    return SIGMA_DATA ** 2 / total * noisy + s * SIGMA_DATA / total.sqrt() * out.float()


def loss(net, image, condition, style, p_mean=-0.4, p_std=1.4,
         boundary_weight=0.5, thin_weight=0.5, sigma=None, noise=None):
    if sigma is None:
        sigma = (p_mean + p_std * torch.randn(image.shape[0], device=image.device)).exp()
    if noise is None:
        noise = torch.randn_like(image)
    s = sigma[:, None, None, None]
    with torch.amp.autocast("cuda", enabled=image.is_cuda, dtype=torch.bfloat16):
        estimate = denoise(net, image + s * noise, sigma, condition, style)
    spatial = torch.ones_like(image)
    if boundary_weight:
        labels = condition[:, :4].argmax(1, keepdim=True).float()
        boundary = F.max_pool2d(labels, 3, 1, 1) != -F.max_pool2d(-labels, 3, 1, 1)
        spatial = spatial + boundary_weight * boundary.float()
    if thin_weight:
        spatial = spatial + thin_weight * condition[:, 2:3]
    spatial = spatial / spatial.mean((-2, -1), keepdim=True)
    weight = (s.square() + SIGMA_DATA ** 2) / (s * SIGMA_DATA).square()
    return (weight * spatial * (estimate - image).square()).mean()


@torch.no_grad()
def sample(net, condition, style, steps=40, generator=None):
    if steps < 2:
        raise ValueError("steps must be at least 2")
    device = condition.device
    schedule = torch.linspace(0, 1, steps, device=device)
    schedule = (80 ** (1 / 7) + schedule * (0.002 ** (1 / 7) - 80 ** (1 / 7))) ** 7
    schedule = torch.cat([schedule, schedule.new_zeros(1)])
    shape = (condition.shape[0], 1, *condition.shape[-2:])
    image = torch.randn(shape, device=device, generator=generator) * schedule[0]
    ones = torch.ones(shape[0], device=device)

    def slope(x, sigma):
        with torch.amp.autocast("cuda", enabled=x.is_cuda, dtype=torch.bfloat16):
            return (x - denoise(net, x, sigma * ones, condition, style)) / sigma

    for a, b in zip(schedule[:-1], schedule[1:]):
        d = slope(image, a)
        next_image = image + (b - a) * d
        if b > 0:
            next_image = image + (b - a) * (d + slope(next_image, b)) / 2
        image = next_image
    return ((image[:, 0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
