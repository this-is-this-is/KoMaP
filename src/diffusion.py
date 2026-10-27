"""Joint image-mask diffusion: the model, its noise parameterisation and its sampler.

One denoiser runs over five planes -- the gray micrograph and the four phases
as one-hot planes, all in [-1, 1] -- so an image and its mask come out of one
sampling trajectory and agree pixel for pixel. This is the one-stage design of
PF-DiffSeg (Nguyen et al., 2025; multiphase steel, 42 micrographs) and SatSynth
(Toker et al., CVPR 2024), which beat mask-then-image pipelines there. As in
PF-DiffSeg the model is conditioned on the phase fractions of the patch, so
rare phases can be asked for at sampling time.

Choices that differ from those papers, and why:
    one-hot planes, argmax to decode   bit or colour codes put a third class
                                       on every boundary between two phases
    EDM (Karras et al., NeurIPS 2022)  preconditioned denoiser, log-normal
                                       noise levels, Heun sampler: stable in
                                       pixel space at 256 px, where both papers
                                       fell back to 128 px plus super-resolution
                                       -- a 3 px lamella does not survive that
    no attention, patches of scans     the micrographs are textures; a purely
                                       convolutional net trained on 256 px
                                       patches samples any size (Patch
                                       Diffusion, Wang et al., NeurIPS 2023)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_DATA = 0.5
NUM_CLASSES = 4
CHANNELS = 1 + NUM_CLASSES
# The image is one plane against four mask planes; weight it to half the loss.
CHANNEL_WEIGHT = (2.5, 0.625, 0.625, 0.625, 0.625)


def to_planes(image, mask):
    """uint8 gray (H, W) and index mask (H, W) -> float planes (5, H, W) in [-1, 1]."""
    x = torch.empty(CHANNELS, *mask.shape, dtype=torch.float32)
    x[0] = torch.as_tensor(image, dtype=torch.float32) / 127.5 - 1
    x[1:] = F.one_hot(torch.as_tensor(mask, dtype=torch.long), NUM_CLASSES).permute(2, 0, 1) * 2.0 - 1
    return x


def from_planes(x):
    """(B, 5, H, W) planes -> uint8 gray (B, H, W) and index masks (B, H, W)."""
    image = ((x[:, 0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return image, x[:, 1:].argmax(1).to(torch.uint8)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb, dropout):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(32, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(emb, cout)
        self.norm2, self.conv2 = nn.GroupNorm(32, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        # Each block starts as the identity, as in ADM.
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x))) + self.emb(emb)[:, :, None, None]
        return self.skip(x) + self.conv2(self.dropout(F.silu(self.norm2(h))))


class UNet(nn.Module):
    """ADM-style U-Net without attention: 5 planes in, 5 planes out, any size divisible by 16."""

    def __init__(self, base=96, mult=(1, 2, 2, 4, 4), blocks=2, emb=384, dropout=0.1):
        super().__init__()
        self.base = base
        self.noise = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.fraction = nn.Sequential(nn.Linear(NUM_CLASSES, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.conv_in = nn.Conv2d(CHANNELS, base, 3, padding=1)
        chans, c = [base], base
        self.down = nn.ModuleList()
        for level, m in enumerate(mult):
            for _ in range(blocks):
                # Dropout only below full resolution, where most parameters are.
                self.down.append(ResBlock(c, base * m, emb, dropout if level >= 2 else 0.0))
                c = base * m
                chans.append(c)
            if level < len(mult) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb, dropout), ResBlock(c, c, emb, dropout)])
        self.up = nn.ModuleList()
        for level, m in reversed(list(enumerate(mult))):
            for _ in range(blocks + 1):
                self.up.append(ResBlock(c + chans.pop(), base * m, emb, dropout if level >= 2 else 0.0))
                c = base * m
            if level > 0:
                self.up.append(nn.Conv2d(c, c, 3, padding=1))
        self.norm_out, self.conv_out = nn.GroupNorm(32, c), nn.Conv2d(c, CHANNELS, 3, padding=1)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, x, c_noise, fraction):
        # Sinusoidal features of ln(sigma) / 4, stretched so they span many periods.
        half = self.base // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        t = c_noise.float()[:, None] * 250 * freqs[None]
        emb = self.noise(torch.cat([t.cos(), t.sin()], 1)) + self.fraction(fraction.float())
        emb = emb.to(x.dtype)
        h = self.conv_in(x)
        skips = [h]
        for layer in self.down:
            h = layer(h, emb) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)
        for layer in self.mid:
            h = layer(h, emb)
        for layer in self.up:
            if isinstance(layer, ResBlock):
                h = layer(torch.cat([h, skips.pop()], 1), emb)
            else:
                h = layer(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.conv_out(F.silu(self.norm_out(h)))


def denoise(net, x, sigma, fraction):
    """EDM-preconditioned denoiser D(x; sigma): the clean planes estimated from x = clean + sigma * noise."""
    s = sigma.float().view(-1, 1, 1, 1)
    total = s ** 2 + SIGMA_DATA ** 2
    out = net((x / total.sqrt()).to(memory_format=torch.channels_last), sigma.float().log() / 4, fraction)
    return SIGMA_DATA ** 2 / total * x + s * SIGMA_DATA / total.sqrt() * out.float()


def diffusion_loss(net, clean, fraction, p_mean=-0.4, p_std=1.4, sigma=None, noise=None):
    """EDM training loss. Noise levels are log-normal; p_mean is raised from EDM's -1.2
    because at 256 px the layout only dissolves at much higher noise than at 32-64 px
    (Hoogeboom et al., "simple diffusion", ICML 2023)."""
    if sigma is None:
        sigma = torch.exp(p_mean + p_std * torch.randn(clean.shape[0], device=clean.device))
    if noise is None:
        noise = torch.randn_like(clean)
    s = sigma.view(-1, 1, 1, 1)
    with torch.amp.autocast("cuda", enabled=clean.is_cuda):
        estimate = denoise(net, clean + s * noise, sigma, fraction)
    weight = (s ** 2 + SIGMA_DATA ** 2) / (s * SIGMA_DATA) ** 2
    channel = torch.tensor(CHANNEL_WEIGHT, device=clean.device).view(1, -1, 1, 1)
    return (weight * channel * (estimate - clean) ** 2).mean()


@torch.no_grad()
def sample(net, fraction, size, steps=32, sigma_min=0.002, sigma_max=80.0, rho=7.0, generator=None):
    """EDM's deterministic second-order (Heun) sampler. Returns (B, 5, H, W) planes."""
    device = fraction.device
    i = torch.arange(steps, device=device)
    t = (sigma_max ** (1 / rho) + i / (steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t = torch.cat([t, torch.zeros(1, device=device)])
    x = torch.randn(fraction.shape[0], CHANNELS, *size, device=device, generator=generator) * t[0]
    ones = torch.ones(fraction.shape[0], device=device)

    def slope(x, sigma):
        with torch.amp.autocast("cuda", enabled=x.is_cuda):
            return (x - denoise(net, x, sigma * ones, fraction)) / sigma

    for a, b in zip(t[:-1], t[1:]):
        d = slope(x, a)
        x_next = x + (b - a) * d
        if b > 0:
            x_next = x + (b - a) * 0.5 * (d + slope(x_next, b))
        x = x_next
    return x
