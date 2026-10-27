"""Small residual logit fusion shared by OOF training and final inference."""
from __future__ import annotations

import torch
import torch.nn as nn


def normalise_probability(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.float().clamp_min(1e-7)
    return probability / probability.sum(1, keepdim=True).clamp_min(1e-7)


def fusion_features(b3: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
    """Log-probabilities and normalised entropy from both base families."""
    b3, se = normalise_probability(b3), normalise_probability(se)
    b3_log, se_log = b3.log(), se.log()
    scale = torch.log(torch.tensor(float(b3.shape[1]), device=b3.device))
    b3_entropy = -(b3 * b3_log).sum(1, keepdim=True) / scale
    se_entropy = -(se * se_log).sum(1, keepdim=True) / scale
    return torch.cat((b3_log, se_log, b3_entropy, se_entropy), 1)


class LogitFusionHead(nn.Module):
    """A low-capacity spatial correction on top of equal probability fusion.

    The last layer starts at zero, so the initial model is exactly the safe
    arithmetic probability average. Training only has to learn a residual.
    """

    def __init__(self, classes=4, hidden=16):
        super().__init__()
        self.classes, self.hidden = classes, hidden
        self.correction = nn.Sequential(
            nn.Conv2d(2 * classes + 2, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, classes, 1),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, b3: torch.Tensor, se: torch.Tensor) -> torch.Tensor:
        b3, se = normalise_probability(b3), normalise_probability(se)
        base = ((b3 + se) * 0.5).clamp_min(1e-7).log()
        return base + self.correction(fusion_features(b3, se))


def load_fusion_checkpoint(path, device="cpu"):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("format") != "komap_logit_fusion_v1":
        raise ValueError(f"unsupported fusion checkpoint: {path}")
    model = LogitFusionHead(value.get("classes", 4), value.get("hidden", 16))
    model.load_state_dict(value["model"])
    return model.to(device).eval(), value
