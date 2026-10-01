"""Boundary-oriented training terms for KoMaP phase segmentation.

Each term follows its paper's (or the authors' code's) recipe; train_v2.py
decides how they are combined. They take raw logits (B, K, H, W) and an index
mask (B, H, W) and compute in float32 even under autocast.

    label relaxation           Zhu et al., CVPR 2019
    boundary label smoothing   Wang et al., NeurIPS 2023 (with the Jaccard metric loss)
    interface-weighted CE      Strohmann et al., Sci. Rep. 2019
    Boundary DoU               Sun et al., MICCAI 2023
    boundary (surface) loss    Kervadec et al., MIDL 2019
    Active Boundary Loss       Wang et al., AAAI 2022

The distance maps the last two need are built on the CPU by the data loader
(signed_distance / boundary_distance), not inside the training step.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _onehot(target, num_classes):
    return F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()


def neighbour_classes(target, num_classes, k=3):
    """Multi-hot (B, K, H, W): which classes occur in each pixel's k x k window."""
    return F.max_pool2d(_onehot(target, num_classes), k, stride=1, padding=k // 2)


# --------------------------------------------------------------------------
# CE variants

def relaxed_ce(logits, target, num_classes, k=3):
    """Boundary label relaxation (Zhu et al., CVPR 2019).

    -log sum_{c in N(p)} p_c, where N(p) is the set of classes in the k x k
    window. Inside a phase the window holds one class and this is plain CE; on
    a boundary the mass may sit on either side, so the +-1 px placement that
    the annotation itself cannot pin down is no longer penalised.
    """
    allowed = neighbour_classes(target, num_classes, k) > 0
    logp = F.log_softmax(logits.float(), 1)
    return -torch.logsumexp(logp.masked_fill(~allowed, float("-inf")), 1).mean()


def boundary_smoothed_targets(target, num_classes, eps=0.5, k=3):
    """Boundary label smoothing (Wang et al., NeurIPS 2023).

    Uniform label smoothing with coefficient eps, but only on pixels whose
    k x k window holds more than one class. The paper's best setting is k=3,
    eps=0.5: smooth a thin band, and smooth it hard.
    """
    onehot = _onehot(target, num_classes)
    boundary = (F.max_pool2d(onehot, k, stride=1, padding=k // 2).sum(1, keepdim=True) > 1).float()
    return boundary * (onehot * (1 - eps) + eps / num_classes) + (1 - boundary) * onehot


def soft_ce(logits, soft_targets):
    return -(soft_targets * F.log_softmax(logits.float(), 1)).sum(1).mean()


def interface_weights(target, num_classes, w_interface=1.5, w_si=1.25,
                      aluminide=1, silicon=(2, 3)):
    """Pixel weights from Strohmann et al. (Sci. Rep. 2019), AlSi12Cu3Ni2.

    Si pixels get w_si, and pixels whose 3x3 window holds both the aluminide
    (Al3Ni here) and Si get w_interface -- the paper's best pair was 1.25 /
    1.5. That interface is where this data's largest confusion,
    Al3Ni <-> Eutectic Si, sits.
    """
    nb = neighbour_classes(target, num_classes, 3) > 0
    interface = nb[:, aluminide] & nb[:, list(silicon)].any(1)
    w = torch.ones(target.shape, device=target.device)
    w[torch.isin(target, torch.tensor(silicon, device=target.device))] = w_si
    w[interface] = w_interface
    return w


def weighted_ce(logits, target, weights):
    return (F.cross_entropy(logits.float(), target, reduction="none") * weights).mean()


# --------------------------------------------------------------------------
# region terms

class JaccardMetricLoss(nn.Module):
    """JML1 (Wang et al., NeurIPS 2023).

    1 - (|x|+|y|-|x-y|) / (|x|+|y|+|x-y|) with L1 norms, per class over the
    batch, averaged over classes. Equal to soft Jaccard on hard labels, but
    unlike it, still minimised at x = y when the targets are soft -- which is
    what lets it train on boundary-smoothed labels.
    """

    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps

    def forward(self, logits, soft_targets):
        prob = logits.float().softmax(1)
        dims = (0, 2, 3)
        x, y = prob.sum(dims), soft_targets.sum(dims)
        diff = (prob - soft_targets).abs().sum(dims)
        return (1 - (x + y - diff + self.eps) / (x + y + diff + self.eps)).mean()


class BoundaryDoULoss(nn.Module):
    """Boundary DoU (Sun et al., MICCAI 2023), following the authors' code.

    Per class: (|P|^2 + |G|^2 - 2|P.G|) / (|P|^2 + |G|^2 - (1+a)|P.G|), with
    a = 1 - 2C/S from the class's boundary (C) and total (S) pixel counts, so
    small or thin classes -- mostly boundary -- get a sharper loss. a is
    truncated at 0.8 as the authors recommend.
    """

    def forward(self, logits, target):
        prob = logits.float().softmax(1)
        onehot = _onehot(target, prob.shape[1])
        cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], device=prob.device)
        # How many of (self + 4 neighbours) share the class; 5 means interior.
        same = F.conv2d(onehot.flatten(0, 1)[:, None], cross[None, None], padding=1).view_as(onehot)
        dims, smooth = (0, 2, 3), 1e-5
        C = ((onehot > 0) & (same < 5)).sum(dims).float()
        S = onehot.sum(dims)
        alpha = (2 * (1 - (C + smooth) / (S + smooth)) - 1).clamp(max=0.8)
        inter = (prob * onehot).sum(dims)
        y, z = (onehot * onehot).sum(dims), (prob * prob).sum(dims)
        return ((z + y - 2 * inter + smooth) / (z + y - (1 + alpha) * inter + smooth)).mean()


# --------------------------------------------------------------------------
# distance-map terms

def signed_distance(mask, num_classes):
    """Per-class signed distance to the class contour, negative inside (Kervadec et al.).

    Same convention as the authors' one_hot2dist: outside pixels hold their
    distance to the class, inside pixels minus (distance to the outside - 1),
    so the class's own boundary pixels sit at 0. A class absent from the crop
    stays all-zero, as in their code.
    """
    out = np.zeros((num_classes,) + mask.shape, np.float32)
    for c in range(num_classes):
        pos = (mask == c).astype(np.uint8)
        if not pos.any() or pos.all():
            continue
        neg = 1 - pos
        d_out = cv2.distanceTransform(neg, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        d_in = cv2.distanceTransform(pos, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        out[c] = d_out * neg - (d_in - 1) * pos
    return out


def surface_loss(logits, sdf):
    """Kervadec et al.'s boundary loss: mean of softmax x signed distance."""
    return (logits.float().softmax(1) * sdf).mean()


def boundary_distance(mask):
    """ABL's target map: distance to the nearest GT boundary pixel minus 1, floored at 0.

    Boundary pixels are marked the way the authors' gt2boundary does (a pixel
    whose lower or right neighbour differs), so both sides of an edge read 0.
    """
    b = np.zeros(mask.shape, bool)
    b[:-1] |= mask[1:] != mask[:-1]
    b[:, :-1] |= mask[:, 1:] != mask[:, :-1]
    if not b.any():
        return np.zeros(mask.shape, np.float32)
    d = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.maximum(d - 1, 0).astype(np.float32)


def _kl(a, b):
    """Element-wise KL(b || a) over the class axis, as in the authors' abl.py."""
    return F.softmax(b, 1) * (F.log_softmax(b, 1) - F.log_softmax(a, 1))


class ActiveBoundaryLoss(nn.Module):
    """Active Boundary Loss (Wang et al., AAAI 2022), ported from the authors' abl.py.

    Pixels on the predicted boundary (large KL divergence between neighbours)
    are taught which of their 8 neighbours lies closer to the GT boundary: the
    KL divergences to the neighbours act as logits of a label-smoothed 8-way
    CE, weighted by the pixel's distance to the GT boundary clipped at theta.
    Pixels already on the GT boundary are dropped.
    """

    # neighbour offsets (row, col); index 8 is the pixel itself
    DR = [1, -1, 0, 0, -1, 1, -1, 1, 0]
    DC = [0, 0, -1, 1, 1, 1, -1, -1, 0]

    def __init__(self, max_n_ratio=1 / 100, label_smoothing=0.2, theta=20.0, detach=True):
        super().__init__()
        self.max_n_ratio = max_n_ratio
        self.label_smoothing = label_smoothing
        self.theta = theta
        self.detach = detach

    def predicted_boundary(self, logits):
        _, _, h, w = logits.shape
        max_n = h * w * self.max_n_ratio
        kl_ud = F.pad(_kl(logits[:, :, 1:], logits[:, :, :-1]).sum(1, keepdim=True), (0, 0, 0, 1))
        kl_lr = F.pad(_kl(logits[:, :, :, 1:], logits[:, :, :, :-1]).sum(1, keepdim=True), (0, 1, 0, 0))
        kl = kl_ud + kl_lr
        # Raise the threshold until at most max_n pixels remain (per batch,
        # as the original does), so flat predictions cannot flood the loss.
        eps = 1e-5
        while True:
            edge = (kl > eps).float()
            if edge.sum() > max_n:
                eps *= 1.2
            else:
                break
        edge = F.conv2d(edge, torch.ones(1, 1, 3, 3, device=logits.device), padding=1)
        return edge[:, 0] > 0

    def forward(self, logits, dist):
        logits = logits.float()
        bound = self.predicted_boundary(logits)
        if bound.sum() < 1:
            return logits.sum() * 0
        n, r, c = torch.nonzero(bound).T
        big = 1e5
        dist_p = F.pad(dist, (1, 1, 1, 1), value=big)
        logits_p = F.pad(logits, (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)
        center = logits.permute(0, 2, 3, 1)[n, r, c]

        dists, kls = [], []
        for dr, dc in zip(self.DR, self.DC):
            dists.append(dist_p[n, r + dr + 1, c + dc + 1])
            if dr or dc:
                nb = logits_p[n, r + dr + 1, c + dc + 1]
                if self.detach:
                    nb = nb.detach()
                kls.append(_kl(center, nb).sum(1))
        direction_gt = torch.stack(dists).argmin(0)
        keep = direction_gt != 8
        if not keep.any():
            return logits.sum() * 0
        direction_gt = direction_gt[keep]
        direction_pred = torch.stack(kls, 1)[keep]
        weight = dist[n, r, c][keep].clamp(max=self.theta) / self.theta

        # Label-smoothed CE exactly as the authors' LabelSmoothSoftmaxCEV1:
        # 1 - eps on the true direction, eps / 8 on every other one.
        k = direction_pred.shape[1]
        target = torch.full_like(direction_pred, self.label_smoothing / k)
        target.scatter_(1, direction_gt[:, None], 1 - self.label_smoothing)
        ce = -(target * F.log_softmax(direction_pred, 1)).sum(1)
        return (ce * weight).mean()
