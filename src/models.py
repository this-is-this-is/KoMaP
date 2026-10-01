"""Model zoo: SegFormer, MicroNet-pretrained U-Net++, and a DINO ViT + mask head.

Every model here takes (B, 3, H, W) and returns logits (B, 4, h, w). Heads that
stride down are upsampled by the caller, so the training loop does not need to
know which family it is holding.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 4

MICRONET_URLS = {
    "efficientnet-b3": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                       "efficientnet-b3_pretrained_imagenet-microscopynet_v1.0.pth.tar",
    "efficientnet-b4": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                       "efficientnet-b4_pretrained_imagenet-microscopynet_v1.0.pth.tar",
    "resnet50": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                "resnet50_pretrained_microscopynet_v1.1.pth.tar",
}


DINOV3_WEIGHTS = {
    "dinov3_vitb16": "weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
}
DINOV3_REPO = "third_party_dinov3"


def _build_dinov3(name, weights=None, repo_dir=None):
    """Instantiate a DINOv3 backbone from the authors' repo and local weights.

    The weights are gated: they are requested from Meta and downloaded by hand,
    so there is no automatic fetch here -- point DINOV3_WEIGHTS at the files.
    """
    import importlib
    import sys

    root = os.path.abspath(repo_dir or DINOV3_REPO)
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"DINOv3 repo not found at {root}. Clone it with:\n"
            f"  git clone https://github.com/facebookresearch/dinov3 {root}")
    if root not in sys.path:
        sys.path.insert(0, root)

    ckpt = os.path.abspath(weights or DINOV3_WEIGHTS[name])
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"DINOv3 weights not found at {ckpt}. They are gated: request access at "
            "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/")

    backbones = importlib.import_module("dinov3.hub.backbones")
    return getattr(backbones, name)(pretrained=True, weights=ckpt)


class SegFormerWrap(nn.Module):
    """HuggingFace SegFormer, returning logits at the input resolution."""

    def __init__(self, name="nvidia/mit-b2", num_classes=NUM_CLASSES):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation

        self.net = SegformerForSemanticSegmentation.from_pretrained(
            name,
            num_labels=num_classes,
            # The phases are ordinary foreground classes; nothing may be dropped.
            # ADE20K-style label reduction would silently delete class 0 (Al).
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        # SegFormer predicts at 1/4 scale by design.
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


class DinoSegmenter(nn.Module):
    """A DINO ViT backbone with a lightweight multi-scale mask head.

    EoMT proper reuses the ViT's own late blocks with learned queries; that
    needs the authors' code. This keeps the part that matters for a 70-image
    dataset -- a strong frozen-or-finetuned ViT feature extractor -- and puts a
    small decoder on top, upsampling patch tokens back to pixels.
    """

    def __init__(self, name="facebook/dinov2-base", num_classes=NUM_CLASSES,
                 freeze_encoder=False, weights=None, repo_dir=None):
        super().__init__()
        # DINOv3 ships as Meta's own checkpoint format (cls + storage tokens,
        # keys like patch_embed.proj.weight), which AutoModel cannot read, so
        # it goes through the authors' backbone code instead.
        self.is_v3 = name.startswith("dinov3_")
        if self.is_v3:
            self.backbone = _build_dinov3(name, weights, repo_dir)
            self.patch = self.backbone.patch_size
            dim = self.backbone.embed_dim
        else:
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(name)
            self.patch = getattr(self.backbone.config, "patch_size", 14)
            dim = self.backbone.config.hidden_size

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Patch tokens land at 1/16 of the input. Since 93% of this dataset's
        # errors sit within 2px of a phase boundary, the head learns a 2x
        # upsample before the final bilinear rather than jumping 16x at once.
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        # ViT needs both sides to be a whole number of patches.
        ph, pw = h // self.patch * self.patch, w // self.patch * self.patch
        if (ph, pw) != (h, w):
            x = F.interpolate(x, size=(ph, pw), mode="bilinear", align_corners=False)

        gh, gw = ph // self.patch, pw // self.patch
        with torch.set_grad_enabled(self.training and not self.freeze_encoder):
            if self.is_v3:
                # forward_features hands back the patch grid on its own, with
                # the cls and storage tokens already separated out.
                tokens = self.backbone.forward_features(x)["x_norm_patchtokens"]
            else:
                out = self.backbone(pixel_values=x).last_hidden_state
                # Drop CLS (and DINOv2's register tokens) to keep the grid.
                tokens = out[:, out.shape[1] - gh * gw:]

        feat = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, gh, gw)
        return F.interpolate(self.head(feat), size=(h, w), mode="bilinear", align_corners=False)


class FullResSkip(nn.Module):
    """smp U-Net / U-Net++ plus a stride-1 branch into the segmentation head.

    smp's decoders drop the encoder's stride-1 feature (the image itself), so
    their last block only nearest-upsamples stride-2 features into 16 channels
    before the head. Eutectic Si averages 3.1 px across, so its boundary is
    being placed from features that never saw it at full resolution. A shallow
    conv branch on the input restores that view (cf. FR-UNet for thin vessels).
    """

    def __init__(self, net, channels=32):
        super().__init__()
        # smp's attribute names are kept, so the "encoder." LR group in
        # train_v2.py and the MicroNet loader still find the backbone.
        self.encoder = net.encoder
        self.decoder = net.decoder
        self.segmentation_head = net.segmentation_head
        dec = net.segmentation_head[0].in_channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(dec + channels, dec, 3, padding=1, bias=False), nn.BatchNorm2d(dec), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        d = self.decoder(self.encoder(x))
        return self.segmentation_head(self.fuse(torch.cat([d, self.stem(x)], 1)))


def _load_micronet(encoder_module, encoder_name, weights_dir="weights"):
    """Fetch and apply NASA's microscopy-pretrained encoder weights."""
    url = MICRONET_URLS.get(encoder_name)
    if url is None:
        raise ValueError(f"no MicroNet weights published for {encoder_name}; "
                         f"available: {sorted(MICRONET_URLS)}")
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, os.path.basename(url))
    if not os.path.exists(path):
        torch.hub.download_url_to_file(url, path)

    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    target = encoder_module.state_dict()
    keep = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
    encoder_module.load_state_dict(keep, strict=False)
    return len(keep), len(target)


def build(arch="unetpp", encoder="efficientnet-b3", pretrain="micronet",
          num_classes=NUM_CLASSES, freeze_encoder=False, fullres_skip=False, fullres_ch=32):
    """Build a model.

    arch: segformer_b2 | segformer_b4 | unet | unetpp | deeplabv3p |
          dinov2 | dinov3 (= dinov3_vitb16) | dinov3_vitl16
    pretrain: micronet | image-micronet | imagenet | none  (smp architectures)
    fullres_skip: add a stride-1 branch before the head (unet / unetpp only)
    """
    if fullres_skip and arch not in ("unet", "unetpp"):
        raise ValueError("fullres_skip needs a full-resolution decoder: unet or unetpp")
    if arch.startswith("segformer"):
        return SegFormerWrap({"segformer_b2": "nvidia/mit-b2",
                              "segformer_b4": "nvidia/mit-b4"}[arch], num_classes)

    if arch == "dinov2":
        return DinoSegmenter("facebook/dinov2-base", num_classes, freeze_encoder)
    if arch in ("dinov3", "dinov3_vitb16", "dinov3_vitl16"):
        name = "dinov3_vitb16" if arch == "dinov3" else arch
        return DinoSegmenter(name, num_classes, freeze_encoder)

    import segmentation_models_pytorch as smp

    factory = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "deeplabv3p": smp.DeepLabV3Plus}
    if arch not in factory:
        raise ValueError(f"unknown arch {arch}")

    # MicroNet weights replace the encoder wholesale, so start from random and
    # load them in; otherwise fall back to ImageNet.
    smp_weights = "imagenet" if pretrain == "imagenet" else None
    model = factory[arch](encoder_name=encoder, encoder_weights=smp_weights,
                          in_channels=3, classes=num_classes)

    if pretrain in ("micronet", "image-micronet"):
        got, total = _load_micronet(model.encoder, encoder)
        print(f"  MicroNet weights: {got}/{total} encoder tensors loaded")

    return FullResSkip(model, fullres_ch) if fullres_skip else model
