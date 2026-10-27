"""Model zoo: SegFormer, MicroNet-pretrained U-Net++, and a DINO ViT + mask head.

Every model here takes (B, 3, H, W) and returns logits (B, 4, h, w). Heads that
stride down are upsampled by the caller, so the training loop does not need to
know which family it is holding.
"""
import hashlib
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
    # The encoder NASA's own benchmark ranks first with U-Net++ (Stuckner et al., 2022);
    # the ImageNet -> MicroNet variant, like the EfficientNets above.
    "se_resnext50_32x4d": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                          "se_resnext50_32x4d_pretrained_imagenet-microscopynet_v1.0.pth.tar",
    # After SE-ResNeXt-50 beat EfficientNet-B3: its deeper sibling, and two encoders built
    # differently (inception branches; residual + dense paths). The same variant as above.
    "se_resnext101_32x4d": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                           "se_resnext101_32x4d_pretrained_imagenet-microscopynet_v1.0.pth.tar",
    "inceptionresnetv2": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
                         "inceptionresnetv2_pretrained_imagenet-microscopynet_v1.0.pth.tar",
    "dpn92": "https://nasa-public-data.s3.amazonaws.com/microscopy_segmentation_models/"
             "dpn92_pretrained_imagenet-microscopynet_v1.0.pth.tar",
}

# MatSSL's public "all metallography datasets" encoder.  Pinning the Hugging
# Face revision and checking the digest makes a queued experiment reproducible
# even if the model hub entry is updated later.
MATSSL_URL = (
    "https://huggingface.co/dhp510/MatSSL/resolve/"
    "40465a494399de7cc21a16c056a9009a9c73d2c8/model_hub/ssl-models/matssl/"
    "matssl-aachen-uhcs-metaldam.pth"
)
MATSSL_SHA256 = "3bac102385d019172afc67a04c688447750102be4bcc09eec54bede77badebe2"


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

    edge_head adds a 1x1 conv on the same fused feature the segmentation head
    reads, trained to mark class boundaries (team experiment E3). It only
    shapes that shared feature during training: in train mode forward returns
    (logits, edge_logits), in eval mode the logits alone, so inference and
    checkpoint loading see the plain model plus one unused conv.
    """

    def __init__(self, net, channels=32, edge_head=False):
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
        self.edge_head = nn.Conv2d(dec, 1, 1) if edge_head else None

    def forward(self, x):
        d = self.decoder(self.encoder(x))
        f = self.fuse(torch.cat([d, self.stem(x)], 1))
        if self.training and self.edge_head is not None:
            return self.segmentation_head(f), self.edge_head(f)
        return self.segmentation_head(f)


def _conv_bn_relu(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class HRNetSeg(nn.Module):
    """HRNetV2 for segmentation, decoded up to the input resolution.

    HRNet keeps a 1/4-resolution stream alive from the first stage to the last
    and keeps exchanging information with coarser streams that run beside it,
    instead of shrinking the image and restoring it: the one strong family
    here that is not U-shaped, which is what makes it worth having next to the
    U-Net++ models in an ensemble. The four streams are upsampled to 1/4 and
    concatenated, as in the HRNetV2 head (Wang et al., TPAMI 2020).

    1/4 is where that head stops, and where SegFormer lost its points on this
    data -- a 3 px lamella needs finer features. So two light steps follow, each
    doubling the resolution and fusing what the image looked like there: the
    backbone's own stride-2 stem feature, then a stride-1 conv branch on the
    input (the FullResSkip idea, built in rather than optional).
    """

    def __init__(self, name="hrnet_w18", num_classes=NUM_CLASSES, pretrained=True, fullres_ch=32):
        super().__init__()
        import timm

        # "encoder", so train_v2.py's encoder LR group finds the backbone.
        # feature_location="" hands back the stem and the four raw streams
        # rather than the widened features of the classification head.
        self.encoder = timm.create_model(name, pretrained=pretrained, features_only=True,
                                         feature_location="")
        stem, *streams = self.encoder.feature_info.channels()
        self.at_quarter = _conv_bn_relu(sum(streams), 128)
        self.at_half = _conv_bn_relu(128 + stem, 64)
        self.stem = nn.Sequential(_conv_bn_relu(3, fullres_ch), _conv_bn_relu(fullres_ch, fullres_ch))
        self.fuse = _conv_bn_relu(64 + fullres_ch, 32)
        self.segmentation_head = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        stem, *streams = self.encoder(x)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        f = self.at_quarter(torch.cat([streams[0]] + [up(s, streams[0]) for s in streams[1:]], 1))
        f = self.at_half(torch.cat([up(f, stem), stem], 1))
        f = self.fuse(torch.cat([up(f, x), self.stem(x)], 1))
        return self.segmentation_head(f)


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
    # Only a leading "module." (DataParallel) is dropped. Replacing it anywhere would also
    # turn SENet's se_module.fc1 into se_fc1 and leave its 64 squeeze-excitation tensors unloaded.
    sd = {k.removeprefix("module."): v for k, v in sd.items()}

    target = encoder_module.state_dict()
    keep = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
    encoder_module.load_state_dict(keep, strict=False)
    return len(keep), len(target)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_matssl(encoder_module, encoder_name, weights_dir="weights"):
    """Load the authors' MatSSL ResNet-50 backbone into an SMP encoder.

    The released checkpoint also registers duplicate ``layer*`` modules and
    its SSL-only projection/gating heads.  ``backbone.*`` is the one complete
    ResNet state and matches SMP 0.5.0 exactly (318 tensors).
    """
    if encoder_name != "resnet50":
        raise ValueError("MatSSL publishes a ResNet-50 encoder; use --encoder resnet50")
    directory = os.path.join(weights_dir, "matssl")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, os.path.basename(MATSSL_URL))
    if not os.path.exists(path) or _sha256(path) != MATSSL_SHA256:
        temporary = path + ".part"
        if os.path.exists(temporary):
            os.remove(temporary)
        torch.hub.download_url_to_file(MATSSL_URL, temporary)
        actual = _sha256(temporary)
        if actual != MATSSL_SHA256:
            os.remove(temporary)
            raise RuntimeError(f"MatSSL SHA-256 mismatch: expected {MATSSL_SHA256}, got {actual}")
        os.replace(temporary, path)

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    state = {key[len("backbone."):]: value for key, value in state.items()
             if key.startswith("backbone.")}
    encoder_module.load_state_dict(state, strict=True)
    return len(state), path


def load_encoder_checkpoint(encoder_module, path):
    """Strictly load an encoder produced by ``pretrain_cross_microscope.py``."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format") != "komap_encoder_v1":
        raise ValueError(f"unsupported encoder checkpoint format in {path}")
    state = value.get("encoder", value.get("state_dict", value))
    if not isinstance(state, dict):
        raise ValueError(f"no encoder state dictionary in {path}")
    # Accept the two common wrappers while keeping the actual load strict.
    for prefix in ("module.encoder.", "encoder.", "module."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: tensor for key, tensor in state.items()}
            break
    encoder_module.load_state_dict(state, strict=True)
    return len(state)


def load_transfer_checkpoint(model, path):
    """Load a segmentation pretrain while deliberately resetting class heads.

    NASA Super1-4 has three labels whereas KoMaP has four.  All compatible
    encoder, decoder, full-resolution stem and fusion tensors transfer; the
    segmentation and optional edge heads always start fresh.
    """
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format") != "komap_segmentation_transfer_v1":
        raise ValueError(f"unsupported segmentation transfer checkpoint format in {path}")
    source = value.get("model", value.get("state_dict", value))
    if not isinstance(source, dict):
        raise ValueError(f"no model state dictionary in {path}")
    source = {key.removeprefix("module."): tensor for key, tensor in source.items()}
    target = model.state_dict()
    reset = ("segmentation_head.", "edge_head.")
    keep = {key: tensor for key, tensor in source.items()
            if not key.startswith(reset) and key in target and tensor.shape == target[key].shape}
    missing_body = [key for key in target if not key.startswith(reset) and key not in keep]
    if missing_body:
        preview = ", ".join(missing_body[:5])
        raise ValueError(f"transfer checkpoint is incompatible; missing body tensors: {preview}")
    model.load_state_dict(keep, strict=False)
    return len(keep), len(target)


def build(arch="unetpp", encoder="efficientnet-b3", pretrain="micronet",
          num_classes=NUM_CLASSES, freeze_encoder=False, fullres_skip=False, fullres_ch=32,
          edge_head=False, decoder_attention=None):
    """Build a model.

    arch: segformer_b2 | segformer_b4 | unet | unetpp | deeplabv3p | hrnet |
          dinov2 | dinov3 (= dinov3_vitb16) | dinov3_vitl16
          (hrnet: encoder hrnet_w18 | hrnet_w32 | ..., pretrain imagenet | none;
          its full-resolution branch is built in)
    pretrain: micronet | image-micronet | matssl | imagenet | none  (smp architectures)
    fullres_skip: add a stride-1 branch before the head (unet / unetpp only)
    edge_head: auxiliary class-boundary head on the fused feature (needs fullres_skip)
    decoder_attention: "scse" puts smp's concurrent spatial and channel
        squeeze-and-excitation (Roy et al., MICCAI 2018) in every decoder block,
        on the concatenated skip input and on the block output (unet / unetpp only)
    """
    if fullres_skip and arch not in ("unet", "unetpp"):
        raise ValueError("fullres_skip needs a full-resolution decoder: unet or unetpp")
    if edge_head and not fullres_skip:
        raise ValueError("edge_head is attached to the fullres_skip fused feature")
    if decoder_attention and arch not in ("unet", "unetpp"):
        raise ValueError("decoder_attention is an option of the unet / unetpp decoders")
    if arch.startswith("segformer"):
        return SegFormerWrap({"segformer_b2": "nvidia/mit-b2",
                              "segformer_b4": "nvidia/mit-b4"}[arch], num_classes)

    if arch == "hrnet":
        if pretrain not in ("imagenet", "none"):
            raise ValueError("hrnet comes with ImageNet weights only: --pretrain imagenet (or none)")
        return HRNetSeg(encoder if encoder.startswith("hrnet") else "hrnet_w18", num_classes,
                        pretrained=pretrain == "imagenet", fullres_ch=fullres_ch)

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
    # Passed only when set: DeepLabV3+ has no such argument.
    attention = {"decoder_attention_type": decoder_attention} if decoder_attention else {}
    model = factory[arch](encoder_name=encoder, encoder_weights=smp_weights,
                          in_channels=3, classes=num_classes, **attention)

    if pretrain in ("micronet", "image-micronet"):
        got, total = _load_micronet(model.encoder, encoder)
        print(f"  MicroNet weights: {got}/{total} encoder tensors loaded")
    elif pretrain == "matssl":
        got, path = _load_matssl(model.encoder, encoder)
        print(f"  MatSSL weights: {got} encoder tensors loaded from {path}")
    elif pretrain not in ("imagenet", "none"):
        raise ValueError(f"unknown pretraining source {pretrain}")

    return FullResSkip(model, fullres_ch, edge_head) if fullres_skip else model
