"""Train and evaluate a small B3 + SE-ResNeXt OOF logit fusion head.

Inputs are probabilities saved by run_group_cv.py --stage oof. Each base
probability is out-of-fold at the A-ID specimen level. Multiple seeds within a
model family are averaged before the fusion head sees them.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from group_cv import mask_path
from komap import CLASSES, NUM_CLASSES, per_image_miou, rgb_to_index
from logit_fusion import LogitFusionHead, normalise_probability
from postprocess import si_particle_vote

SIGNATURE_KEYS = ("tile", "overlap", "flip_tta", "d4", "scale_tta")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probability_files(root):
    grouped = defaultdict(list)
    for path in Path(root).rglob("*_prob.npz"):
        grouped[path.name.replace("_prob.npz", "_image.png")].append(path)
    if not grouped:
        raise FileNotFoundError(f"no *_prob.npz files under {root}")
    return dict(grouped)


def inference_signature(roots):
    signatures = []
    for root in roots:
        for path in Path(root).rglob("scores.json"):
            value = json.loads(path.read_text())
            signatures.append(({key: value.get(key) for key in SIGNATURE_KEYS}, path))
    if not signatures:
        raise FileNotFoundError("no scores.json found beside OOF probabilities")
    expected, source = signatures[0]
    for value, path in signatures[1:]:
        if value != expected:
            raise ValueError(f"OOF inference settings disagree: {source} vs {path}")
    if expected["tile"] is None or expected["overlap"] is None:
        raise ValueError(f"incomplete inference signature in {source}")
    return expected


def load_average(paths):
    total, source, shape = None, None, None
    for path in sorted(paths):
        with np.load(path) as value:
            probability = value["prob"].astype(np.float32)
            image = Path(str(value["image"].item())) if "image" in value.files else None
        if probability.ndim != 3 or probability.shape[0] != NUM_CLASSES:
            raise ValueError(f"expected [4,H,W] probability in {path}, got {probability.shape}")
        if shape is None:
            shape = probability.shape
        elif probability.shape != shape:
            raise ValueError(f"probability shapes disagree for {path.name}")
        if image is None:
            raise ValueError(f"OOF probability does not identify its source image: {path}")
        if source is None:
            source = image
        elif image.resolve() != source.resolve():
            raise ValueError(f"OOF sources disagree for {path.name}: {source} vs {image}")
        total = probability if total is None else total + probability
    return (total / len(paths)).astype(np.float16), source


def load_folds(path):
    value = json.loads(Path(path).read_text())
    if value.get("format") != "komap_group_folds_v1":
        raise ValueError(f"unsupported fold manifest: {path}")
    held_out = [i for fold in value["folds"] for i in fold["valid_indices"]]
    if sorted(held_out) != list(range(len(value["records"]))):
        raise ValueError("fold manifest must hold every record out exactly once")
    return value


def load_inputs(b3_root, se_root, folds):
    b3_files, se_files = probability_files(b3_root), probability_files(se_root)
    expected = {row["name"] for row in folds["records"]}
    for family, found in (("B3", set(b3_files)), ("SE-ResNeXt", set(se_files))):
        missing, extra = sorted(expected - found), sorted(found - expected)
        if missing or extra:
            detail = []
            if missing:
                detail.append(f"missing {len(missing)} ({', '.join(missing[:3])})")
            if extra:
                detail.append(f"extra {len(extra)} ({', '.join(extra[:3])})")
            raise ValueError(f"{family} OOF set is incomplete: {'; '.join(detail)}")

    rows = []
    print(f"loading {len(expected)} paired OOF scans into memory ...", flush=True)
    for position, record in enumerate(folds["records"], 1):
        name = record["name"]
        b3, image_b3 = load_average(b3_files[name])
        se, image_se = load_average(se_files[name])
        if image_b3.resolve() != image_se.resolve():
            raise ValueError(f"base families identify different source images for {name}")
        mask_file = mask_path(image_b3)
        mask_bgr = cv2.imread(str(mask_file), cv2.IMREAD_COLOR)
        if mask_bgr is None:
            raise FileNotFoundError(mask_file)
        target = rgb_to_index(cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB))
        if b3.shape != se.shape or b3.shape[1:] != target.shape:
            raise ValueError(f"probability/target shape mismatch for {name}: "
                             f"{b3.shape}, {se.shape}, {target.shape}")
        rows.append({"name": name, "group": record["group"], "image": str(image_b3),
                     "b3": b3, "se": se, "target": target,
                     "b3_members": len(b3_files[name]), "se_members": len(se_files[name])})
        if position % 10 == 0 or position == len(expected):
            print(f"  loaded {position}/{len(expected)}", flush=True)
    return rows


class FusionCropDataset(Dataset):
    def __init__(self, rows, indices, tile=512, samples_per_image=4, seed=0):
        self.rows, self.indices = rows, list(indices)
        self.tile, self.samples = tile, samples_per_image
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.indices) * self.samples

    def __getitem__(self, item):
        row = self.rows[self.indices[item // self.samples]]
        rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + 9_973 * item)
        b3, se, target = row["b3"], row["se"], row["target"]
        height, width = target.shape
        pad_h, pad_w = max(0, self.tile - height), max(0, self.tile - width)
        if pad_h or pad_w:
            pad = ((0, 0), (0, pad_h), (0, pad_w))
            b3 = np.pad(b3, pad, mode="reflect")
            se = np.pad(se, pad, mode="reflect")
            target = np.pad(target, ((0, pad_h), (0, pad_w)), mode="reflect")
            height, width = target.shape
        y = int(rng.integers(0, height - self.tile + 1))
        x = int(rng.integers(0, width - self.tile + 1))
        b3 = b3[:, y:y + self.tile, x:x + self.tile]
        se = se[:, y:y + self.tile, x:x + self.tile]
        target = target[y:y + self.tile, x:x + self.tile]
        rotation, flip = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
        if rotation:
            b3 = np.rot90(b3, rotation, axes=(1, 2))
            se = np.rot90(se, rotation, axes=(1, 2))
            target = np.rot90(target, rotation, axes=(0, 1))
        if flip:
            b3, se, target = b3[:, :, ::-1], se[:, :, ::-1], target[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(b3)).float(),
                torch.from_numpy(np.ascontiguousarray(se)).float(),
                torch.from_numpy(np.ascontiguousarray(target)).long())


def dice_loss(logits, target):
    probability = logits.float().softmax(1)
    onehot = F.one_hot(target, NUM_CLASSES).permute(0, 3, 1, 2).float()
    dims, smooth = (0, 2, 3), 1.0
    intersection = (probability * onehot).sum(dims)
    denominator = probability.sum(dims) + onehot.sum(dims)
    return 1 - ((2 * intersection + smooth) / (denominator + smooth)).mean()


def train_head(rows, indices, args, device, seed):
    seed_everything(seed)
    model = LogitFusionHead(NUM_CLASSES, args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dataset = FusionCropDataset(rows, indices, args.crop, args.samples_per_image, seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        generator=generator, persistent_workers=False)
    history = []
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        total, batches = 0.0, 0
        for b3, se, target in loader:
            b3 = b3.to(device, non_blocking=True)
            se = se.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(b3, se)
                loss = (0.5 * F.cross_entropy(logits.float(), target)
                        + 0.5 * dice_loss(logits, target))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        mean = total / max(batches, 1)
        history.append({"epoch": epoch + 1, "loss": mean,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"    epoch {epoch + 1:3d}/{args.epochs}: loss {mean:.5f}", flush=True)
    return model.eval(), history


@torch.inference_mode()
def fused_probability(model, row, device):
    b3 = torch.from_numpy(row["b3"].astype(np.float32))[None].to(device)
    se = torch.from_numpy(row["se"].astype(np.float32))[None].to(device)
    return model(b3, se).softmax(1)[0].cpu().numpy()


def score_probability(probability, target):
    return per_image_miou(si_particle_vote(probability), target)


def metric_summary(rows):
    scores = np.array([row["miou"] for row in rows], np.float64)
    per_class = np.array([
        [np.nan if value is None else value for value in row["per_class"]]
        for row in rows
    ], np.float64)
    return {"mean_miou": float(scores.mean()),
            "per_class_iou": {name: float(value) for name, value in
                              zip(CLASSES, np.nanmean(per_class, axis=0))}}


def paired_bootstrap(rows, seed=0, samples=20_000):
    by_name, groups = defaultdict(dict), {}
    for row in rows:
        by_name[row["name"]][row["method"]] = row["miou"]
        groups[row["name"]] = row["group"]
    names = sorted(by_name)
    differences = np.array([by_name[name]["fusion"] - by_name[name]["equal"]
                            for name in names], np.float64)
    rng = np.random.default_rng(seed)
    image_boot = differences[
        rng.integers(0, len(differences), (samples, len(differences)))].mean(1)
    unique_groups = sorted({groups[name] for name in names})
    membership = np.array([groups[name] for name in names])
    group_sums = np.array([differences[membership == group].sum()
                           for group in unique_groups])
    group_counts = np.array([(membership == group).sum() for group in unique_groups])
    draws = rng.integers(0, len(unique_groups), (samples, len(unique_groups)))
    # Preserve the competition's per-image estimand even though whole A-ID
    # clusters, rather than individual scans, are resampled.
    cluster_boot = group_sums[draws].sum(1) / group_counts[draws].sum(1)
    return {"candidate": "fusion", "reference": "equal",
            "delta_pp": float(100 * differences.mean()),
            "image_bootstrap_95ci_pp":
                (100 * np.percentile(image_boot, [2.5, 97.5])).tolist(),
            "group_bootstrap_95ci_pp":
                (100 * np.percentile(cluster_boot, [2.5, 97.5])).tolist(),
            "images": len(names), "groups": len(unique_groups), "samples": samples}


def checkpoint_payload(model, history, args, signature, rows, indices, stage, fold=None):
    return {"format": "komap_logit_fusion_v1", "model": model.state_dict(),
            "classes": NUM_CLASSES, "hidden": args.hidden, "stage": stage, "fold": fold,
            "train_indices": list(indices), "history": history,
            "inference_signature": signature,
            "base_roots": {"b3": str(args.b3_oof), "se": str(args.se_oof)},
            "members_per_family": {
                "b3": sorted({row["b3_members"] for row in rows}),
                "se": sorted({row["se_members"] for row in rows})},
            "training": {"epochs": args.epochs, "crop": args.crop,
                         "samples_per_image": args.samples_per_image,
                         "batch_size": args.batch_size, "lr": args.lr,
                         "weight_decay": args.weight_decay, "seed": args.seed}}


def cross_validate(rows, folds, args, signature, device):
    output = args.out / "cv_results.json"
    if output.is_file() and not args.force:
        print(f"skip: {output} (use --force to recompute)")
        return json.loads(output.read_text())
    results = []
    for fold in folds["folds"]:
        number = fold["fold"]
        print(f"\nmeta fold {number}: train {len(fold['train_indices'])}, "
              f"held out {len(fold['valid_indices'])}", flush=True)
        model, history = train_head(rows, fold["train_indices"], args, device,
                                    args.seed + number)
        torch.save(checkpoint_payload(model, history, args, signature, rows,
                                      fold["train_indices"], "crossfit", number),
                   args.out / f"fold_{number}.pth")
        for index in fold["valid_indices"]:
            row = rows[index]
            b3 = normalise_probability(
                torch.from_numpy(row["b3"].astype(np.float32))[None])[0]
            se = normalise_probability(
                torch.from_numpy(row["se"].astype(np.float32))[None])[0]
            probabilities = {"b3": b3.numpy(), "se": se.numpy(),
                             "equal": ((b3 + se) * 0.5).numpy(),
                             "fusion": fused_probability(model, row, device)}
            image_scores = {}
            for method, probability in probabilities.items():
                miou, per_class = score_probability(probability, row["target"])
                image_scores[method] = miou
                results.append({"name": row["name"], "group": row["group"],
                                "fold": number, "method": method, "miou": miou,
                                "per_class": [None if np.isnan(v) else float(v)
                                              for v in per_class]})
            print(f"  {row['name']:42s} fusion {100*image_scores['fusion']:6.2f}%  "
                  f"vs equal {100*(image_scores['fusion']-image_scores['equal']):+6.2f}pp",
                  flush=True)

    summaries = {method: metric_summary(
        [row for row in results if row["method"] == method])
        for method in ("b3", "se", "equal", "fusion")}
    comparison = paired_bootstrap(results, seed=args.seed)
    payload = {"format": "komap_oof_logit_fusion_cv_v1", "summaries": summaries,
               "fusion_vs_equal": comparison, "inference_signature": signature,
               "per_image": results}
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print("\nOOF cross-fit summary")
    for method, value in summaries.items():
        print(f"  {method:7s}: {100*value['mean_miou']:.2f}%")
    print(f"  fusion - equal: {comparison['delta_pp']:+.2f}pp  "
          f"A-ID bootstrap 95% CI {comparison['group_bootstrap_95ci_pp']}")
    return payload


def train_final(rows, args, signature, device):
    output = args.out / "fusion_final.pth"
    if output.is_file() and not args.force:
        print(f"skip: {output} (use --force to recompute)")
        return
    print(f"\nfinal fusion fit: all {len(rows)} OOF scans", flush=True)
    indices = list(range(len(rows)))
    model, history = train_head(rows, indices, args, device, args.seed + 10_000)
    torch.save(checkpoint_payload(model, history, args, signature, rows, indices,
                                  "final"), output)
    print(f"final fusion checkpoint -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("cv", "final", "all"), default="all")
    parser.add_argument("--b3_oof", type=Path,
                        default=ROOT / "runs/group_cv/oof/b3_aug3")
    parser.add_argument("--se_oof", type=Path,
                        default=ROOT / "runs/group_cv/oof/seresnext50_aug3")
    parser.add_argument("--folds_json", type=Path,
                        default=ROOT / "runs/group_cv/manifests/folds.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/oof_logit_fusion")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--samples_per_image", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.crop, args.samples_per_image,
           args.batch_size, args.hidden) <= 0:
        parser.error("epochs/crop/samples_per_image/batch_size/hidden must be positive")
    if args.workers < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("workers/learning-rate/weight-decay are invalid")
    args.b3_oof, args.se_oof = args.b3_oof.resolve(), args.se_oof.resolve()
    args.folds_json, args.out = args.folds_json.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = load_folds(args.folds_json)
    signature = inference_signature([args.b3_oof, args.se_oof])
    rows = load_inputs(args.b3_oof, args.se_oof, folds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  OOF signature={signature}")
    if args.stage in ("cv", "all"):
        cross_validate(rows, folds, args, signature, device)
    if args.stage in ("final", "all"):
        train_final(rows, args, signature, device)


if __name__ == "__main__":
    main()
