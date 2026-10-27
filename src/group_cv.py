"""Grouped cross-validation manifests for the labelled KoMaP scans.

The prefix before the first underscore (``A125`` in
``A125_step1_sample4_image.png``) is treated as the specimen/group id.  A
group is never split between training and validation.  Manifests contain paths
relative to ``data_root`` so they remain usable when the repository moves.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


GROUP_RE = re.compile(r"^(A\d+)(?:_|$)")


def specimen_group(name: str) -> str:
    """Return the specimen id encoded at the start of a KoMaP filename."""
    match = GROUP_RE.match(Path(name).name)
    if not match:
        raise ValueError(f"cannot read A### specimen group from {name!r}")
    return match.group(1)


def mask_path(image_path: str | Path) -> Path:
    """Corresponding mask for an image in an ``images`` directory."""
    image_path = Path(image_path)
    if image_path.parent.name != "images":
        raise ValueError(f"image is not inside an images directory: {image_path}")
    return image_path.parent.parent / "masks" / image_path.name.replace("_image.png", "_mask.png")


def discover_labelled(data_root: str | Path, splits=("train/train", "valid/valid")) -> list[dict]:
    """Find labelled scans and retain their original split for diagnostics."""
    root = Path(data_root).resolve()
    records = []
    seen = set()
    for split in splits:
        for image in sorted((root / split / "images").glob("*.png")):
            relative = image.resolve().relative_to(root)
            if image.name in seen:
                raise ValueError(f"duplicate labelled filename across splits: {image.name}")
            seen.add(image.name)
            mask = mask_path(image)
            if not mask.is_file():
                raise FileNotFoundError(f"mask missing for {image}: {mask}")
            records.append({"name": image.name, "group": specimen_group(image.name),
                            "image": relative.as_posix(), "source_split": split})
    if not records:
        raise FileNotFoundError(f"no labelled images under {root}")
    return records


def grouped_folds(records: list[dict], n_splits=5, seed=0) -> list[dict]:
    """Return balanced GroupKFold indices, with optional deterministic shuffle."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = [row["group"] for row in records]
    unique = sorted(set(groups))
    if len(unique) < n_splits:
        raise ValueError(f"{len(unique)} groups cannot make {n_splits} folds")

    # sklearn 1.6+ supports shuffled GroupKFold.  Keep a deterministic fallback
    # for environments with an older sklearn rather than silently splitting a
    # specimen by image.
    try:
        from sklearn.model_selection import GroupKFold
        try:
            splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        except TypeError:
            splitter = GroupKFold(n_splits=n_splits)
        pairs = splitter.split(records, groups=groups)
        folds = []
        for fold, (train, valid) in enumerate(pairs):
            folds.append({"fold": fold, "train_indices": train.tolist(),
                          "valid_indices": valid.tolist()})
    except ImportError:
        import numpy as np
        rng = np.random.default_rng(seed)
        by_group = {group: [] for group in unique}
        for index, group in enumerate(groups):
            by_group[group].append(index)
        shuffled = list(unique)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda group: -len(by_group[group]))
        bins, counts = [[] for _ in range(n_splits)], [0] * n_splits
        for group in shuffled:
            fold = min(range(n_splits), key=lambda value: (counts[value], value))
            bins[fold].extend(by_group[group])
            counts[fold] += len(by_group[group])
        all_indices = set(range(len(records)))
        folds = []
        for fold, valid in enumerate(bins):
            valid = sorted(valid)
            folds.append({"fold": fold, "train_indices": sorted(all_indices - set(valid)),
                          "valid_indices": valid})

    held_out = []
    for item in folds:
        train_groups = {groups[i] for i in item["train_indices"]}
        valid_groups = {groups[i] for i in item["valid_indices"]}
        if train_groups & valid_groups:
            raise AssertionError(f"group leakage in fold {item['fold']}: {train_groups & valid_groups}")
        item["train_groups"] = sorted(train_groups)
        item["valid_groups"] = sorted(valid_groups)
        held_out.extend(item["valid_indices"])
    if sorted(held_out) != list(range(len(records))):
        raise AssertionError("each labelled image must be held out exactly once")
    return folds


def write_manifest(path: str | Path, records: list[dict], data_root: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {"format": "komap_image_manifest_v1", "data_root": str(Path(data_root).resolve()),
             "records": records}
    path.write_text(json.dumps(value, indent=2) + "\n")


def read_manifest(path: str | Path, data_root: str | Path | None = None) -> list[str]:
    """Resolve image paths from a v1 JSON manifest or a plain text list."""
    path = Path(path)
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        rows = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
        root = Path(data_root).resolve() if data_root else path.parent.resolve()
        return [str(Path(row) if Path(row).is_absolute() else root / row) for row in rows]

    if isinstance(value, list):
        records, embedded_root = value, None
    elif value.get("format") == "komap_image_manifest_v1":
        records, embedded_root = value["records"], value.get("data_root")
    else:
        raise ValueError(f"unsupported manifest format: {path}")
    root = Path(data_root or embedded_root or path.parent).resolve()
    paths = []
    for row in records:
        image = row["image"] if isinstance(row, dict) else row
        image = Path(image)
        paths.append(str(image if image.is_absolute() else root / image))
    if not paths:
        raise ValueError(f"empty manifest: {path}")
    return sorted(paths)


def prepare_manifests(data_root: str | Path, out: str | Path, n_splits=5, seed=0) -> dict:
    """Discover the 90 labelled scans and write fold/all-labelled manifests."""
    out = Path(out)
    records = discover_labelled(data_root)
    folds = grouped_folds(records, n_splits=n_splits, seed=seed)
    for item in folds:
        fold = item["fold"]
        write_manifest(out / f"fold_{fold}_train.json",
                       [records[i] for i in item["train_indices"]], data_root)
        write_manifest(out / f"fold_{fold}_valid.json",
                       [records[i] for i in item["valid_indices"]], data_root)
    write_manifest(out / "all_labelled.json", records, data_root)
    summary = {"format": "komap_group_folds_v1", "data_root": str(Path(data_root).resolve()),
               "n_splits": n_splits, "seed": seed, "records": records, "folds": folds}
    out.mkdir(parents=True, exist_ok=True)
    (out / "folds.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
