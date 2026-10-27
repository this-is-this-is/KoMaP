"""Run leakage-safe grouped cross-validation and the final all-label fit.

The unit of splitting is the ``A###`` filename prefix.  The script is
restartable: completed checkpoints, OOF scores and audit output are skipped
unless ``--force`` is given.  It runs jobs sequentially on purpose because a
single scale-2 model already occupies a large GPU.

Typical flow::

    # Inspect every command without starting GPU work.
    ./.conda/bin/python run_group_cv.py --stage plan

    # Five fold models, OOF predictions/audit, then all 90 labelled scans.
    ./.conda/bin/python run_group_cv.py --stage all --seeds 0 1 2

    # Add the complementary small-scan and SE-ResNeXt recipes.
    ./.conda/bin/python run_group_cv.py --stage all \
        --recipes b3_aug3 b3_plus seresnext50_aug3 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from group_cv import prepare_manifests


COMMON_RECIPE = [
    "--arch", "unetpp",
    "--pretrain", "micronet",
    "--fullres_skip",
    "--color_aug", "oneof",
    "--input_scale", "2",
    "--prisi_paste_p", "0.25",
    "--fourier_p", "0.1",
    "--illumination_p", "0.2",
]

RECIPES = {
    "b3_aug3": ["--encoder", "efficientnet-b3", *COMMON_RECIPE],
    "b3_plus": ["--encoder", "efficientnet-b3", *COMMON_RECIPE,
                "--scale_by_height", "800:4"],
    "seresnext50_aug3": ["--encoder", "se_resnext50_32x4d", *COMMON_RECIPE],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_is_current(history_path: Path, args, recipe: str, fold: int | None,
                        seed: int, epochs: int) -> bool:
    """Avoid silently reusing a run after its budget or manifest changed."""
    if not history_path.is_file():
        return False
    try:
        saved = json.loads(history_path.read_text())["args"]
    except (KeyError, json.JSONDecodeError):
        return False
    manifests = args.work_dir / "manifests"
    train_manifest = manifests / (f"fold_{fold}_train.json" if fold is not None
                                  else "all_labelled.json")
    valid_manifest = manifests / f"fold_{fold}_valid.json" if fold is not None else None
    encoder = {"b3_aug3": "efficientnet-b3", "b3_plus": "efficientnet-b3",
               "seresnext50_aug3": "se_resnext50_32x4d"}[recipe]
    expected = {
        "epochs": epochs, "seed": seed, "arch": "unetpp", "encoder": encoder,
        "pretrain": "micronet", "fullres_skip": True, "color_aug": "oneof",
        "input_scale": 2.0, "prisi_paste_p": 0.25, "fourier_p": 0.1,
        "illumination_p": 0.2,
        "scale_by_height": ["800:4"] if recipe == "b3_plus" else None,
        "train_manifest": str(train_manifest),
        "valid_manifest": str(valid_manifest) if valid_manifest else None,
        "no_validation": fold is None,
        "train_manifest_sha256": file_sha256(train_manifest),
        "valid_manifest_sha256": file_sha256(valid_manifest) if valid_manifest else None,
    }
    if fold is not None:
        expected["val_every"] = args.val_every
    return all(saved.get(key) == value for key, value in expected.items())


def display(command: list[str]) -> str:
    return shlex.join(str(value) for value in command)


def run(command: list[str], dry_run=False) -> None:
    print(("plan: " if dry_run else "run:  ") + display(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def fold_run(work: Path, recipe: str, fold: int, seed: int) -> Path:
    return work / recipe / f"fold_{fold}_seed_{seed}"


def train_command(args, recipe: str, fold: int, seed: int) -> tuple[list[str], Path]:
    destination = fold_run(args.work_dir, recipe, fold, seed)
    manifests = args.work_dir / "manifests"
    command = [args.python, str(ROOT / "train_v2.py"),
               "--data_root", str(args.data_root),
               "--train_manifest", str(manifests / f"fold_{fold}_train.json"),
               "--valid_manifest", str(manifests / f"fold_{fold}_valid.json"),
               *RECIPES[recipe], "--epochs", str(args.epochs),
               "--val_every", str(args.val_every), "--seed", str(seed),
               "--out", str(destination)]
    return command, destination


def oof_command(args, recipe: str, fold: int, seed: int) -> tuple[list[str], Path]:
    checkpoint = fold_run(args.work_dir, recipe, fold, seed) / "best.pth"
    destination = args.work_dir / "oof" / recipe / f"seed_{seed}" / f"fold_{fold}"
    command = [args.python, str(ROOT / "predict.py"), "--ckpt", str(checkpoint),
               "--data_root", str(args.data_root),
               "--manifest", str(args.work_dir / "manifests" / f"fold_{fold}_valid.json"),
               "--tile", str(args.tile), "--overlap", str(args.overlap),
               "--tta", "--d4", "--si_vote", "--scale_tta",
               *[str(value) for value in args.scale_tta],
               "--save_dir", str(destination), "--save_prob"]
    return command, destination


def best_epoch(history_path: Path) -> int:
    value = json.loads(history_path.read_text())
    rows = value.get("history", [])
    if not rows:
        raise ValueError(f"no validation history in {history_path}")
    return int(max(rows, key=lambda row: row["miou"])["epoch"])


def selected_final_epoch(args, recipe: str) -> tuple[int, list[int]]:
    if args.final_epochs is not None:
        return args.final_epochs, []
    epochs = []
    for seed in args.seeds:
        for fold in args.folds:
            history = fold_run(args.work_dir, recipe, fold, seed) / "history.json"
            if not history.is_file():
                raise FileNotFoundError(
                    f"cannot select final epoch before fold training finishes: {history}")
            epochs.append(best_epoch(history))
    # The median is robust to a single fold peaking unusually early/late.  Use
    # conventional half-up rounding rather than Python's banker rounding.
    chosen = int(statistics.median(epochs) + 0.5)
    return max(1, chosen), epochs


def final_command(args, recipe: str, seed: int, epochs: int) -> tuple[list[str], Path]:
    destination = args.work_dir / "final" / recipe / f"seed_{seed}"
    command = [args.python, str(ROOT / "train_v2.py"),
               "--data_root", str(args.data_root),
               "--train_manifest", str(args.work_dir / "manifests" / "all_labelled.json"),
               "--no_validation", *RECIPES[recipe], "--epochs", str(epochs),
               "--seed", str(seed), "--out", str(destination)]
    return command, destination


def prepare(args) -> dict:
    summary = prepare_manifests(args.data_root, args.work_dir / "manifests",
                                n_splits=args.n_splits, seed=args.split_seed)
    print(f"manifests: {len(summary['records'])} scans, "
          f"{len({row['group'] for row in summary['records']})} groups, "
          f"{args.n_splits} folds -> {args.work_dir / 'manifests'}")
    for fold in summary["folds"]:
        print(f"  fold {fold['fold']}: train {len(fold['train_indices'])} scans/"
              f"{len(fold['train_groups'])} groups, valid {len(fold['valid_indices'])} scans/"
              f"{len(fold['valid_groups'])} groups")
    return summary


def train_folds(args, dry_run=False) -> None:
    for recipe in args.recipes:
        for seed in args.seeds:
            for fold in args.folds:
                command, destination = train_command(args, recipe, fold, seed)
                finished = destination / "history.json"
                checkpoint = destination / "best.pth"
                current = training_is_current(
                    finished, args, recipe, fold, seed, args.epochs)
                if not args.force and current and checkpoint.is_file():
                    print(f"skip: {destination} (finished)")
                    continue
                if finished.is_file() and not args.force:
                    print(f"rerun: {destination} (configuration or manifest changed)")
                run(command, dry_run)


def predict_oof(args, dry_run=False) -> None:
    for recipe in args.recipes:
        for seed in args.seeds:
            for fold in args.folds:
                command, destination = oof_command(args, recipe, fold, seed)
                checkpoint = fold_run(args.work_dir, recipe, fold, seed) / "best.pth"
                if not checkpoint.is_file() and not dry_run:
                    raise FileNotFoundError(f"fold checkpoint missing: {checkpoint}")
                scores_path = destination / "scores.json"
                current = False
                if scores_path.is_file():
                    try:
                        saved = json.loads(scores_path.read_text())
                        expected = {"tile": args.tile, "overlap": args.overlap,
                                    "flip_tta": True, "d4": True,
                                    "scale_tta": args.scale_tta, "si_vote": True,
                                    "manifest": str(args.work_dir / "manifests" /
                                                    f"fold_{fold}_valid.json"),
                                    "checkpoints": [str(checkpoint)]}
                        probabilities = list(destination.glob("*_prob.npz"))
                        current = (all(saved.get(key) == value for key, value in expected.items())
                                   and len(probabilities) == len(saved.get("per_image", []))
                                   and scores_path.stat().st_mtime >= checkpoint.stat().st_mtime)
                    except (json.JSONDecodeError, OSError):
                        current = False
                if not args.force and current:
                    print(f"skip: {destination} (OOF predictions finished)")
                    continue
                if scores_path.is_file() and not args.force:
                    print(f"rerun: {destination} (inference configuration changed)")
                run(command, dry_run)


def audit(args, dry_run=False) -> None:
    destination = args.work_dir / "label_audit"
    command = [args.python, str(ROOT / "audit_oof_labels.py"),
               "--oof_dirs", str(args.work_dir / "oof"), "--out", str(destination)]
    audit_path = destination / "audit.json"
    probabilities = list((args.work_dir / "oof").rglob("*_prob.npz"))
    current = (audit_path.is_file() and probabilities
               and audit_path.stat().st_mtime >= max(path.stat().st_mtime
                                                      for path in probabilities))
    if not args.force and current:
        print(f"skip: {destination} (audit finished)")
        return
    run(command, dry_run)


def train_final(args, dry_run=False) -> None:
    selections = {}
    for recipe in args.recipes:
        epochs, observed = selected_final_epoch(args, recipe) if not dry_run or \
            args.final_epochs is not None else (args.epochs, [])
        selections[recipe] = {"selected_epoch": epochs, "fold_best_epochs": observed}
        source = f"median fold peak from {observed}" if observed else \
            ("--final_epochs" if args.final_epochs is not None else "plan fallback")
        print(f"final epoch [{recipe}]: {epochs} ({source})")
        for seed in args.seeds:
            command, destination = final_command(args, recipe, seed, epochs)
            current = training_is_current(
                destination / "history.json", args, recipe, None, seed, epochs)
            if not args.force and current and (destination / "final.pth").is_file():
                print(f"skip: {destination} (final fit finished)")
                continue
            if (destination / "history.json").is_file() and not args.force:
                print(f"rerun: {destination} (configuration or manifest changed)")
            run(command, dry_run)
    if not dry_run:
        selection_path = args.work_dir / "final" / "epoch_selection.json"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(selections, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="plan",
                        choices=["plan", "prepare", "train", "oof", "audit", "final", "all"])
    parser.add_argument("--data_root", type=Path, default=ROOT)
    parser.add_argument("--work_dir", type=Path, default=ROOT / "runs" / "group_cv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--recipes", nargs="+", choices=sorted(RECIPES), default=["b3_aug3"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--folds", nargs="+", type=int, default=None,
                        help="fold subset to run; default is every fold")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--split_seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_every", type=int, default=6)
    parser.add_argument("--final_epochs", type=int, default=None,
                        help="fixed all-label budget; default is median fold-best epoch")
    parser.add_argument("--tile", type=int, default=768)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--scale_tta", nargs="+", type=float, default=[1.0])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.data_root = args.data_root.resolve()
    args.work_dir = args.work_dir.resolve()
    args.folds = list(range(args.n_splits)) if args.folds is None else args.folds
    if args.n_splits < 2:
        parser.error("--n_splits must be at least 2")
    if len(set(args.folds)) != len(args.folds) or any(
            fold < 0 or fold >= args.n_splits for fold in args.folds):
        parser.error("--folds must be distinct values in [0, n_splits)")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if args.epochs <= 0 or args.val_every <= 0:
        parser.error("--epochs and --val_every must be positive")
    if args.final_epochs is not None and args.final_epochs <= 0:
        parser.error("--final_epochs must be positive")
    if args.tile <= 0 or not 0 <= args.overlap < 1:
        parser.error("--tile must be positive and --overlap must be in [0, 1)")
    if not args.scale_tta or any(value <= 0 for value in args.scale_tta):
        parser.error("--scale_tta values must be positive")

    prepare(args)
    if args.stage == "prepare":
        return
    dry_run = args.stage == "plan"
    if args.stage in ("plan", "train", "all"):
        train_folds(args, dry_run)
    if args.stage in ("plan", "oof", "all"):
        predict_oof(args, dry_run)
    if args.stage in ("plan", "audit", "all"):
        audit(args, dry_run)
    if args.stage in ("plan", "final", "all"):
        train_final(args, dry_run)


if __name__ == "__main__":
    main()
