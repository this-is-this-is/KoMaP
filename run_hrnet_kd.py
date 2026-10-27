"""Offline distillation from the best heterogeneous 12-model ensemble to HRNet-W18.

The expensive teacher is evaluated once on each clean training scan. Training
then crops and geometrically augments those cached soft probabilities together
with the image/mask, so no teacher model occupies GPU memory during student
updates.

Examples::

    # Show cache/training commands without running them.
    ./.conda/bin/python run_hrnet_kd.py --stage plan

    # One-seed screen, then the confirmatory three-seed run.
    ./.conda/bin/python run_hrnet_kd.py --stage all --seeds 0
    ./.conda/bin/python run_hrnet_kd.py --stage all --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
DEFAULT_TEACHERS = [
    *(ROOT / "runs" / f"b_cs2_aug3_ep200_s{seed}" / "best.pth" for seed in range(3)),
    *(ROOT / "runs" / f"b_cs2_plus_ep200_s{seed}" / "best.pth" for seed in range(3)),
    *(ROOT / "runs" / f"b_seresnext50_cs2_aug3_s{seed}" / "best.pth" for seed in range(3)),
    *(ROOT / "runs" / f"b_seresnext50_cs2_aug3_ep200_s{seed}" / "best.pth"
      for seed in range(3)),
]


def show(command):
    print(shlex.join(str(value) for value in command), flush=True)


def execute(command, dry_run=False):
    print(("plan: " if dry_run else "run:  "), end="")
    show(command)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def cache_is_current(directory: Path, checkpoints: list[Path]) -> bool:
    scores = directory / "scores.json"
    probabilities = list(directory.glob("*_prob.npz"))
    if not scores.is_file() or len(probabilities) != 70:
        return False
    try:
        value = json.loads(scores.read_text())
    except json.JSONDecodeError:
        return False
    expected = {
        "tile": 768, "overlap": 0.5, "flip_tta": True, "d4": True,
        "scale_tta": [1.0], "split": "train", "manifest": None,
        "checkpoints": [str(path) for path in checkpoints],
    }
    return (all(value.get(key) == item for key, item in expected.items())
            and scores.stat().st_mtime >= max(path.stat().st_mtime for path in checkpoints))


def training_is_current(history: Path, args, seed: int) -> bool:
    if not history.is_file():
        return False
    try:
        saved = json.loads(history.read_text())["args"]
    except (KeyError, json.JSONDecodeError):
        return False
    expected = {
        "arch": "hrnet", "encoder": "hrnet_w18", "pretrain": "imagenet",
        "epochs": args.epochs, "seed": seed, "teacher_prob_dir": str(args.teacher_dir),
        "kd_weight": args.kd_weight, "kd_boundary_w": args.kd_boundary_w,
        "kd_temperature": args.kd_temperature, "kd_confidence": args.kd_confidence,
    }
    return all(saved.get(key) == value for key, value in expected.items())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="plan",
                        choices=["plan", "cache", "train", "evaluate", "all"])
    parser.add_argument("--teachers", nargs="+", type=Path, default=DEFAULT_TEACHERS)
    parser.add_argument("--teacher_dir", type=Path,
                        default=ROOT / "runs" / "kd_teacher_mixed12_train")
    parser.add_argument("--out_group", default="b_hrnet18_kd_mixed12")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--val_every", type=int, default=2)
    parser.add_argument("--kd_weight", type=float, default=0.3)
    parser.add_argument("--kd_boundary_w", type=float, default=0.1)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--kd_confidence", type=float, default=0.5)
    parser.add_argument("--force_cache", action="store_true")
    parser.add_argument("--force_train", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    args.teachers = [path.resolve() for path in args.teachers]
    args.teacher_dir = args.teacher_dir.resolve()
    missing = [path for path in args.teachers if not path.is_file()]
    if missing:
        parser.error("teacher checkpoints missing: " + ", ".join(map(str, missing)))
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be distinct")
    if args.epochs <= 0 or args.val_every <= 0:
        parser.error("--epochs and --val_every must be positive")
    if args.kd_weight < 0 or args.kd_boundary_w < 0 or \
            args.kd_weight + args.kd_boundary_w <= 0:
        parser.error("KD weights must be non-negative and at least one must be positive")
    if args.kd_temperature <= 0 or not 0 <= args.kd_confidence < 1:
        parser.error("temperature must be positive and confidence must be in [0, 1)")

    dry_run = args.stage == "plan"
    cache_command = [args.python, str(ROOT / "predict.py"), "--ckpt",
                     *map(str, args.teachers), "--data_root", str(ROOT),
                     "--split", "train", "--tile", "768", "--overlap", "0.5",
                     "--tta", "--d4", "--save_dir", str(args.teacher_dir), "--save_prob"]
    if args.stage in ("plan", "cache", "all"):
        if not args.force_cache and cache_is_current(args.teacher_dir, args.teachers):
            print(f"skip: {args.teacher_dir} (70 current teacher probability maps)")
        else:
            execute(cache_command, dry_run)

    if args.stage in ("plan", "train", "all"):
        if not dry_run and not cache_is_current(args.teacher_dir, args.teachers):
            raise RuntimeError("teacher cache is absent/stale; run --stage cache or --stage all")
        for seed in args.seeds:
            destination = ROOT / "runs" / f"{args.out_group}_s{seed}"
            command = [args.python, str(ROOT / "train_v2.py"),
                       "--data_root", str(ROOT),
                       "--arch", "hrnet", "--encoder", "hrnet_w18",
                       "--pretrain", "imagenet", "--color_aug", "oneof",
                       "--input_scale", "2", "--prisi_paste_p", "0.25",
                       "--fourier_p", "0.1", "--illumination_p", "0.2",
                       "--teacher_prob_dir", str(args.teacher_dir),
                       "--kd_weight", str(args.kd_weight),
                       "--kd_boundary_w", str(args.kd_boundary_w),
                       "--kd_temperature", str(args.kd_temperature),
                       "--kd_confidence", str(args.kd_confidence),
                       "--epochs", str(args.epochs), "--val_every", str(args.val_every),
                       "--seed", str(seed), "--out", str(destination)]
            if not args.force_train and training_is_current(
                    destination / "history.json", args, seed) and \
                    (destination / "best.pth").is_file():
                print(f"skip: {destination} (finished)")
            else:
                execute(command, dry_run)

    if args.stage in ("plan", "evaluate", "all"):
        if set(args.seeds) != {0, 1, 2}:
            print("skip evaluation: use --seeds 0 1 2 for the matched three-seed comparison")
        else:
            for seed in args.seeds:
                checkpoint = ROOT / "runs" / f"{args.out_group}_s{seed}" / "best.pth"
                if not checkpoint.is_file() and not dry_run:
                    raise FileNotFoundError(checkpoint)
            execute([args.python, str(ROOT / "split_half_check.py"),
                     "b_hrnet18_cs2_aug3", args.out_group], dry_run)
            execute([args.python, str(ROOT / "compare_groups.py"),
                     "b_hrnet18_cs2_aug3", args.out_group, "--seeds", "0", "1", "2"],
                    dry_run)


if __name__ == "__main__":
    main()
