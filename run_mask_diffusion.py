"""Reproducible stages for mask-conditional diffusion augmentation (D2).

Default: print the plan. --stage all trains/selects/refits a generator, samples
data + matched real repeats, and runs 60-epoch segmentation screens. Promotion
to the existing 200-epoch six-model ensemble is an explicit --stage final.
"""
import argparse
import fcntl
import json
import re
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = ROOT / ".conda/bin/python"


def clone_training(reference, out, synth_dir, ratio, seed):
    """Retain the reference recipe, including scale table, LR and train length."""
    args = json.loads((ROOT / "runs" / f"{reference}_s{seed}" / "history.json").read_text())["args"]
    args.update(seed=seed, out=str(out), synth_dir=str(synth_dir), synth_ratio=ratio)
    cmd = [str(PY), "train_v2.py"]
    for key, value in args.items():
        if key == "tone_target" or value is None or value is False:
            continue
        cmd.append("--" + key)
        if value is not True:
            cmd.extend(map(str, value if isinstance(value, list) else [value]))
    return cmd


def validate_export(directory, count):
    meta = json.loads((directory / "meta.json").read_text())
    if meta["generator_mode"] != "refit" or meta["size"] != 512:
        raise ValueError("downstream studies require all-train refit and native 512px crops")
    if meta["count"] != count:
        raise ValueError("export count changed; use a new --name for a different recipe")
    rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != count or len({r["name"] for r in rows}) != count:
        raise ValueError(f"incomplete manifest: {directory}")
    for row in rows:
        if not (directory / "images" / row["name"]).is_file() or not (
                directory / "masks" / row["name"].replace("_image", "_mask")).is_file():
            raise ValueError(f"incomplete pair: {directory}/{row['name']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["plan", "generator", "sample", "screen", "final", "all"], default="plan")
    ap.add_argument("--name", default="md2")
    ap.add_argument("--variant", choices=["full", "mask_only", "no_geometry", "no_style", "unweighted"], default="full")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--hours", type=float, default=3.0, help="selection-run wall-time budget")
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--modes", nargs="+", choices=["appearance", "remodel"], default=["appearance"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_]+", args.name):
        ap.error("name must use only letters, digits and underscores")
    if min(args.steps, args.count) < 1 or args.hours <= 0 or len(set(args.seeds)) != len(args.seeds):
        ap.error("steps/count/hours must be positive and seeds must be unique")
    if any(r not in (0.1, 0.2, 0.3) for r in args.ratios):
        ap.error("ratios must be 0.1, 0.2 or 0.3")
    if args.stage == "final" and (len(args.ratios) != 1 or len(args.modes) != 1 or args.seeds != [0, 1, 2]):
        ap.error("final requires one selected --ratios value, one --modes value and seeds 0 1 2")
    dry = args.dry_run or args.stage == "plan"
    name = args.name + ("_" + args.variant if args.variant != "full" else "")
    root = ROOT / "runs" / name
    select, refit = root / "select", root / "refit"
    extras = {"full": [], "mask_only": ["--no-geometry", "--no-style", "--boundary_weight", "0", "--thin_weight", "0"],
              "no_geometry": ["--no-geometry"], "no_style": ["--no-style"],
              "unweighted": ["--boundary_weight", "0", "--thin_weight", "0"]}[args.variant]
    commands = []

    def run(cmd, marker=None, log=None):
        commands.append(cmd)
        print(shlex.join(cmd), flush=True)
        if dry:
            return
        if marker and Path(marker).exists():
            print(f"skip completed {marker}", flush=True)
            return
        if log:
            log = Path(log)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as stream:
                subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        else:
            subprocess.run(cmd, cwd=ROOT, check=True)

    if not dry:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / "pipeline.lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another {name} pipeline is running")
        (root / f"protocol_{args.stage}.json").write_text(json.dumps(vars(args), indent=2))
    if args.stage in ("generator", "all", "plan"):
        if not dry and (select / "history.json").exists():
            old = json.loads((select / "history.json").read_text())["args"]
            if old["steps"] != args.steps or old["hours"] != args.hours:
                raise SystemExit("generator budget changed; use a new --name")
        cmd = [str(PY), "train_mask_diffusion.py", "--out", str(select), "--steps", str(args.steps),
               "--hours", str(args.hours), *extras]
        if (select / "last.pt").exists():
            cmd.append("--resume")
        run(cmd, select / "complete.json", root / "select.log")
        cmd = [str(PY), "train_mask_diffusion.py", "--out", str(refit), "--mode", "refit",
               "--selection", str(select / "best.pt")]
        if (refit / "last.pt").exists():
            cmd.append("--resume")
        run(cmd, refit / "complete.json", root / "refit.log")
    if args.stage in ("sample", "all", "plan"):
        for mode in args.modes:
            synth = ROOT / "synth" / f"{name}_{mode}"
            control = ROOT / "synth" / f"{name}_control_{mode}"
            if not dry and not (refit / "complete.json").exists():
                raise SystemExit("finish the refit stage before exporting downstream data")
            run([str(PY), "sample_mask_diffusion.py", "--ckpt", str(refit / "best.pt"),
                 "--out", str(synth), "--control_out", str(control), "--mode", mode,
                 "--count", str(args.count)], synth / "meta.json", root / f"sample_{mode}.log")
            if not dry:
                validate_export(synth, args.count)
                validate_export(control, args.count)

    def compare(ref, new):
        suffix = "_seeds" + "_".join(map(str, args.seeds))
        path = ROOT / "runs" / f"compare_{ref}_vs_{new}{suffix}.json"
        run([str(PY), "compare_groups.py", ref, new, "--seeds", *map(str, args.seeds)],
            path, root / f"compare_{ref}_vs_{new}{suffix}.log")
        run([str(PY), "summarize_mask_diffusion.py", "--comparison", str(path),
             "--ref_group", ref, "--new_group", new, "--out", str(path.with_suffix(".paired.json"))],
            path.with_suffix(".paired.json"))

    if args.stage in ("screen", "all", "plan"):
        for mode in args.modes:
            for ratio in args.ratios:
                for kind in (f"control_{mode}", mode):
                    synth = ROOT / "synth" / f"{name}_{kind}"
                    group = f"{name}_{kind}_r{round(ratio * 100)}"
                    if not dry:
                        validate_export(synth, args.count)
                    for seed in args.seeds:
                        out = ROOT / "runs" / f"{group}_s{seed}"
                        run(clone_training("b_cs2_aug3", out, synth, ratio, seed),
                            out / "history.json", root / f"{group}_s{seed}.log")
                    compare("b_cs2_aug3", group)
                control_group = f"{name}_control_{mode}_r{round(ratio * 100)}"
                gen_group = f"{name}_{mode}_r{round(ratio * 100)}"
                compare(control_group, gen_group)
    if args.stage == "final":
        ratio, mode = args.ratios[0], args.modes[0]
        synth = ROOT / "synth" / f"{name}_{mode}"
        if not dry:
            validate_export(synth, args.count)
        baseline, generated = [], []
        for recipe in ("cs2_aug3", "cs2_plus"):
            reference = f"b_{recipe}_ep200"
            group = f"{name}_final_{recipe}_{mode}_r{round(ratio * 100)}"
            for seed in args.seeds:
                out = ROOT / "runs" / f"{group}_s{seed}"
                baseline.append(str(ROOT / "runs" / f"{reference}_s{seed}" / "best.pth"))
                generated.append(str(out / "best.pth"))
                run(clone_training(reference, out, synth, ratio, seed), out / "history.json",
                    root / f"{group}_s{seed}.log")
        for label, checkpoints in (("baseline6", baseline), ("generated6", generated)):
            out = root / f"final_{mode}_r{round(ratio * 100)}_{label}"
            run([str(PY), "predict.py", "--ckpt", *checkpoints,
                 "--split", "valid", "--tta", "--d4", "--tile", "768", "--si_vote",
                 "--save_dir", str(out)], out / "scores.json", root / f"final_{label}.log")
        prefix = root / f"final_{mode}_r{round(ratio * 100)}"
        run([str(PY), "summarize_mask_diffusion.py", "--ref_scores", f"{prefix}_baseline6/scores.json",
             "--new_scores", f"{prefix}_generated6/scores.json", "--out", f"{prefix}_paired.json"],
            f"{prefix}_paired.json")
    if not dry:
        (root / f"commands_{args.stage}.json").write_text(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
