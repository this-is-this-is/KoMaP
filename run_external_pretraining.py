"""Prepare, pretrain, screen and conditionally promote microscopy-transfer models."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".conda/bin/python"
SUITE = ROOT / "runs/external_pretraining_suite"
LOGS = SUITE / "logs"
BASELINE6 = [ROOT / "runs" / f"{recipe}_s{seed}" / "best.pth"
             for recipe in ("b_cs2_aug3_ep200", "b_cs2_plus_ep200") for seed in range(3)]


def method_initialisation(method):
    if method == "matssl":
        return ["--encoder", "resnet50", "--pretrain", "matssl"]
    if method == "cross":
        return ["--encoder", "efficientnet-b3", "--pretrain", "none",
                "--encoder_checkpoint", str(SUITE / "cross_ssl/best_encoder.pth")]
    if method == "nasa":
        return ["--encoder", "efficientnet-b3", "--pretrain", "none",
                "--init_checkpoint", str(SUITE / "nasa_super/best.pth")]
    raise ValueError(method)


def downstream_command(method, seed, out, recipe="cs2_aug3", epochs=60):
    command = [str(PYTHON), "train_v2.py", "--data_root", str(ROOT), "--arch", "unetpp",
               *method_initialisation(method), "--fullres_skip", "--color_aug", "oneof",
               "--input_scale", "2", "--prisi_paste_p", "0.25",
               "--illumination_p", "0.2", "--fourier_p", "0.1",
               "--epochs", str(epochs), "--val_every", "6" if epochs >= 200 else "2",
               "--seed", str(seed), "--out", str(out)]
    if recipe == "cs2_plus":
        command += ["--scale_by_height", "800:4"]
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["prepare", "pretrain", "screen", "evaluate", "promote", "all", "plan"],
                    default="all")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force_promote", action="store_true")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()
    if args.jobs <= 0:
        ap.error("--jobs must be positive")
    dry = args.stage == "plan"
    if not dry and not args.allow_cpu and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --allow_cpu only for a smoke test")
    SUITE.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    commands = []

    def run(command, marker, log):
        command, marker, log = list(map(str, command)), Path(marker), Path(log)
        commands.append(command)
        if marker.exists():
            print(f"skip {marker} (finished)")
            return
        print("RUN " + " ".join(command), flush=True)
        if dry:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, PYTHONUNBUFFERED="1",
                                             NO_ALBUMENTATIONS_UPDATE="1"))
        if result.returncode:
            raise SystemExit(f"command failed ({result.returncode}); see {log}")
        if not marker.exists():
            if marker.suffix != ".done":
                raise SystemExit(f"command succeeded but did not create {marker}; see {log}")
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("completed\n")

    def run_many(jobs):
        queue, active = deque(jobs), []
        while queue or active:
            while queue and len(active) < args.jobs:
                command, marker, log = queue.popleft()
                command, marker, log = list(map(str, command)), Path(marker), Path(log)
                commands.append(command)
                if marker.exists():
                    print(f"skip {marker} (finished)")
                    continue
                print("START " + " ".join(command), flush=True)
                if dry:
                    continue
                log.parent.mkdir(parents=True, exist_ok=True)
                handle = log.open("w")
                process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                           env=dict(os.environ, PYTHONUNBUFFERED="1",
                                                    NO_ALBUMENTATIONS_UPDATE="1"))
                active.append((process, handle, marker, log))
            if dry:
                continue
            process, handle, marker, log = active.pop(0)
            code = process.wait()
            handle.close()
            if code:
                for other, other_handle, _, _ in active:
                    other.terminate()
                    other_handle.close()
                raise SystemExit(f"command failed ({code}); see {log}")
            if not marker.exists():
                raise SystemExit(f"command succeeded but did not create {marker}; see {log}")
            print(f"DONE {marker}", flush=True)

    do_prepare = args.stage in ("prepare", "all", "plan")
    do_pretrain = args.stage in ("pretrain", "all", "plan")
    do_screen = args.stage in ("screen", "all", "plan")
    do_evaluate = args.stage in ("evaluate", "all", "plan")
    do_promote = args.stage in ("promote", "all", "plan")

    if do_prepare:
        run([str(PYTHON), "prepare_external_microscopy.py", "--out", str(ROOT / "external_data")],
            ROOT / "external_data/READY.json", LOGS / "prepare.log")

    if do_pretrain:
        run([str(PYTHON), "pretrain_cross_microscope.py",
             "--manifest", str(ROOT / "external_data/cross_microscope/manifest.jsonl"),
             "--data_root", str(ROOT), "--out", str(SUITE / "cross_ssl")],
            SUITE / "cross_ssl/history.json", LOGS / "cross_ssl.log")
        run([str(PYTHON), "pretrain_nasa_super.py",
             "--nasa_root", str(ROOT / "external_data/nasa_pretrained_microscopy_models/source"),
             "--out", str(SUITE / "nasa_super"), "--mode", "all"],
            SUITE / "nasa_super/history.json", LOGS / "nasa_super.log")

    methods = ("matssl", "cross", "nasa")
    if do_screen:
        jobs = []
        for method in methods:
            for seed in args.seeds:
                out = ROOT / "runs" / f"xfer_{method}_s{seed}"
                jobs.append((downstream_command(method, seed, out), out / "history.json",
                             LOGS / f"xfer_{method}_s{seed}.log"))
        run_many(jobs)

    screen_result = SUITE / "screen_ensemble.json"
    if do_evaluate:
        command = [str(PYTHON), "evaluate_external_transfer.py", "--baseline",
                   *map(str, BASELINE6),
                   "--candidate", "micronet_screen=runs/b_cs2_aug3_s[012]/best.pth"]
        command += [item for method in methods
                    for item in ("--candidate", f"{method}=runs/xfer_{method}_s[012]/best.pth")]
        command += ["--out", str(screen_result)]
        run(command, screen_result, LOGS / "screen_ensemble.log")
        split_marker = SUITE / "split_half.done"
        run([str(PYTHON), "split_half_check.py", "b_cs2_aug3",
             *[f"xfer_{method}" for method in methods]], split_marker, LOGS / "split_half.log")

    if do_promote:
        decision_path = SUITE / "promotion_decision.json"
        if dry:
            print("PLAN conditional 200-epoch promotion of the best screen candidate")
            winner, promote = "matssl", True  # representative command; winner is selected at runtime
        else:
            if not screen_result.is_file():
                raise FileNotFoundError(f"run --stage evaluate first: {screen_result}")
            result = json.loads(screen_result.read_text())
            eligible = {name: value for name, value in result["candidates"].items()
                        if name in methods}
            winner = max(eligible, key=lambda key: eligible[key]["delta_equal_members_pp"])
            reference = result["candidates"]["micronet_screen"]["ensemble"]["miou"]
            own_gap = eligible[winner]["ensemble"]["miou"] - reference
            addition = eligible[winner]["delta_equal_members_pp"]
            # A candidate advances if it helps the incumbent at a fixed weight,
            # or is close enough in the 60-epoch screen that the longer run can
            # plausibly recover the small gap.  This prevents a clearly harmful
            # transfer source from consuming six 200-epoch runs.
            promote = args.force_promote or addition >= 0.02 or own_gap >= -0.15
            decision = {"winner": winner, "promote": promote,
                        "screen_ensemble_vs_micronet_pp": own_gap,
                        "baseline6_addition_pp": addition,
                        "rule": "addition >= +0.02pp or screen ensemble within 0.15pp of MicroNet"}
            decision_path.write_text(json.dumps(decision, indent=2))
            print(json.dumps(decision, indent=2), flush=True)
        if promote:
            jobs = []
            for recipe in ("cs2_aug3", "cs2_plus"):
                for seed in args.seeds:
                    out = ROOT / "runs" / f"xfer_final_{winner}_{recipe}_s{seed}"
                    jobs.append((downstream_command(winner, seed, out, recipe, 200),
                                 out / "history.json", LOGS / f"{out.name}.log"))
            run_many(jobs)
            if not dry:
                final_result = SUITE / f"final_{winner}_ensemble.json"
                run([str(PYTHON), "evaluate_external_transfer.py", "--baseline", *map(str, BASELINE6),
                     "--candidate", f"{winner}_final=runs/xfer_final_{winner}_*_s[012]/best.pth",
                     "--out", str(final_result)], final_result, LOGS / f"final_{winner}_ensemble.log")

    (SUITE / f"commands_{args.stage}.json").write_text(json.dumps(commands, indent=2))


if __name__ == "__main__":
    main()
