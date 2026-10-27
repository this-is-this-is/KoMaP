"""Run one command after a specific process exits and the GPU becomes idle.

The Linux PID plus its start ticks identify the process without a PID-reuse
race.  The queue itself does not import torch or create a CUDA context.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def process_running(pid, start_ticks):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[19] == str(start_ticks) and fields[0] not in ("Z", "X")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def gpu_compute_pids():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    return sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait_pid", type=int, required=True)
    ap.add_argument("--wait_start_ticks", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="queued command")
    ap.add_argument("--poll", type=float, default=15)
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="command after a -- separator")
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.wait_pid <= 0 or args.poll <= 0:
        ap.error("wait_pid and poll must be positive")
    if not command:
        ap.error("a command is required after --")
    if not args.workdir.is_dir():
        ap.error(f"workdir does not exist: {args.workdir}")

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queue.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"queue already active: {args.out}")

        state = dict(queue_pid=os.getpid(), wait_pid=args.wait_pid,
                     wait_start_ticks=str(args.wait_start_ticks),
                     workdir=str(args.workdir.resolve()), command=command,
                     label=args.label, queued_at=datetime.now().astimezone().isoformat())
        previous = None

        def report(phase, detail=None, **extra):
            nonlocal previous
            state.update(phase=phase, detail=detail,
                         updated_at=datetime.now().astimezone().isoformat(), **extra)
            temporary = args.out / "status.json.tmp"
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(args.out / "status.json")
            if (phase, detail) != previous:
                print(f"[{state['updated_at']}] {phase}: {detail or ''}", flush=True)
                previous = phase, detail

        while process_running(args.wait_pid, args.wait_start_ticks):
            report("waiting_for_process", f"PID {args.wait_pid}")
            time.sleep(args.poll)
        while True:
            try:
                pids = gpu_compute_pids()
                if not pids:
                    break
                report("waiting_for_gpu", ", ".join(pids))
            except (OSError, subprocess.SubprocessError) as error:
                report("waiting_for_gpu", f"GPU status unavailable: {error}")
            time.sleep(args.poll)

        report("starting", args.label)
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1", NO_ALBUMENTATIONS_UPDATE="1")
            child = subprocess.Popen(command, cwd=args.workdir, stdin=subprocess.DEVNULL, env=env)
            report("running", args.label, child_pid=child.pid,
                   started_at=datetime.now().astimezone().isoformat())
            code = child.wait()
            report("completed" if code == 0 else "failed", f"exit code {code}",
                   exit_code=code, finished_at=datetime.now().astimezone().isoformat())
            return code
        except Exception as error:
            report("failed", str(error))
            raise


if __name__ == "__main__":
    sys.exit(main())
