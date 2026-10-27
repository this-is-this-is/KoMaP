"""Wait for an existing experiment suite, then run the complete D2 screen.

Run detached with stdin closed and stdout/stderr redirected to a log. The
target PID and Linux start ticks identify the original process, even if the
PID is later reused. No CUDA context is created while waiting.
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
    except FileNotFoundError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wait_pid", type=int, required=True)
    ap.add_argument("--wait_start_ticks", required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--poll", type=float, default=10)
    args = ap.parse_args()
    if args.wait_pid <= 0 or args.poll <= 0:
        ap.error("wait_pid and poll must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "queue.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("this D2 queue is already active")
        command = [sys.executable, "run_mask_diffusion.py", "--stage", "all"]
        state = dict(queue_pid=os.getpid(), wait_pid=args.wait_pid,
                     wait_start_ticks=args.wait_start_ticks, workdir=str(args.workdir.resolve()),
                     command=command, queued_at=datetime.now().astimezone().isoformat())
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

        while True:
            if process_running(args.wait_pid, args.wait_start_ticks):
                report("waiting_for_suite", f"PID {args.wait_pid}")
            else:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True, timeout=10)
                    pids = sorted(set(result.stdout.split()))
                    if not pids:
                        break
                    report("waiting_for_gpu", ", ".join(pids))
                except (OSError, subprocess.SubprocessError) as error:
                    report("waiting_for_gpu", f"GPU status unavailable: {error}")
            time.sleep(args.poll)
        report("starting", "generator -> refit -> sampling -> 10%/30% screens")
        try:
            env = dict(os.environ, PYTHONUNBUFFERED="1", NO_ALBUMENTATIONS_UPDATE="1")
            child = subprocess.Popen(command, cwd=args.workdir, stdin=subprocess.DEVNULL, env=env)
            report("running", "D2 pipeline", child_pid=child.pid,
                   started_at=datetime.now().astimezone().isoformat())
            code = child.wait()
            report("completed" if code == 0 else "failed", f"exit code {code}", exit_code=code,
                   finished_at=datetime.now().astimezone().isoformat())
            return code
        except Exception as error:
            report("failed", str(error))
            raise


if __name__ == "__main__":
    sys.exit(main())
