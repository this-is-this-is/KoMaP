#!/usr/bin/env bash
# Long, restartable KoMaP GPU queue. WeatherCity was explicitly cancelled.
set -euo pipefail

KOMAP_ROOT=/home/user/Desktop/komap
KOMAP_LOG="$KOMAP_ROOT/logs_group_cv_all.log"

stamp() {
    echo "[$(date '+%F %T')] $*"
}

exec 9> "$KOMAP_ROOT/.run_group_cv_then_weather.lock"
if ! flock -n 9; then
    stamp "another group-CV queue already owns the lock"
    exit 0
fi
if pgrep -f '[p]ython .*run_group_cv.py --stage all|[p]ython .*train_v2.py .*runs/group_cv' >/dev/null; then
    stamp "a group-CV orchestrator or fold trainer is already running; this copy leaves"
    exit 0
fi

status=running
finish() {
    code=$?
    if [ "$code" -eq 0 ]; then
        stamp "queue completed successfully"
    else
        stamp "queue stopped during: $status (exit $code)"
    fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd "$KOMAP_ROOT"
stamp "checking CUDA"
./.conda/bin/python -c \
    'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0), flush=True)'

status="KoMaP grouped CV / OOF / audit / final fits"
stamp "starting $status"
stamp "detailed KoMaP output: $KOMAP_LOG"
./.conda/bin/python -u run_group_cv.py \
    --stage all \
    --recipes b3_aug3 b3_plus seresnext50_aug3 \
    --seeds 0 1 2 \
    --scale_tta 0.875 1 1.125 \
    >> "$KOMAP_LOG" 2>&1
stamp "KoMaP completed"
status=complete
