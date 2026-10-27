#!/usr/bin/env bash
# Queue: wait for whatever is training now, then the epoch-length experiment,
# then the new-augmentation suite. Everything is logged to this script's stdout.
#
#   nohup ./run_queue.sh > suite_queue.log 2>&1 &
#   ./run_queue.sh --check        # only report whether the GPU counts as busy
#
# "Busy" means a train_v2.py / predict.py / eval_ensembles.py process exists or
# more than 2 GiB of GPU memory is in use. Scale-2 runs fill 90 of 98 GB, so
# nothing here may start next to them.
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python

busy() {
  pgrep -f "[t]rain_v2.py|[p]redict.py|[e]val_ensembles.py" > /dev/null && return 0
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( used > 2048 ))
}

wait_idle() {
  # Two idle readings a minute apart, so the gap between two runs of another
  # suite is not mistaken for the end of it.
  local idle=0
  while (( idle < 2 )); do
    if busy; then idle=0; else idle=$((idle + 1)); fi
    (( idle < 2 )) && sleep 60
  done
}

if [[ "${1:-}" == "--check" ]]; then
  if busy; then echo "busy"; else echo "idle"; fi
  exit 0
fi

echo "[$(date '+%F %T')] queue started; waiting for the GPU to be idle"
wait_idle

# 1) Epoch length, with per-image scores logged so split_half_check.py can
#    separate a real gain from picking the best epoch on the validation scans.
#    val_every keeps about 30 evaluations per run at every length.
echo "[$(date '+%F %T')] epoch experiment: 60 / 120 / 200 epochs x 3 seeds"
for cfg in "60 2" "120 4" "200 6"; do
  set -- $cfg
  for s in 0 1 2; do
    out=runs/c_ep$1_s$s
    if [[ -f "$out/history.json" ]]; then echo "skip $out (finished)"; continue; fi
    $PY train_v2.py --arch unetpp --encoder efficientnet-b3 --pretrain micronet \
        --fullres_skip --color_aug oneof --epochs "$1" --val_every "$2" --seed "$s" \
        --out "$out" > "logs_c_ep$1_s$s.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] $1 epochs done: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_c_ep$1_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
done
$PY split_half_check.py c_ep60 c_ep120 c_ep200 || echo "split_half_check failed"

# 2) New augmentations and per-scan standardisation, each alone on top of
#    fullres + aug_color: 6 methods x 3 seeds, then the 3-seed ensembles.
wait_idle
METHODS="illumination blur fourier tone scale_small prisi_paste"
echo "[$(date '+%F %T')] augmentation suite: $METHODS x 3 seeds"
JOBS=3 ./run_augmentation_suite.sh $METHODS || echo "augmentation suite reported a failure"
echo "[$(date '+%F %T')] queue finished"
