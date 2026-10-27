#!/usr/bin/env bash
# Final candidates at 200 epochs: cs2_aug3, then cs2_plus, 3 seeds each.
# One group at a time -- three scale-2 runs fill 90 of 98 GB of GPU memory.
# Finished runs are skipped, so this can be restarted after an interruption.
#   nohup ./run_final_ep200.sh > suite_final_ep200.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
BASE="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2 --epochs 200 --val_every 6"
declare -A EXTRA=([cs2_aug3]="" [cs2_plus]="--scale_by_height 800:4")
for g in cs2_aug3 cs2_plus; do
  echo "[$(date '+%F %T')] start b_${g}_ep200 x 3 seeds"
  for s in 0 1 2; do
    out=runs/b_${g}_ep200_s$s
    if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
    $PY train_v2.py $BASE ${EXTRA[$g]} --seed $s --out "$out" > "logs_b_${g}_ep200_s$s.log" 2>&1 &
  done
  wait
  echo "[$(date '+%F %T')] done b_${g}_ep200: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_b_${g}_ep200_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
done
echo "[$(date '+%F %T')] all finished"
