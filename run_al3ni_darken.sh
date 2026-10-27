#!/usr/bin/env bash
# Round 7: cs2_aug3 + Al3Ni darkening (--al3ni_darken_p 0.3), 60 epochs, 3 seeds
# in parallel (three scale-2 runs fill 90 of 98 GB), then the selection-free
# check against cs2_aug3 and the targeted error breakdown.
# Finished runs are skipped, so this can be restarted after an interruption.
#   nohup ./run_al3ni_darken.sh > suite_al3ni_darken.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
# Same flags as b_cs2_aug3 in run_boundary_suite.sh, plus the new one.
FLAGS="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2 --al3ni_darken_p 0.3"
echo "[$(date '+%F %T')] start b_cs2_aug3_dark x 3 seeds"
for s in 0 1 2; do
  out=runs/b_cs2_aug3_dark_s$s
  if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
  $PY train_v2.py $FLAGS --seed $s --out "$out" > "logs_b_cs2_aug3_dark_s$s.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] trained: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_b_cs2_aug3_dark_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
echo "[$(date '+%F %T')] split-half check (epoch picked on half the scans, scored on the other half)"
$PY split_half_check.py b_cs2_aug3 b_cs2_aug3_dark
echo "[$(date '+%F %T')] final-inference evaluation"
$PY eval_al3ni_darken.py
echo "[$(date '+%F %T')] all finished"
