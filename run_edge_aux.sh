#!/usr/bin/env bash
# Round 8: team experiment E3 (edge auxiliary head) on our cs2_aug3 recipe,
# 60 epochs, 3 seeds in parallel (three scale-2 runs fill 90 of 98 GB), then
# the selection-free check and the final-inference comparison with cs2_aug3.
# --edge_w 0.05 next to our 0.5 CE + 0.5 Dice is E3's 0.1 next to CE + Dice.
# Finished runs are skipped, so this can be restarted after an interruption.
#   nohup ./run_edge_aux.sh > suite_edge_aux.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
# Same flags as b_cs2_aug3 in run_boundary_suite.sh, plus the edge head.
FLAGS="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2 --edge_w 0.05"
echo "[$(date '+%F %T')] start b_cs2_aug3_edge x 3 seeds"
for s in 0 1 2; do
  out=runs/b_cs2_aug3_edge_s$s
  if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
  $PY train_v2.py $FLAGS --seed $s --out "$out" > "logs_b_cs2_aug3_edge_s$s.log" 2>&1 &
done
wait
echo "[$(date '+%F %T')] trained: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_b_cs2_aug3_edge_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
echo "[$(date '+%F %T')] split-half check (epoch picked on half the scans, scored on the other half)"
$PY split_half_check.py b_cs2_aug3 b_cs2_aug3_edge
echo "[$(date '+%F %T')] final-inference comparison"
$PY compare_groups.py b_cs2_aug3 b_cs2_aug3_edge
echo "[$(date '+%F %T')] all finished"
