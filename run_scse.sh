#!/usr/bin/env bash
# Round 9: scSE decoder attention (--decoder_attention scse), 60 epochs, 3
# seeds, then the selection-free check and the final-inference comparison
# with the same recipe without attention.
#
#   ./run_scse.sh scale1   screening run at native scale on the aug_color
#                          recipe: runs/a_scse_s* against runs/c_ep60_s*.
#                          About 9 GiB per run, three at a time, ~35 minutes.
#   ./run_scse.sh          the real thing on cs2_aug3 (scale 2):
#                          runs/b_cs2_aug3_scse_s* against runs/b_cs2_aug3_s*.
#                          One run peaks at 34 GiB allocated (26 without scSE),
#                          so only two fit in 98 GB at a time, ~2.5 hours.
#
# JOBS overrides how many train at once. Finished runs are skipped, so this
# can be restarted after an interruption.
#   nohup ./run_scse.sh scale1 > suite_scse_scale1.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
COMMON="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof"
case "${1:-scale2}" in
  scale1)  # same flags as c_ep60 / b_aug_color, plus the attention
    NAME=a_scse; REF=c_ep60; JOBS=${JOBS:-3}
    FLAGS="$COMMON --decoder_attention scse" ;;
  scale2)  # same flags as b_cs2_aug3 in run_boundary_suite.sh, plus the attention
    NAME=b_cs2_aug3_scse; REF=b_cs2_aug3; JOBS=${JOBS:-2}
    FLAGS="$COMMON --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2 --decoder_attention scse" ;;
  *) echo "usage: $0 [scale1|scale2]"; exit 1 ;;
esac
echo "[$(date '+%F %T')] start $NAME x 3 seeds, $JOBS at a time (reference: $REF)"
for s in 0 1 2; do
  out=runs/${NAME}_s$s
  if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
  $PY train_v2.py $FLAGS --seed $s --out "$out" > "logs_${NAME}_s$s.log" 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
done
wait
echo "[$(date '+%F %T')] trained: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_${NAME}_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
echo "[$(date '+%F %T')] split-half check (epoch picked on half the scans, scored on the other half)"
$PY split_half_check.py $REF $NAME
echo "[$(date '+%F %T')] final-inference comparison (tile 768)"
$PY compare_groups.py $REF $NAME
# The channel half of scSE pools over the whole tile; 512 is the area it was trained on.
echo "[$(date '+%F %T')] same comparison at the training tile (512)"
$PY compare_groups.py $REF $NAME --tile 512
echo "[$(date '+%F %T')] all finished"
