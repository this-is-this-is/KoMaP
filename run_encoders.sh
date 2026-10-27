#!/usr/bin/env bash
# Round 14: three more MicroNet encoders in place of EfficientNet-B3, after SE-ResNeXt-50
# (round 13) became the first one to beat it:
#     se_resnext101_32x4d   its deeper sibling                           -> runs/b_seresnext101_cs2_aug3_s*
#     inceptionresnetv2     inception branches with residual shortcuts   -> runs/b_incresv2_cs2_aug3_s*
#     dpn92                 residual and dense paths side by side        -> runs/b_dpn92_cs2_aug3_s*
# The model is the same otherwise (U-Net++ with the full-resolution branch), on the cs2_aug3
# recipe at scale 2, 60 epochs, 3 seeds. After each encoder: the selection-free check, the
# final-inference comparison with b_cs2_aug3, and the test that decides -- the best ensemble
# so far (six B3 and three SE-ResNeXt-50, 80.30%) with and without the new models.
#
#   setsid nohup ./run_encoders.sh > suite_encoders.log 2>&1 < /dev/null &               all three, in this order
#   ENCODERS="dpn92" setsid nohup ./run_encoders.sh > suite_dpn92.log 2>&1 < /dev/null &   only some
#
# Round 15: none of the three earned a place, so SE-ResNeXt-50 itself gets the 200 epochs the
# six B3 models had (-> runs/b_seresnext50_cs2_aug3_ep200_s*), compared with B3 at 200 epochs
# and with its own 60-epoch runs:
#   ENCODERS=se_resnext50_32x4d SUFFIX=_ep200 EXTRA="--epochs 200 --val_every 6" \
#     REF="b_cs2_aug3_ep200 b_seresnext50_cs2_aug3" \
#     setsid nohup ./run_encoders.sh > suite_seresnext50_ep200.log 2>&1 < /dev/null &
#
# One run at a time: the GPU is already full with one, so two side by side each take twice
# as long (SE-ResNeXt-50: 110 min together, 56 min alone) and only cost memory.
# As in run_seresnext.sh: the code is copied to .snap/<name> at the first launch and run
# from there, and a run starts only when the GPU has its memory plus RESERVE_MB free.
# Finished runs are skipped; a second copy leaves at once.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
ENCODERS=${ENCODERS:-se_resnext101_32x4d inceptionresnetv2 dpn92}
SEEDS=${SEEDS:-0 1 2}
ALSO=${ALSO-b_seresnext50_cs2_aug3}   # groups already in the best ensemble, next to the six B3
SUFFIX=${SUFFIX:-}               # appended to the group names, e.g. _ep200
REF=${REF:-b_cs2_aug3}           # groups to compare with; the first is the reference of the selection-free check
P=${PREFIX:-}                    # prefix of the run names (a smoke test uses its own)
EXTRA=${EXTRA:-}                 # appended to every training command
RESERVE_MB=${RESERVE_MB:-12000}  # GPU memory left free for whoever else needs it
PY="$ROOT/.conda/bin/python"
declare -A TAG=([se_resnext50_32x4d]=seresnext50 [se_resnext101_32x4d]=seresnext101 [inceptionresnetv2]=incresv2 [dpn92]=dpn92)
# MiB one run holds at scale 2 (batch 8 of 1024-px crops), measured, plus 4 GiB.
declare -A NEED=([se_resnext50_32x4d]=46000 [se_resnext101_32x4d]=52000 [inceptionresnetv2]=43000 [dpn92]=52000)
stamp() { echo "[$(date '+%F %T')] $*"; }

for enc in $ENCODERS; do
  [ -n "${TAG[$enc]:-}" ] || { echo "unknown encoder '$enc'; this script knows: ${!TAG[*]}"; exit 1; }
done

exec 9> ".run_${P}encoders.lock"
if ! flock -n 9; then stamp "run_encoders.sh is already running; this copy leaves"; exit 0; fi

SNAP="$ROOT/.snap/${P}encoders"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp train_v2.py predict.py compare_groups.py split_half_check.py eval_with_best.py "$SNAP/"
  cp -r src "$SNAP/src"
  for d in train valid test runs synth weights; do ln -s "$ROOT/$d" "$SNAP/$d"; done
  stamp "code frozen in $SNAP"
fi
cd "$SNAP"

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_wait() {  # MiB needed; says once that it is waiting
  local told=0
  while [ "$(free_mb)" -lt $(( $1 + RESERVE_MB )) ]; do
    if [ $told = 0 ]; then stamp "waiting for $(( $1 + RESERVE_MB )) MiB of free GPU memory ($(free_mb) free now)"; told=1; fi
    sleep 20
  done
}

for enc in $ENCODERS; do
  GROUP=${P}b_${TAG[$enc]}_cs2_aug3${SUFFIX}
  # b_cs2_aug3 with the encoder swapped.
  FLAGS="--arch unetpp --encoder $enc --pretrain micronet --fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"
  stamp "train $GROUP ($enc), seeds $SEEDS, one at a time"
  for pass in 1 2; do   # a run that died gets one more try
    for s in $SEEDS; do
      out=runs/${GROUP}_s$s
      [ -f "$out/history.json" ] && continue
      gpu_wait "${NEED[$enc]}"
      stamp "start ${GROUP}_s$s"
      $PY train_v2.py $FLAGS $EXTRA --seed $s --data_root . --out "$out" > "$ROOT/logs_${GROUP}_s$s.log" 2>&1 \
        || stamp "${GROUP}_s$s stopped with an error (logs_${GROUP}_s$s.log)"
    done
  done
  # A run that never finished may still have left a best.pth; such a group is not scored.
  missing=""
  for s in $SEEDS; do [ -f "runs/${GROUP}_s$s/history.json" ] || missing="$missing s$s"; done
  if [ -n "$missing" ]; then stamp "$GROUP: no finished run for$missing -> not evaluated"; continue; fi
  stamp "trained $GROUP: $(grep -ohE 'best val mIoU: [0-9.]+%' "$ROOT"/logs_${GROUP}_s?.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
  $PY split_half_check.py $REF "$GROUP"
  for ref in $REF; do
    gpu_wait 12000
    $PY compare_groups.py "$ref" "$GROUP" 2>&1 | grep -v "Warning\|^cached"
  done
  gpu_wait 12000
  $PY eval_with_best.py "$GROUP" ${ALSO:+--also $ALSO} 2>&1 | grep -v -i "warn"
  stamp "finished $GROUP"
done
stamp "all finished"
