#!/usr/bin/env bash
# Round 13: SE-ResNeXt-50 (MicroNet weights) in place of EfficientNet-B3 -- the encoder
# NASA's own micrograph benchmark ranks first with U-Net++. Same model otherwise (U-Net++
# with the full-resolution branch) and the cs2_aug3 recipe at scale 2, 60 epochs, 3 seeds;
# then the selection-free check, the final-inference comparison with b_cs2_aug3 and the
# test that matters for an ensemble: the best six with and without the new models.
#
#   setsid nohup ./run_seresnext.sh > suite_seresnext.log 2>&1 < /dev/null &
#
# One run at scale 2 peaks at 33 GiB allocated (26 for B3), so two train at a time.
# As in run_hrnet.sh: the code is copied to .snap/<name> at the first launch and run from
# there, and a run starts only when the GPU has its memory plus RESERVE_MB free.
# Finished runs are skipped; a second copy leaves at once.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
P=${PREFIX:-}                    # prefix of the run names (a smoke test uses its own)
EXTRA=${EXTRA:-}                 # appended to every training command
RESERVE_MB=${RESERVE_MB:-12000}  # GPU memory left free for whoever else needs it
NEED_MB=40000                    # one run at scale 2
PY="$ROOT/.conda/bin/python"
GROUP=${P}b_seresnext50_cs2_aug3
stamp() { echo "[$(date '+%F %T')] $*"; }

exec 9> ".run_${P}seresnext50.lock"
if ! flock -n 9; then stamp "run_seresnext.sh ($GROUP) is already running; this copy leaves"; exit 0; fi

SNAP="$ROOT/.snap/${P}seresnext50"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp train_v2.py predict.py compare_groups.py split_half_check.py eval_with_best.py "$SNAP/"
  cp -r src "$SNAP/src"
  for d in train valid test runs synth weights; do ln -s "$ROOT/$d" "$SNAP/$d"; done
  stamp "code frozen in $SNAP"
fi
cd "$SNAP"

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_wait() { while [ "$(free_mb)" -lt $(( $1 + RESERVE_MB )) ]; do sleep 20; done; }
# b_cs2_aug3 with the encoder swapped.
FLAGS="--arch unetpp --encoder se_resnext50_32x4d --pretrain micronet --fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"

stamp "train $GROUP x 3 seeds, two at a time"
for pass in 1 2; do   # a run that died gets one more try
  for s in 0 1 2; do
    out=runs/${GROUP}_s$s
    [ -f "$out/history.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge 2 ] || [ "$(free_mb)" -lt $(( NEED_MB + RESERVE_MB )) ]; do sleep 20; done
    stamp "start ${GROUP}_s$s"
    $PY train_v2.py $FLAGS $EXTRA --seed $s --data_root . --out "$out" > "$ROOT/logs_${GROUP}_s$s.log" 2>&1 &
    sleep "${STAGGER:-120}"   # let it take its memory before the next one is weighed
  done
  wait
done
stamp "trained $GROUP: $(grep -ohE 'best val mIoU: [0-9.]+%' "$ROOT"/logs_${GROUP}_s?.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
$PY split_half_check.py b_cs2_aug3 "$GROUP"
gpu_wait 15000
$PY compare_groups.py b_cs2_aug3 "$GROUP" 2>&1 | grep -v "Warning\|^cached"
gpu_wait 15000
$PY eval_with_best.py "$GROUP" 2>&1 | grep -v -i "warn"
stamp "all finished"
