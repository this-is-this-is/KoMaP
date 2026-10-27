#!/usr/bin/env bash
# Round 12: HRNet as a second model family (not U-shaped, ImageNet weights), trained
# with the cs2_aug3 recipe at scale 2, 60 epochs, 3 seeds, then the selection-free
# check and the final-inference comparison with b_cs2_aug3. The last row of that
# comparison is the six models together: whether a different family adds to the
# U-Net++ ensemble is what this run is for.
#
#   setsid nohup ./run_hrnet.sh > suite_hrnet.log 2>&1 < /dev/null &      HRNet-W18 -> runs/b_hrnet18_cs2_aug3_s*
#   ENCODER=hrnet_w32 setsid nohup ./run_hrnet.sh > suite_hrnet32.log 2>&1 < /dev/null &
#
# As in run_synth_variants.sh: the code is copied to .snap/<name> at the first launch
# and run from there, and a run starts only when the GPU has its memory plus
# RESERVE_MB free. Finished runs are skipped; a second copy leaves at once.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
ENCODER=${ENCODER:-hrnet_w18}
TAG=${ENCODER/_w/}               # hrnet_w18 -> hrnet18
P=${PREFIX:-}                    # prefix of the run names (a smoke test uses its own)
EXTRA=${EXTRA:-}                 # appended to every training command
RESERVE_MB=${RESERVE_MB:-20000}  # GPU memory left free for whoever else needs it
NEED_MB=24000                    # one run at scale 2 (20 GiB allocated at batch 8)
PY="$ROOT/.conda/bin/python"
GROUP=${P}b_${TAG}_cs2_aug3
stamp() { echo "[$(date '+%F %T')] $*"; }

exec 9> ".run_${P}${TAG}.lock"
if ! flock -n 9; then stamp "run_hrnet.sh ($GROUP) is already running; this copy leaves"; exit 0; fi

SNAP="$ROOT/.snap/${P}${TAG}"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp train_v2.py predict.py compare_groups.py split_half_check.py "$SNAP/"
  cp -r src "$SNAP/src"
  for d in train valid test runs synth weights; do ln -s "$ROOT/$d" "$SNAP/$d"; done
  stamp "code frozen in $SNAP"
fi
cd "$SNAP"

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
# The cs2_aug3 recipe; the full-resolution branch is part of the HRNet model here.
FLAGS="--arch hrnet --encoder $ENCODER --pretrain imagenet --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"

stamp "train $GROUP x 3 seeds"
for pass in 1 2; do   # a run that died gets one more try
  for s in 0 1 2; do
    out=runs/${GROUP}_s$s
    [ -f "$out/history.json" ] && continue
    while [ "$(jobs -rp | wc -l)" -ge 3 ] || [ "$(free_mb)" -lt $(( NEED_MB + RESERVE_MB )) ]; do sleep 20; done
    stamp "start ${GROUP}_s$s"
    $PY train_v2.py $FLAGS $EXTRA --seed $s --data_root . --out "$out" > "$ROOT/logs_${GROUP}_s$s.log" 2>&1 &
    sleep "${STAGGER:-120}"   # let it take its memory before the next one is weighed
  done
  wait
done
stamp "trained $GROUP: $(grep -ohE 'best val mIoU: [0-9.]+%' "$ROOT"/logs_${GROUP}_s?.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
$PY split_half_check.py b_cs2_aug3 "$GROUP"
while [ "$(free_mb)" -lt $(( 12000 + RESERVE_MB )) ]; do sleep 20; done
$PY compare_groups.py b_cs2_aug3 "$GROUP" 2>&1 | grep -v "Warning\|^cached"
stamp "all finished"
