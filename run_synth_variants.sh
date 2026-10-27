#!/usr/bin/env bash
# Round 11: (A) other ways to use the synthetic pairs of synth/d1, and
#           (B) small-scan-style scans made by shrinking the large ones (make_shrink_scans.py).
#
#   1. scale-1 screen on the aug_color recipe, 3 seeds each, against c_ep60. In every
#      group 30% of the crops come from the extra folder:
#        a_diff30_late     synthetic pairs, in the first two thirds of the epochs only
#        a_diff30_noedge   synthetic pairs, their 1 px edge band without labels
#        a_diff30_both     both of the above
#        a_small_real      the 17 real small scans again (control: only their share rises)
#        a_shrink          large scans shrunk to 480 px with a small scan's tone
#        a_shrink_sharp    the same, sharpened to the small scans' crispness
#   2. the best group of each kind on cs2_aug3 (scale 2), 3 seeds each, against b_cs2_aug3
#
# Two precautions, because another agent edits this repository and uses the GPU:
#   - the code is copied to .snap/<NAME> at the first launch and run from there,
#     so later edits to train_v2.py or src/ do not reach runs already planned;
#   - a run starts only when the GPU has its memory plus RESERVE_MB free, and at
#     most 8 scale-1 or 2 scale-2 runs train at once.
# Finished runs are skipped; a second copy with the same NAME leaves at once.
#   setsid nohup ./run_synth_variants.sh > suite_synth_variants.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
NAME=${NAME:-variants}
P=${PREFIX:-}                    # prefix of the run names (a smoke test uses its own)
EXTRA=${EXTRA:-}                 # appended to every training command
RESERVE_MB=${RESERVE_MB:-20000}  # GPU memory left free for whoever else needs it
PY="$ROOT/.conda/bin/python"
stamp() { echo "[$(date '+%F %T')] $*"; }

exec 9> ".run_synth_$NAME.lock"
if ! flock -n 9; then stamp "run_synth_variants.sh ($NAME) is already running; this copy leaves"; exit 0; fi

SNAP="$ROOT/.snap/$NAME"
if [ ! -d "$SNAP" ]; then
  mkdir -p "$SNAP"
  cp train_v2.py predict.py compare_groups.py split_half_check.py "$SNAP/"
  cp -r src "$SNAP/src"
  for d in train valid test runs synth weights; do ln -s "$ROOT/$d" "$SNAP/$d"; done
  stamp "code frozen in $SNAP"
fi
cd "$SNAP"

COMMON="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof"
SCALE2="--input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"
declare -A FLAGS=(
  [diff30_late]="--synth_dir synth/d1 --synth_ratio 0.3 --synth_until 0.67"
  [diff30_noedge]="--synth_dir synth/d1 --synth_ratio 0.3 --synth_ignore_edge 1"
  [diff30_both]="--synth_dir synth/d1 --synth_ratio 0.3 --synth_until 0.67 --synth_ignore_edge 1"
  [small_real]="--synth_dir synth/small_real --synth_ratio 0.3"
  [shrink]="--synth_dir synth/shrink --synth_ratio 0.3"
  [shrink_sharp]="--synth_dir synth/shrink_sharp --synth_ratio 0.3"
)
KIND_A=${KIND_A:-diff30_late diff30_noedge diff30_both}
KIND_B=${KIND_B:-small_real shrink shrink_sharp}

free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_wait() { while [ "$(free_mb)" -lt $(( $1 + RESERVE_MB )) ]; do sleep 20; done; }

# train NAME:KIND... : three seeds of each group, trained with FLAGS[KIND]; uses NEED_MB,
# MAXJOBS, STAGGER and BASE.
# Two passes, so a run that died (say, out of memory) gets one more try.
train() {
  for pass in 1 2; do
    for g in "$@"; do
      name=${P}${g%%:*}
      for s in 0 1 2; do
        out=runs/${name}_s$s
        [ -f "$out/history.json" ] && continue
        while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ] || [ "$(free_mb)" -lt $(( NEED_MB + RESERVE_MB )) ]; do sleep 20; done
        stamp "start ${name}_s$s"
        $PY train_v2.py $BASE ${FLAGS[${g#*:}]} $EXTRA --seed $s --data_root . --out "$out" \
          > "$ROOT/logs_${name}_s$s.log" 2>&1 &
        sleep "$STAGGER"   # let it take its memory before the next one is weighed
      done
    done
    wait
  done
  for g in "$@"; do
    name=${P}${g%%:*}
    stamp "trained $name: $(grep -ohE 'best val mIoU: [0-9.]+%' "$ROOT"/logs_${name}_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
  done
}

compare() {  # reference group, new group
  gpu_wait 12000
  $PY compare_groups.py "$1" "$2" 2>&1 | grep -v "Warning\|^cached"
}

best_of() {  # reference, groups... -> the group whose single models score highest under final inference
  $PY - "$@" <<'EOF'
import json, sys
ref, groups = sys.argv[1], sys.argv[2:]
score = {}
for g in groups:
    try:
        with open(f"runs/compare_{ref}_vs_{g}.json") as f:
            rows = json.load(f)
        runs = [v["miou"] for k, v in rows.items() if k.startswith(g + "_s")]
        score[g] = sum(runs) / len(runs)
    except (OSError, ZeroDivisionError):
        pass
print(max(score, key=score.get) if score else groups[0])
EOF
}

stamp "1. scale-1 screen: $KIND_A $KIND_B (30% of the crops from the extra folder, 3 seeds each)"
BASE="$COMMON" NEED_MB=8500 MAXJOBS=8 STAGGER=${STAGGER1:-45}
train $(for k in $KIND_A $KIND_B; do printf 'a_%s:%s ' "$k" "$k"; done)
$PY split_half_check.py c_ep60 $(for k in $KIND_A $KIND_B; do printf '%s ' "${P}a_$k"; done)
for k in $KIND_A $KIND_B; do compare c_ep60 "${P}a_$k"; done

A=$(best_of c_ep60 $(for k in $KIND_A; do printf '%s ' "${P}a_$k"; done)); A=${A#${P}a_}
B=$(best_of c_ep60 $(for k in $KIND_B; do printf '%s ' "${P}a_$k"; done)); B=${B#${P}a_}
stamp "2. scale 2 (cs2_aug3): synthetic pairs as '$A', small-scan style as '$B', 3 seeds each"
BASE="$COMMON $SCALE2" NEED_MB=32000 MAXJOBS=2 STAGGER=${STAGGER2:-120}
train "b_cs2_aug3_$A:$A" "b_cs2_aug3_$B:$B"
$PY split_half_check.py b_cs2_aug3 "${P}b_cs2_aug3_$A" "${P}b_cs2_aug3_$B"
compare b_cs2_aug3 "${P}b_cs2_aug3_$A"
compare b_cs2_aug3 "${P}b_cs2_aug3_$B"
stamp "all finished"
