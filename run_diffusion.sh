#!/usr/bin/env bash
# Round 10: synthetic image-mask pairs from a joint diffusion model.
#
#   1. train the diffusion model on train/train               (HOURS, default 3)
#   2. sample COUNT (1000) pairs of 512 x 512                 -> synth/d1
#   3. quality checks                                         -> synth/d1/quality.json
#   4. team-protocol row (batch 4, 12000 steps, seed 42)      -> AUG_TABLE_RESULTS.md
#   5. scale-1 screen on the aug_color recipe, 3 seeds, with 30% and with 60%
#      of the crops synthetic, against c_ep60
#   6. the better ratio on cs2_aug3 (scale 2), 3 seeds, against b_cs2_aug3
#
# Every stage is skipped if its output exists, so this can be restarted.
#   nohup ./run_diffusion.sh > suite_diffusion.log 2>&1 &
# WAIT_PID=<pid> queues it behind another job (it starts when that process is
# gone); setsid keeps it running after the terminal or editor is closed:
#   WAIT_PID=12345 setsid nohup ./run_diffusion.sh > suite_diffusion.log 2>&1 < /dev/null &
# Only one copy per NAME runs: a second one finds the lock taken and leaves.
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
NAME=${NAME:-d1}
HOURS=${HOURS:-3}
COUNT=${COUNT:-1000}
DIFF=runs/diffusion_$NAME
SYNTH=synth/$NAME
COMMON="--arch unetpp --encoder efficientnet-b3 --pretrain micronet --fullres_skip --color_aug oneof"
stamp() { echo "[$(date '+%F %T')] $*"; }

exec 9> ".run_diffusion_$NAME.lock"
if ! flock -n 9; then
  stamp "run_diffusion.sh for $NAME is already running or queued; this copy leaves"; exit 0
fi
if [ -n "${WAIT_PID:-}" ]; then
  stamp "queued behind process $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  stamp "process $WAIT_PID is gone; starting"
fi
# STOP_AFTER=0 ends here: the queueing alone, for a dry run.
if [ "${STOP_AFTER:-6}" -le 0 ]; then stamp "stopping before stage 1"; exit 0; fi

train3() {  # group name, train_v2.py flags: three seeds at once, finished runs skipped
  for s in 0 1 2; do
    out=runs/${1}_s$s
    if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
    $PY train_v2.py $2 --seed $s --out "$out" > "logs_${1}_s$s.log" 2>&1 &
  done
  wait
  stamp "trained $1: $(grep -ohE 'best val mIoU: [0-9.]+%' logs_${1}_s*.log | grep -oE '[0-9.]+%' | tr '\n' ' ')"
}

if [ ! -f "$DIFF/done" ]; then
  stamp "1. train the diffusion model for $HOURS h -> $DIFF"
  if ! $PY train_diffusion.py --out "$DIFF" --hours "$HOURS" > "logs_diffusion_$NAME.log" 2>&1; then
    stamp "diffusion training failed, see logs_diffusion_$NAME.log"; exit 1
  fi
  touch "$DIFF/done"
fi
grep -E "^U-Net|^step|^stopping|^best held-out" "logs_diffusion_$NAME.log" | tail -40

if [ ! -f "$SYNTH/meta.json" ]; then
  stamp "2. sample $COUNT pairs -> $SYNTH"
  if ! $PY sample_diffusion.py --ckpt "$DIFF/best.pt" --out "$SYNTH" --count "$COUNT" > "logs_sample_$NAME.log" 2>&1; then
    stamp "sampling failed, see logs_sample_$NAME.log"; exit 1
  fi
fi
tail -1 "logs_sample_$NAME.log"

stamp "3. quality checks"
[ -f "$SYNTH/quality.json" ] || $PY check_synth.py "$SYNTH" 2>&1 | grep -v Warning
# STOP_AFTER=3 ends here: the generator and its samples without the segmentation runs.
if [ "${STOP_AFTER:-6}" -le 3 ]; then stamp "stopping after stage 3"; exit 0; fi

stamp "4. team-protocol row (t_diffusion_s42)"
./run_aug_table.sh diffusion 2>&1 | grep -v "Warning\|^evaluated"

stamp "5. scale-1 screen: 30% and 60% synthetic crops, 3 seeds each"
for r in 30 60; do
  train3 a_diff$r "$COMMON --synth_dir $SYNTH --synth_ratio 0.$r"
done
$PY split_half_check.py c_ep60 a_diff30 a_diff60
for r in 30 60; do
  $PY compare_groups.py c_ep60 a_diff$r 2>&1 | grep -v "Warning\|^cached"
done

# The ratio whose single models score higher under the final inference setting goes on to scale 2.
BEST=$($PY - <<'EOF'
import json
score = {}
for r in (30, 60):
    with open(f"runs/compare_c_ep60_vs_a_diff{r}.json") as f:
        rows = json.load(f)
    score[r] = sum(rows[f"a_diff{r}_s{s}"]["miou"] for s in range(3)) / 3
print(max(score, key=score.get))
EOF
)
BEST=${BEST:-30}
stamp "6. scale 2: cs2_aug3 with ${BEST}% synthetic crops, 3 seeds"
train3 b_cs2_aug3_diff "$COMMON --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2 --synth_dir $SYNTH --synth_ratio 0.$BEST"
$PY split_half_check.py b_cs2_aug3 b_cs2_aug3_diff
$PY compare_groups.py b_cs2_aug3 b_cs2_aug3_diff 2>&1 | grep -v "Warning\|^cached"
stamp "all finished"
