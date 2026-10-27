#!/usr/bin/env bash
# Boundary-precision experiments: every method x 3 seeds on MicroNet B3 U-Net++.
#
# Seed noise on this data is about +-0.15pp, so a method needs all 3 seeds
# before it counts. The matching ce_dice baseline (default flags, enc_lr 3e-4)
# is runs/seed_base_{0,1,2}; s2_micronet_unetpp used --enc_lr 1e-4, so not it.
#
#   ./run_boundary_suite.sh                  # everything, one run at a time
#   JOBS=3 ./run_boundary_suite.sh           # 3 runs side by side on the GPU
#   ./run_boundary_suite.sh fullres relax    # only these methods
#   SEEDS=0 ./run_boundary_suite.sh          # quick single-seed pass first
#
# Runs land in runs/b_<method>_s<seed>, logs in logs_b_<method>_s<seed>.log.
# Finished runs (history.json present) are skipped, so an interrupted suite
# can simply be started again. Summary: ./.conda/bin/python summarize_runs.py
set -uo pipefail
cd "$(dirname "$0")"

PY=./.conda/bin/python
BASE="--arch unetpp --encoder efficientnet-b3 --pretrain micronet"
JOBS=${JOBS:-1}
SEEDS=${SEEDS:-"0 1 2"}

declare -A FLAGS=(
  [fullres]="--fullres_skip"                         # stride-1 branch before the head
  [relax]="--ce_mode relax"                          # label relaxation, Zhu+ CVPR19
  [jml_bls]="--loss ce_jml --bls_eps 0.5"            # JML + boundary label smoothing, NeurIPS23
  [jml]="--loss ce_jml"                              # control for jml_bls: same loss, hard labels
  [interface]="--ce_mode interface"                  # Al3Ni-Si interface weights, Strohmann+ 2019
  [bdou]="--loss ce_bdou"                            # Boundary DoU, MICCAI23
  [kervadec]="--aux_loss kervadec"                   # boundary (surface) loss, MIDL19
  [abl]="--loss ce_lovasz --aux_loss abl"            # Active Boundary Loss, AAAI22
  [lovasz]="--loss ce_lovasz"                        # control for abl: its base loss alone
  # Round 2 (2026-10-01): fullres was the only clear win, so push resolution.
  # fullres_b4 keeps s4_micronet_b4's settings (enc_lr 1e-4) to compare with it.
  [fullres_b4]="--encoder efficientnet-b4 --enc_lr 1e-4 --fullres_skip"
  [fullres_ch64]="--fullres_skip --fullres_ch 64"    # wider stride-1 branch
  [fullres_lovasz]="--fullres_skip --loss ce_lovasz"
  # Round 3: augmentation / sampling on top of fullres. Compare with fullres:
  #   ./.conda/bin/python summarize_runs.py --baseline runs/b_fullres_s{0,1,2}
  [aug_d4]="--fullres_skip --aug_d4"                 # uniform D4 instead of flips + rot90
  [aug_color]="--fullres_skip --color_aug oneof"     # one of brightness/contrast/gamma, +-10%
  [scale15]="--fullres_skip --input_scale 1.5"       # 2.25x compute per run
  [scale2]="--fullres_skip --input_scale 2"          # 4x compute per run
  [thin30]="--fullres_skip --thin_ratio 0.3"         # 30% crops on thin, dense Eutectic Si
  [cutmix20]="--fullres_skip --cutmix_p 0.2"
  # Round 4: the two round-3 wins together (4x compute, ~26 GiB per run).
  [color_scale2]="--fullres_skip --color_aug oneof --input_scale 2"
  # Round 6: color_scale2 plus everything that held up on the aug_color recipe.
  # Small scans (<= 800 px tall) go to x4 while the rest stay at x2, which is
  # the scale-2 counterpart of the tested "small scans x2, rest x1".
  [cs2_plus]="--fullres_skip --color_aug oneof --input_scale 2 --scale_by_height 800:4 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"
  # cs2_plus without the x4 small scans, the suspect for its Al3Ni drop.
  [cs2_aug3]="--fullres_skip --color_aug oneof --input_scale 2 --prisi_paste_p 0.25 --fourier_p 0.1 --illumination_p 0.2"
)
ORDER=(fullres relax jml_bls jml interface bdou kervadec abl lovasz
       fullres_b4 fullres_ch64 fullres_lovasz
       aug_d4 aug_color scale15 scale2 thin30 cutmix20
       color_scale2 cs2_plus cs2_aug3)

if [ $# -gt 0 ]; then METHODS=("$@"); else METHODS=("${ORDER[@]}"); fi
for m in "${METHODS[@]}"; do
  [ -n "${FLAGS[$m]+x}" ] || { echo "unknown method '$m'; choose from: ${ORDER[*]}"; exit 1; }
done

run() {
  local m=$1 s=$2
  local out=runs/b_${m}_s${s} log=logs_b_${m}_s${s}.log
  if [ -f "$out/history.json" ]; then echo "skip  $out (already finished)"; return; fi
  echo "start $out  [${FLAGS[$m]}]"
  if $PY train_v2.py $BASE ${FLAGS[$m]} --seed "$s" --out "$out" > "$log" 2>&1; then
    echo "done  $out  $(grep 'best val mIoU' "$log")"
  else
    echo "FAIL  $out  (see $log)"
  fi
}

# Seeds outer, methods inner: after the first pass every method has one run.
for s in $SEEDS; do
  for m in "${METHODS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    run "$m" "$s" &
  done
done
wait

$PY summarize_runs.py
