#!/usr/bin/env bash
# Train each new augmentation on the three fullres B3 seeds, then compare the
# resulting B3x3 + original B4 ensemble at the established inference settings.
#
#   ./run_augmentation_suite.sh                     # five single-method studies
#   SEEDS=0 ./run_augmentation_suite.sh illumination # short exploratory run
#   MODE=eval ./run_augmentation_suite.sh blur       # evaluate finished 3 seeds
#   ./run_augmentation_suite.sh imaging_combo        # optional after singles
#
# Run directories: runs/a_<method>_s<seed>
# Ensemble predictions/scores: runs/a_ensemble_<method>_valid
set -euo pipefail
cd "$(dirname "$0")"

PY=./.conda/bin/python
MODE=${MODE:-all}
JOBS=${JOBS:-1}
SEEDS=${SEEDS:-"0 1 2"}
[[ "$MODE" == train || "$MODE" == eval || "$MODE" == all ]] || {
  echo "MODE must be train, eval, or all" >&2; exit 2;
}
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "JOBS must be positive" >&2; exit 2; }

declare -A FLAGS=(
  [illumination]="--illumination_p 0.2"
  [blur]="--blur_p 0.15"
  [fourier]="--fourier_p 0.1"
  [context_paste]="--context_paste_p 0.1"
  [si_attenuate]="--si_attenuate_p 0.05"
  [imaging_combo]="--illumination_p 0.2 --blur_p 0.15 --fourier_p 0.1"
)
DEFAULT=(illumination blur fourier context_paste si_attenuate)
if (( $# )); then METHODS=("$@"); else METHODS=("${DEFAULT[@]}"); fi
for method in "${METHODS[@]}"; do
  [[ -n "${FLAGS[$method]+x}" ]] || {
    echo "Unknown method: $method (choose: ${!FLAGS[*]})" >&2; exit 2;
  }
done

if [[ "${ALLOW_CPU:-0}" != 1 ]] && ! $PY -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  echo "CUDA is unavailable. Connect a GPU, or set ALLOW_CPU=1 for a slow CPU run." >&2
  exit 1
fi
if [[ "$MODE" != eval && "${ALLOW_CPU:-0}" != 1 ]]; then
  free_bytes=$($PY -c 'import torch; print(torch.cuda.mem_get_info()[0])')
  required_bytes=$((JOBS * 9 * 1024 * 1024 * 1024))
  if (( free_bytes < required_bytes )); then
    echo "Only $((free_bytes / 1024 / 1024)) MiB GPU memory is free; "\
         "need about $((required_bytes / 1024 / 1024)) MiB for JOBS=$JOBS." >&2
    exit 1
  fi
fi

train_one() {
  local method=$1 seed=$2
  local out="runs/a_${method}_s${seed}" log="logs_a_${method}_s${seed}.log"
  if [[ -f "$out/history.json" && -f "$out/best.pth" ]]; then
    echo "skip $out (finished)"
    return 0
  fi
  echo "start $out ${FLAGS[$method]}"
  # All other settings match runs/b_fullres_s0..2, including legacy color.
  if $PY train_v2.py --arch unetpp --encoder efficientnet-b3 --pretrain micronet \
      --fullres_skip ${FLAGS[$method]} --seed "$seed" --out "$out" > "$log" 2>&1; then
    echo "done $out $(rg 'best val mIoU' "$log")"
  else
    echo "FAIL $out (see $log)" >&2
    return 1
  fi
}

if [[ "$MODE" == train || "$MODE" == all ]]; then
  active=0
  failed=0
  for seed in $SEEDS; do
    for method in "${METHODS[@]}"; do
      train_one "$method" "$seed" &
      active=$((active + 1))
      if (( active >= JOBS )); then
        if ! wait -n; then failed=1; fi
        active=$((active - 1))
      fi
    done
  done
  while (( active > 0 )); do
    if ! wait -n; then failed=1; fi
    active=$((active - 1))
  done
  if (( failed )); then
    echo "One or more training runs failed; inspect logs_a_*.log" >&2
    exit 1
  fi
  $PY summarize_runs.py --pattern 'runs/a_*_s*' \
      --baseline runs/b_fullres_s0 runs/b_fullres_s1 runs/b_fullres_s2
fi

if [[ "$MODE" == eval || "$MODE" == all ]]; then
  baseline_dir=runs/a_ensemble_baseline_valid
  if [[ ! -f "$baseline_dir/scores.json" ]]; then
    $PY predict.py --ckpt \
      runs/b_fullres_s0/best.pth runs/b_fullres_s1/best.pth \
      runs/b_fullres_s2/best.pth runs/s4_micronet_b4/best.pth \
      --split valid --tta --d4 --tile 768 --si_vote --save_dir "$baseline_dir" \
      > logs_a_ensemble_baseline_valid.log 2>&1
  fi
  for method in "${METHODS[@]}"; do
    ckpts=()
    for seed in 0 1 2; do
      ckpt="runs/a_${method}_s${seed}/best.pth"
      if [[ ! -f "$ckpt" ]]; then
        echo "skip ensemble $method: missing $ckpt" >&2
        ckpts=()
        break
      fi
      ckpts+=("$ckpt")
    done
    (( ${#ckpts[@]} == 3 )) || continue
    out="runs/a_ensemble_${method}_valid"
    if [[ -f "$out/scores.json" ]]; then
      echo "skip $out (finished)"
      continue
    fi
    echo "evaluate ensemble $method"
    $PY predict.py --ckpt "${ckpts[@]}" runs/s4_micronet_b4/best.pth \
      --split valid --tta --d4 --tile 768 --si_vote --save_dir "$out" \
      > "logs_a_ensemble_${method}_valid.log" 2>&1
  done
  $PY summarize_augmentation_ensembles.py
fi
