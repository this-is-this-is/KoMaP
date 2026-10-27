#!/usr/bin/env bash
# One augmentation at a time on the team protocol (B3, batch 4, 12000 steps,
# seed 42): every row of aug_table.py is trained on its own, then the results
# table is written to AUG_TABLE_RESULTS.md.
#
#   nohup ./run_aug_table.sh > suite_aug_table.log 2>&1 &     all 22 runs
#   ./run_aug_table.sh crop hflip d4                          only these rows
#   SEEDS="42 43 44" ./run_aug_table.sh                       more seeds
#
# Four runs at a time: one batch-4 run takes about 5 GB (15 GB at scale 2), and
# more than four at once does not raise the total throughput. Finished runs are
# skipped, so this can be restarted after an interruption.
set -uo pipefail
cd "$(dirname "$0")"
PY=./.conda/bin/python
JOBS=${JOBS:-4}
SEEDS=${SEEDS:-42}
NAMES=${*:-$($PY aug_table.py names)}
for s in $SEEDS; do
  for n in $NAMES; do
    out=runs/t_${n}_s$s
    if [ -f "$out/history.json" ]; then echo "skip $out (finished)"; continue; fi
    flags=$($PY aug_table.py flags "$n") || exit 1
    echo "[$(date '+%F %T')] start t_${n}_s$s"
    $PY train_v2.py $flags --seed "$s" --out "$out" > "logs_t_${n}_s$s.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
  done
done
wait
echo "[$(date '+%F %T')] training done: $(ls runs/t_*_s*/history.json 2>/dev/null | wc -l) runs finished"
$PY aug_table.py summary --seeds $SEEDS
echo "[$(date '+%F %T')] all finished"
