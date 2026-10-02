#!/bin/bash
# Phrasing test (2026-10-01): synthesize other phrasings with held-out voices now (CPU), then score once the
# ownernoise run (run_ownernoise.sh) has finished and freed GPU 2. Log: variants.log.
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
export OMP_NUM_THREADS=2
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
LOG="$CC/variants.log"
step() { echo "=== $1 $(date '+%H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
step "1 synthesize phrasings (test voices)"
( cd evaluation && run $PY phrasing_variants_test.py synth ) || exit 1
step "2 wait for ownernoise run"
while pgrep -f run_ownernoise.sh >/dev/null; do sleep 30; done
grep -q "=== ALL DONE" ownernoise.log || { step "ownernoise did not finish; scoring bcresnet6 only"; ONLY=1; }
step "3 score (GPU 2)"
RUNS="../model/runs/bcresnet6"; [ -z "$ONLY" ] && RUNS="$RUNS ../model/runs/bcresnet6_ownernoise"
( cd evaluation && CUDA_VISIBLE_DEVICES=2 run $PY phrasing_variants_test.py score $RUNS ) || exit 3
step "ALL DONE"
