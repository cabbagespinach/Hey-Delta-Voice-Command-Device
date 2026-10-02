#!/bin/bash
# Retrain BC-ResNet tau 6 with the owner extra-background fix (2026-10-01), then test and export it.
# One GPU (2). Log: ownernoise.log. A failing step stops the pipeline.
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
LOG="$CC/ownernoise.log"; RUN=bcresnet6_ownernoise
step() { echo "=== $1 $(date '+%H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }

step "0 back up current evaluation results"
cp -r evaluation/results "evaluation/results_before_ownernoise_$(date +%Y%m%d)" || exit 1
step "1 train $RUN (tau 6, owner extra background)"
( cd model && $PY train_commands.py --tau 6 --out runs/$RUN > runs/$RUN.log 2>&1 ) || { tail -n 20 model/runs/$RUN.log | tee -a "$LOG"; exit 2; }
tail -n 2 model/runs/$RUN.log | tee -a "$LOG"
step "2 test results (bcresnet3, bcresnet6, $RUN)"
( cd evaluation && run $PY evaluate_commands.py ../model/runs/bcresnet3 ../model/runs/bcresnet6 ../model/runs/$RUN ) || exit 3
step "3 streaming test"
( cd evaluation && run $PY streaming_commands.py ../model/runs/$RUN --streams 40 ) || exit 4
step "4 streaming-gap test (compare with bcresnet6: owner commands under room noise)"
run $PY evaluation/streaming_gap_test.py $RUN || exit 5
step "5 export $RUN for the Pi"
( cd model && run $PY export_commands_onnx.py runs/$RUN --out export/$RUN ) || exit 6
step "ALL DONE"
