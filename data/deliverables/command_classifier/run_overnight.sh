#!/bin/bash
# Overnight command-classifier pipeline (owner request 2026-09-30): data -> small-scale test -> full training -> phase 4.
# One GPU (1). Every step logs to overnight.log; a failing step stops the pipeline (except the probes: training runs
# regardless of their result, as requested).
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
LOG="$CC/overnight.log"
step() { echo "=== $1 $(date '+%H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }

START=${START:-0}
if [ "$START" -le 0 ]; then
step "0 wait for the voice job (morevoices)"
until grep -q "EXIT=" more_voices.log; do sleep 30; done
grep -q "EXIT=0" more_voices.log || { echo "voice job failed" | tee -a "$LOG"; exit 1; }

step "1 cut-off fragments";                 run $PY build_fragments.py || exit 1
step "2 combined clip list";                run $PY build_commands_all.py || exit 2
step "3 normalisation statistics (train)";  run $PY preprocessing/compute_normalization_stats.py --draws 4000 || exit 3

fi
step "4 small-scale test (shortcut probe + two short trainings)"
cd model
( OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES="" $PY probe_shortcuts.py > runs/probe_shortcuts.log 2>&1 ) &
( $PY train_commands.py --tau 3 --epochs 8 --epoch-draws 24000 --out runs/probe_all > runs/probe_all.log 2>&1 ) &
( $PY train_commands.py --tau 3 --epochs 8 --epoch-draws 24000 --exclude-groups owner converted_owner fragments_owner \
      --out runs/probe_no_owner > runs/probe_no_owner.log 2>&1 ) &
wait
grep -v -i warn runs/probe_shortcuts.log | tee -a "$LOG"; tail -3 runs/probe_all.log runs/probe_no_owner.log | tee -a "$LOG"
run $PY probe_report.py runs/probe_all runs/probe_no_owner > /dev/null
echo "(probe report: model/runs/probes/probe_report.md)" | tee -a "$LOG"

step "5 full training: BC-ResNet tau 3 and tau 6, side by side"
( $PY train_commands.py --tau 3 --out runs/bcresnet3 > runs/bcresnet3.log 2>&1 ) &
( $PY train_commands.py --tau 6 --out runs/bcresnet6 > runs/bcresnet6.log 2>&1 ) &
wait
tail -2 runs/bcresnet3.log runs/bcresnet6.log | tee -a "$LOG"
cd "$CC"

step "6 phase 4 evaluation"
if [ -x evaluation/run_phase4.sh ]; then run evaluation/run_phase4.sh || exit 6
else echo "evaluation/run_phase4.sh not found; phase 4 not run" | tee -a "$LOG"; fi
step "ALL DONE"
