#!/bin/bash
# ONE COMMAND to reproduce the schema-B command classifier (owner, 2026-10-01), from the downloaded raw datasets to
# the tested, exported model:
#   tmux new -s schemab "bash reproduce_schema_b.sh"          (restart at step N: START=N bash reproduce_schema_b.sh)
# Prerequisites: external_raw/ downloads (external_raw/download_class_datasets.sh) and the owner's data already
# relabelled to schema B (schema_b.py -> data/commands_schema_b.csv).
# Steps: 1 label classmates' datasets | 2 combined clip list | 3 dataset inventory | 4 normalisation statistics |
#        5 train BC-ResNet-6 | 6 train DS-CNN baseline (comparable size) | 7 test (all + unseen speakers) |
#        8 ONNX export | 9 commit code, logs and checkpoints to the local git repository.
# Same recipe as the current model (train_config.json, BC-ResNet tau 6); only the classes (schema B) and data differ.
# ONE GPU, pinned. CPU: 8 workers for labelling, 12 loader workers for training (as before).
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
ROOT="$(cd ../../.. && pwd)"
export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=2
export COMMAND_LOADER_CONFIG="$CC/dataloading/dataloader_config_schema_b.json"
export COMMAND_NORM_STATS="$CC/preprocessing/normalization_stats_schema_b${SUFFIX:-}.json"
export COMMAND_RESULTS_DIR="$CC/evaluation/results_schema_b${SUFFIX:-}"
PY=${PY:-/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python}
LOG="$CC/reproduce_schema_b.log"
# reproduction (repro/reproduce.sh): SUFFIX=_repro names every output anew, NO_COMMIT=1 skips step 9,
# TRAIN_ARGS / STATS_DRAWS shrink a smoke test
SFX=${SUFFIX:-}; MAIN=bcresnet6_schema_b$SFX; BASE=dscnn_schema_b$SFX
step() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
START=${START:-1}; STOP_AFTER=${STOP_AFTER:-99}
done_() { [ "$STOP_AFTER" -le "$1" ] && { step "STOPPED after step $1 (STOP_AFTER=$STOP_AFTER)"; exit 0; }; }

if [ "$START" -le 1 ]; then step "1 label the classmates' datasets -> data/commands_classmates/"
  run $PY label_classmate_datasets.py --workers 8 || exit 1; fi; done_ 1
if [ "$START" -le 2 ]; then step "2 combined schema-B clip list -> data/commands_schema_b_all.csv"
  run $PY build_schema_b_all.py || exit 2; fi; done_ 2
if [ "$START" -le 3 ]; then step "3 dataset inventory -> data/command_dataset/dataset_inventory.csv"
  run $PY make_dataset_inventory.py || exit 3; fi; done_ 3
if [ "$START" -le 4 ]; then step "4 normalisation statistics (training draws only) -> $COMMAND_NORM_STATS"
  run $PY preprocessing/compute_normalization_stats.py --draws ${STATS_DRAWS:-4000} || exit 4; fi; done_ 4
if [ "$START" -le 5 ]; then step "5 train $MAIN (BC-ResNet tau 6, current recipe) on GPU $CUDA_VISIBLE_DEVICES"
  ( cd model && $PY train_commands.py --tau 6 ${TRAIN_ARGS:-} --out runs/$MAIN > runs/$MAIN.log 2>&1 ) || { tail -n 20 model/runs/$MAIN.log | tee -a "$LOG"; exit 5; }
  tail -n 2 model/runs/$MAIN.log | tee -a "$LOG"; fi; done_ 5
if [ "$START" -le 6 ]; then step "6 train $BASE (DS-CNN baseline, ~195k parameters, same data and recipe)"
  ( cd model && $PY train_commands.py --arch dscnn ${TRAIN_ARGS:-} --out runs/$BASE > runs/$BASE.log 2>&1 ) || { tail -n 20 model/runs/$BASE.log | tee -a "$LOG"; exit 6; }
  tail -n 2 model/runs/$BASE.log | tee -a "$LOG"; fi; done_ 6
if [ "$START" -le 7 ]; then step "7 held-out test set (all + unseen speakers) -> $COMMAND_RESULTS_DIR"
  ( cd evaluation && run $PY evaluate_commands.py ../model/runs/$MAIN ../model/runs/$BASE ) || exit 7
  ( cd evaluation && run $PY unseen_speakers_report.py $MAIN $BASE ) || exit 7; fi; done_ 7
if [ "$START" -le 8 ]; then step "8 ONNX export (for the Pi 5 latency benchmark)"
  ( cd model && run $PY export_commands_onnx.py runs/$MAIN --out export/$MAIN ) || exit 8
  ( cd model && run $PY export_commands_onnx.py runs/$BASE --out export/$BASE ) || exit 8; fi; done_ 8
if [ "$START" -le 9 ] && [ -z "${NO_COMMIT:-}" ]; then step "9 commit code, logs, checkpoints and results to the local git repository (no push)"
  cd "$ROOT" && git add -A && git commit -q -m "Schema-B command classifier: $MAIN + $BASE baseline (trained $(date +%F))

Labelled classmates' datasets, combined clip list, dataset inventory, training logs, final checkpoints,
held-out and unseen-speaker test results, ONNX exports. Reproduce: data/deliverables/command_classifier/reproduce_schema_b.sh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | tee -a "$LOG"
  git log --oneline -1 | tee -a "$LOG"; fi
step "ALL DONE"
