#!/bin/bash
# Class-benchmark comparison (owner, 2026-10-02): is the class's Hugging Face dataset enough on its own?
# Two BC-ResNet-6 models, same recipe as bcresnet6_schema_b, scored on HF test (the class benchmark):
#   bcresnet6_hf_only   trained on HF train only
#   bcresnet6_hf_plus   trained on HF train + our data (no HF test/holdout speaker, no clip already in HF)
#   tmux new -s hfcmp "bash run_hf_compare.sh"        (restart at step N: START=N bash run_hf_compare.sh)
# Steps: 1 HF download + WAVs | 2 clip lists | 3 normalisation statistics (each model) | 4 smoke test |
#        5 train hf_only | 6 train hf_plus | 7 test both (HF test) | 8 ONNX export | 9 comparison table | 10 git commit
# ONE GPU, pinned. 12 loader workers as before.
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
ROOT="$(cd ../../.. && pwd)"
export CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=2
PY=${PY:-/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python}
# reproduction (reproduce.sh): SUFFIX=_repro names every output anew, NO_COMMIT=1 skips step 10,
# TRAIN_ARGS / STATS_DRAWS shrink a smoke test
SFX=${SUFFIX:-}
LOG="$CC/run_hf_compare.log"
step() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
use() {   # use <hf_only|hf_plus>: loader config, normalisation statistics and results folder of that model
  export COMMAND_LOADER_CONFIG="$CC/dataloading/dataloader_config_$1.json"
  export COMMAND_NORM_STATS="$CC/preprocessing/normalization_stats_$1$SFX.json"
  export COMMAND_RESULTS_DIR="$CC/evaluation/results_$1$SFX"; }
START=${START:-1}

if [ "$START" -le 1 ]; then step "1 HF dataset -> data/commands_hf/"
  [ -s "$ROOT/data/commands_hf/manifest.csv" ] || run $PY hf_extract.py || exit 1; fi
if [ "$START" -le 2 ]; then step "2 clip lists -> data/commands_hf_only.csv, data/commands_hf_plus.csv"
  run $PY build_hf_lists.py || exit 2; fi
if [ "$START" -le 3 ]; then
  for m in hf_only hf_plus; do step "3 normalisation statistics ($m, training draws only)"; use $m
    run $PY preprocessing/compute_normalization_stats.py --draws ${STATS_DRAWS:-4000} || exit 3; done; fi
if [ "$START" -le 4 ] && [ -z "$SFX" ]; then step "4 smoke test (1 tiny epoch per model + test pass)"
  S="$CC/smoke_hf"; rm -rf "$S"; mkdir -p "$S"
  for m in hf_only hf_plus; do use $m; export COMMAND_RESULTS_DIR="$S/results_$m"
    ( cd model && $PY train_commands.py --tau 6 --epochs 1 --epoch-draws 512 --out "$S/run_$m" ) >> "$LOG" 2>&1 || { step "SMOKE FAILED (train $m)"; exit 4; }
    ( cd evaluation && $PY evaluate_commands.py "$S/run_$m" ) >> "$LOG" 2>&1 || { step "SMOKE FAILED (test $m)"; exit 4; }
  done; rm -rf "$S"; step "smoke OK"; fi
if [ "$START" -le 5 ]; then step "5 train bcresnet6_hf_only (HF train only)"; use hf_only
  ( cd model && $PY train_commands.py --tau 6 ${TRAIN_ARGS:-} --out runs/bcresnet6_hf_only$SFX > runs/bcresnet6_hf_only$SFX.log 2>&1 ) || { tail -n 20 model/runs/bcresnet6_hf_only$SFX.log | tee -a "$LOG"; exit 5; }
  tail -n 2 model/runs/bcresnet6_hf_only$SFX.log | tee -a "$LOG"; fi
if [ "$START" -le 6 ]; then step "6 train bcresnet6_hf_plus (HF train + our data)"; use hf_plus
  ( cd model && $PY train_commands.py --tau 6 ${TRAIN_ARGS:-} --out runs/bcresnet6_hf_plus$SFX > runs/bcresnet6_hf_plus$SFX.log 2>&1 ) || { tail -n 20 model/runs/bcresnet6_hf_plus$SFX.log | tee -a "$LOG"; exit 6; }
  tail -n 2 model/runs/bcresnet6_hf_plus$SFX.log | tee -a "$LOG"; fi
if [ "$START" -le 7 ]; then
  for m in hf_only hf_plus; do step "7 HF test set + unseen speakers ($m)"; use $m
    ( cd evaluation && run $PY evaluate_commands.py ../model/runs/bcresnet6_$m$SFX ) || exit 7
    ( cd evaluation && run $PY unseen_speakers_report.py bcresnet6_$m$SFX > /dev/null ) || exit 7; done; fi
if [ "$START" -le 8 ]; then
  for m in hf_only hf_plus; do step "8 ONNX export ($m)"; use $m
    ( cd model && run $PY export_commands_onnx.py runs/bcresnet6_$m$SFX --out export/bcresnet6_$m$SFX ) || exit 8; done; fi
if [ "$START" -le 9 ] && [ -z "$SFX" ]; then step "9 comparison -> evaluation/results_hf_compare.md"
  run $PY evaluation/compare_hf_runs.py || exit 9; fi
if [ "$START" -le 10 ] && [ -z "${NO_COMMIT:-}" ]; then step "10 commit to the local git repository (no push)"
  cd "$ROOT" && git add -A && git commit -q -m "Class-benchmark comparison: bcresnet6_hf_only vs bcresnet6_hf_plus on HF test ($(date +%F))

HF dataset airimonda/ai231-me2-voice-commands (pinned revision); lists, logs, final checkpoints, results, ONNX.
Reproduce: data/deliverables/command_classifier/run_hf_compare.sh

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" 2>&1 | tee -a "$LOG"; git log --oneline -1 | tee -a "$LOG"; fi
step "ALL DONE"
