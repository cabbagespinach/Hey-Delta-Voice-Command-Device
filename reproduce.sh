#!/bin/bash
# Reproduces this project in ONE command, from a fresh clone on the AI231 HPC server (see docs/REPRODUCE.md). ONE GPU.
# If the data is not set up yet, it first runs  bash setup_data.sh --server  itself (SHARED_DIR=<dir> for another
# shared folder); run setup_data.sh again by hand only after adding a dataset you obtained yourself.
#   bash reproduce.sh [all|wakeword|commands|hf]          GPU=<index> picks the GPU (default 0)
#   SMOKE=1 bash reproduce.sh ...                          ~2.5 h: tiny training, checks that everything runs
# Trains every model again from the data, evaluates it the same way and writes, next to the reported results,
# repro_outputs/REPRODUCTION_REPORT.md (reported vs reproduced). Reproduced runs get the suffix _repro; the reported
# runs, checkpoints and results in this repository are never overwritten. Run it inside tmux.
#   wakeword  BC-ResNet-6 "Hey Delta" (round 1c recipe): train ~30 min, evaluation ~15 min
#   commands  schema-B command classifier BC-ResNet-6 + DS-CNN baseline: ~2 x 70 min
#   hf        class-benchmark comparison (HF data only vs HF + our data): ~2 x 70 min
set -o pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
export PY=${PY:-/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=2
PART=${1:-all}; SFX=_repro
mkdir -p repro_outputs; LOG="$ROOT/repro_outputs/reproduce.log"
step() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
if [ -n "$SMOKE" ]; then SFX=_smoke; export TRAIN_ARGS="--epochs 1 --epoch-draws 512" STATS_DRAWS=200; fi
case "$ROOT" in */sandbox/AI231/*) [ -n "$REPRO_ALLOW_HERE" ] || { echo "This is the owner's working copy: reproduce in a fresh clone (docs/REPRODUCE.md)."; exit 1; };; esac
[ -f repro_outputs/optional_data.json ] || { step "setup: linking the shared data (bash setup_data.sh --server)"
  bash setup_data.sh --server ${SHARED_DIR:+"$SHARED_DIR"} 2>&1 | tee -a "$LOG"; [ "${PIPESTATUS[0]}" = 0 ] || exit 1; }

step "0 match the clip lists to the audio present (missing datasets are left out and reported)"
run "$PY" repro/optional_data.py filter || exit 1
REDUCED=$(cat repro_outputs/wakeword_reduced.flag)
QUALCOMM=$("$PY" -c "import json; print(int(json.load(open('repro_outputs/optional_data.json'))['qualcomm']['present']))")

if [ "$PART" = all ] || [ "$PART" = wakeword ]; then
  D="$ROOT/data/deliverables"; RUN=bcresnet6$SFX; R="$D/model/runs/$RUN"
  step "W1 wakeword normalisation statistics (train split only)"
  if [ "$REDUCED" = 1 ] || [ -n "$SMOKE" ]; then      # data changed: statistics and frozen sets are rebuilt
    if [ -n "$SMOKE" ]; then                             # toy statistics: the script never writes partial ones in place
      ( cd "$D/preprocessing" && run "$PY" compute_normalization_stats.py --limit 500 --out "$ROOT/repro_outputs/smoke_stats.json" ) || exit 2
      cp "$ROOT/repro_outputs/smoke_stats.json" "$D/preprocessing/normalization_stats.json"
    else
      ( cd "$D/preprocessing" && run "$PY" compute_normalization_stats.py ) || exit 2
    fi
    [ -n "$SMOKE" ] && rm -f "$D"/evaluation/isolated_set/isolated_*_manifest.* 2>/dev/null
  else                                                   # same data: recompute and check against the frozen file
    ( cd "$D/preprocessing" && run "$PY" compute_normalization_stats.py --out "$ROOT/repro_outputs/wakeword_normalization_stats.json" ) || exit 2
    cmp -s "$D/preprocessing/normalization_stats.json" repro_outputs/wakeword_normalization_stats.json \
      && step "   statistics identical to the reported ones" || step "   statistics differ from the reported file (reported file kept for training)"
  fi
  step "W2 train $RUN"
  ( cd "$D/model" && "$PY" train.py --model bcresnet6 --out "runs/$RUN" ${SMOKE:+--epochs 1} > "runs/$RUN.train.out" 2>&1 ) \
    || { tail -n 20 "$D/model/runs/$RUN.train.out" | tee -a "$LOG"; exit 3; }
  tail -n 1 "$D/model/runs/$RUN.train.out" | tee -a "$LOG"
  step "W3 evaluate $RUN (isolated validation, streaming, diagnostics)"
  ( cd "$D/evaluation" && run "$PY" run_evaluation.py --model-class bc_resnet:BCResNet6 --checkpoint "$R/best.pt" \
      --out "$R/eval" --workers 12 ${SMOKE:+--no-sweep} ) || exit 4
  ( cd "$D/evaluation" && run "$PY" diagnostics/source_confound.py --model-class bc_resnet:BCResNet6 --checkpoint "$R/best.pt" \
      --results "$R/eval" --out "$R/eval/diagnostics" > /dev/null ) || exit 4
  if [ "$QUALCOMM" = 1 ]; then
    ( cd "$D/evaluation" && run "$PY" diagnostics/hey_phrase_negatives.py --model-class bc_resnet:BCResNet6 --checkpoint "$R/best.pt" \
        --results "$R/eval" --out "$R/eval/diagnostics" ) || exit 4
  else
    mkdir -p "$R/eval/diagnostics"; echo "SKIPPED: Qualcomm Keyword Speech Dataset not present (see setup_data.sh)" \
      | tee "$R/eval/diagnostics/hey_phrase_negatives_SKIPPED.txt" | tee -a "$LOG"
  fi
  step "W4 held-out test split (once)"
  ( cd "$D/model" && WAKEWORD_RUN=$RUN run "$PY" evaluate_test.py ) || exit 5
  step "W5 ONNX export"
  ( cd "$D/model" && run "$PY" export_onnx.py --model bcresnet6 --checkpoint "runs/$RUN/best.pt" --results "runs/$RUN/eval" \
      --out "export/$RUN" ) || exit 6
fi

if [ "$PART" = all ] || [ "$PART" = commands ]; then
  step "C schema-B command classifier (BC-ResNet-6 + DS-CNN) -> command_classifier/model/runs/*_schema_b$SFX"
  ( cd data/deliverables/command_classifier && SUFFIX=$SFX NO_COMMIT=1 START=4 GPU=$CUDA_VISIBLE_DEVICES bash reproduce_schema_b.sh ) \
    2>&1 | tee -a "$LOG" | grep --line-buffered "^===" ; [ "${PIPESTATUS[0]}" = 0 ] || exit 7
fi

if [ "$PART" = all ] || [ "$PART" = hf ]; then
  step "H class-benchmark comparison (HF only vs HF + ours) -> command_classifier/model/runs/bcresnet6_hf_*$SFX"
  ( cd data/deliverables/command_classifier && SUFFIX=$SFX NO_COMMIT=1 START=3 GPU=$CUDA_VISIBLE_DEVICES bash run_hf_compare.sh ) \
    2>&1 | tee -a "$LOG" | grep --line-buffered "^===" ; [ "${PIPESTATUS[0]}" = 0 ] || exit 8
fi

step "R reported vs reproduced -> repro_outputs/REPRODUCTION_REPORT.md"
run "$PY" repro/compare_results.py "$SFX" || exit 9
step "ALL DONE"
