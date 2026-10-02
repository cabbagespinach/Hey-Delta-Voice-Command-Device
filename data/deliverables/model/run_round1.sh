#!/usr/bin/env bash
# Round 1: train and evaluate the given BC-ResNet widths one after another on ONE GPU (evaluation scores on CPU).
#   GPU=7 ./run_round1.sh 3 1      GPU=1 ./run_round1.sh 6
# The owner's rule: at most 2 GPUs in use at once on the shared server, never more (so at most two of these lanes).
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
M="$(cd "$(dirname "$0")" && pwd)"
E="$M/../evaluation"
for w in "$@"; do
  R="$M/${RUNS:-runs}/bcresnet$w"
  mkdir -p "$R"
  echo "=== bcresnet$w: train ($(date +%H:%M)) ==="
  python "$M/train.py" --model "bcresnet$w" --out "$R" 2>&1 | tee "$R/train.out" | grep -E "\*|best epoch|Error|Traceback"
  echo "=== bcresnet$w: evaluate ($(date +%H:%M)) ==="
  (cd "$E" && python run_evaluation.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" --out "$R/eval" \
      | grep -E "^\[|Error")
  (cd "$E" && python diagnostics/source_confound.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" \
      --results "$R/eval" --out "$R/eval/diagnostics" > /dev/null)
  (cd "$E" && python diagnostics/hey_phrase_negatives.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" \
      --results "$R/eval" --out "$R/eval/diagnostics" | tail -6)
done
echo "=== lane $* done ($(date +%H:%M)) ==="
