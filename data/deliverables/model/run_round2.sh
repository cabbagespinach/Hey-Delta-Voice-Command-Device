#!/usr/bin/env bash
# Round 2 (hard-negative mining) for the chosen width, on ONE GPU:  GPU=7 ./run_round2.sh 3
# Mines the round-1 model's highest-scoring train negatives, retrains from scratch with them oversampled,
# then evaluates exactly like round 1. At most 2 GPUs may be in use at once on the shared server.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU:-7}"
M="$(cd "$(dirname "$0")" && pwd)"
E="$M/../evaluation"
w="$1"
R="$M/${RUNS:-runs}/bcresnet${w}_hn"
mkdir -p "$R"
echo "=== bcresnet${w}_hn: mine + train ($(date +%H:%M)) ==="
python "$M/train.py" --model "bcresnet$w" --out "$R" --mine-from "$M/${RUNS:-runs}/bcresnet$w/best.pt" 2>&1 | tee "$R/train.out" \
  | grep -E "\[mine\]|\*|best epoch|Error|Traceback"
echo "=== bcresnet${w}_hn: evaluate ($(date +%H:%M)) ==="
(cd "$E" && python run_evaluation.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" --out "$R/eval" \
    | grep -E "^\[|Error")
(cd "$E" && python diagnostics/source_confound.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" \
    --results "$R/eval" --out "$R/eval/diagnostics" > /dev/null)
(cd "$E" && python diagnostics/hey_phrase_negatives.py --model-class "bc_resnet:BCResNet$w" --checkpoint "$R/best.pt" \
    --results "$R/eval" --out "$R/eval/diagnostics" | tail -6)
echo "=== round 2 done ($(date +%H:%M)) ==="
