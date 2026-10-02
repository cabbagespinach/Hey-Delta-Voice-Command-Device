#!/bin/bash
# Resume after the 2026-10-01 14:37 run: the clone/VC branch of step 1 crashed (local coverage/ folder shadowed the
# coverage package numba imports; fixed by installing coverage in envs/chatterbox). Re-run that branch, wait for the
# still-running synth branch, then continue run_improve.sh from step 2.
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
CB="$(cd ../../.. && pwd)/envs/chatterbox/bin/python"; LOG="$CC/improve.log"
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
echo "=== 1b (resume) cloned voices + voice conversion (GPU 2) $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"
run "$CB" generate_phrasing_variants.py clone --voices 30 --threads 4 &&
run "$CB" convert_owner_voices.py --extra-targets 10 --threads 4 || { echo "=== 1b FAILED" | tee -a "$LOG"; exit 1; }
while pgrep -f "generate_phrasing_variants.py synth" >/dev/null; do sleep 60; done
grep -q "Traceback" <(tail -n 5 "$LOG") && echo "check synth result before step 2" | tee -a "$LOG"
START=2 bash run_improve.sh
