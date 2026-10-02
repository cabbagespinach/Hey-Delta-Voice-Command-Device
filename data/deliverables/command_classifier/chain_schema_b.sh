#!/bin/bash
# Overnight chain (owner, 2026-10-01): wait for steps 1-4 (tmux schemab) -> smoke test (tiny train/test/export)
# -> if OK, the full run: START=5 bash reproduce_schema_b.sh (train BC-ResNet-6, DS-CNN, test, export, git commit).
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
LOG="$CC/chain_schema_b.log"
say() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
say "waiting for steps 1-4 (tmux schemab)"
while tmux has-session -t =schemab 2>/dev/null; do sleep 30; done
grep -q "STOPPED after step 4" reproduce_schema_b.log || { say "steps 1-4 did NOT finish cleanly: not training (see reproduce_schema_b.log)"; exit 1; }
say "steps 1-4 OK; smoke test"
bash smoke_schema_b.sh 2>&1 | grep --line-buffered -v -i warn | tee -a "$LOG"
tail -n 3 "$LOG" | grep -q "SMOKE OK" || { say "smoke test FAILED: not training"; exit 2; }
rm -rf "$CC/smoke_schema_b"
say "smoke OK; full run (steps 5-9), log reproduce_schema_b.log"
START=5 bash reproduce_schema_b.sh
say "chain finished: $(tail -n 1 reproduce_schema_b.log)"
