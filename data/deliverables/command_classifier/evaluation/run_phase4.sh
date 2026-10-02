#!/bin/bash
# Phase 4: test results at both validation-chosen cutoffs, then streaming evaluation of both models.
set -o pipefail
cd "$(dirname "$0")"
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
$PY evaluate_commands.py ../model/runs/bcresnet3 ../model/runs/bcresnet6 2>&1 | grep -v -i warn || exit 1
for m in bcresnet3 bcresnet6; do
  $PY streaming_commands.py ../model/runs/$m --streams 40 2>&1 | grep -v -i warn | grep -v "^stream " || exit 2
done
$PY phase4_report.py 2>&1 | grep -v -i warn || exit 3
