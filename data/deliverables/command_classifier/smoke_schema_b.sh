#!/bin/bash
# Smoke test of steps 5-8 with a tiny run (1 epoch, 512 draws) before the overnight run. Outputs in scratchpad.
set -o pipefail
CC="/mnt/jfs_hpc/home/arvir.jane.redondo/sandbox/AI231/ME2 v2/data/deliverables/command_classifier"; cd "$CC"
S="$CC/smoke_schema_b"
mkdir -p "$S"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
export COMMAND_LOADER_CONFIG="$CC/dataloading/dataloader_config_schema_b.json"
export COMMAND_NORM_STATS="$CC/preprocessing/normalization_stats_schema_b.json"
export COMMAND_RESULTS_DIR="$S/results"
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
for arch in bcresnet dscnn; do
  ( cd model && $PY train_commands.py --arch $arch --tau 6 --epochs 1 --epoch-draws 512 --out "$S/run_$arch" ) || { echo "SMOKE FAIL train $arch"; exit 1; }
done
( cd evaluation && $PY evaluate_commands.py "$S/run_bcresnet" "$S/run_dscnn" ) || { echo "SMOKE FAIL eval"; exit 1; }
( cd evaluation && $PY unseen_speakers_report.py run_bcresnet run_dscnn > /dev/null ) || { echo "SMOKE FAIL unseen"; exit 1; }
( cd model && $PY export_commands_onnx.py "$S/run_dscnn" --out "$S/export_dscnn" --n-check 100 ) || { echo "SMOKE FAIL export"; exit 1; }
echo "SMOKE OK"
