#!/usr/bin/env bash
# Rebuild everything downstream of manifest.csv after the tts2 voices were added (2026-09-29).
# Only segmentation (Whisper on the new TTS clips) uses a GPU: one, GPU=${GPU:-1}. Everything else is CPU.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=""
D="$(cd "$(dirname "$0")/.." && pwd)"
cd "$D/segmentation_windowing"
CUDA_VISIBLE_DEVICES="${GPU:-1}" python segment_annotate.py all 2>&1 | grep -E "^\[|Error|Traceback"
python validate_windows.py | grep -E "FAIL|checks passed"
python test_label_rules.py | grep -E "FAIL|passed"
V="$D/evaluation/isolated_set/archive/v5_2026-09-29"
mkdir -p "$V"
mv "$D/evaluation/isolated_set/isolated_validation_manifest.csv" "$D/evaluation/isolated_set/isolated_validation_manifest.json" "$V/"
cat > "$V/WHY_ARCHIVED.md" <<'EOF'
# Isolated validation set v5 (archived 2026-09-29)

3,848 validation windows (65 positive; the 27 synthetic positives were all one voice, en_US-amy-medium).
Archived deliberately, not because of an error: generate_tts_voices_v2.py added 625 Whisper-verified
"Hey Delta" clips and 809 near-miss clips from 405 new Piper speakers (each speaker one source group),
because BC-ResNet round-1 models memorised the 5 training TTS voices (synthetic validation detection fell to
10-40% while real RPI reached 100%). v6 is frozen on the next evaluation run. Round-1 BC-ResNet results
(model/runs/bcresnet3, bcresnet6) were computed on v5.
EOF
cd "$D/preprocessing"
python compute_normalization_stats.py | tail -1
python test_preprocessing.py | grep -E "FAIL|passed"
cd "$D/dataloading"
python test_dataloader.py | grep -E "FAIL|passed"
cd "$D/evaluation"
python -c "import isolated_eval as iso; d, m = iso.load_isolated_set('validation'); print('isolated v6 frozen:', len(m), 'windows,', (m.label=='positive').sum(), 'positive'); print(m[m.label=='positive'].groupby('eval_subset').size().to_dict())"
echo REBUILD_DONE
