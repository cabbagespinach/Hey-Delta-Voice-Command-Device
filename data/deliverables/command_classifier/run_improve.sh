#!/bin/bash
# Command model improvement (owner-approved 2026-10-01, after the Pi A/B test): other wordings + near-misses,
# cloned voices saying them, more voice-converted owner recordings, TV/radio background through the full 5 s window
# -> retrain BC-ResNet tau 6 (bcresnet6_improved) -> test, export, re-score the Pi A/B captures.
# ONE GPU (2). Log: improve.log. Every step resumes where it stopped; restart from step N with START=N.
#   tmux new -s improve "bash run_improve.sh"        (START=5 bash run_improve.sh to resume at step 5)
set -o pipefail
CC="$(cd "$(dirname "$0")" && pwd)"; cd "$CC"
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
CB="$(cd ../../.. && pwd)/envs/chatterbox/bin/python"
LOG="$CC/improve.log"; RUN=bcresnet6_improved
step() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
run() { "$@" 2>&1 | grep --line-buffered -v -i "warn" | tee -a "$LOG"; }
START=${START:-0}
# Owner 2026-10-01: stop after the data steps (1-3) -- more owner recordings coming before the list/training.
# Continue later with START=4 STOP_AFTER=99 bash run_improve.sh
STOP_AFTER=${STOP_AFTER:-3}

if [ "$START" -le 0 ]; then
step "0 back up the clip list, normalisation statistics and evaluation results"
B=backup_before_improve; mkdir -p $B
[ -e $B/commands_all.csv ] || cp ../../commands_all.csv $B/ || exit 1
[ -e $B/normalization_stats.json ] || cp preprocessing/normalization_stats.json $B/ || exit 1
[ -e $B/results ] || cp -r evaluation/results $B/results || exit 1
fi

if [ "$START" -le 1 ]; then
step "1 other wordings: Philippine-accent + Piper voices (CPU, internet) | in parallel: cloned voices + voice conversion (GPU 2)"
( run $PY generate_phrasing_variants.py synth --piper-voices 60 ) &
SYN=$!
( run "$CB" generate_phrasing_variants.py clone --voices 30 --threads 4 &&
  run "$CB" convert_owner_voices.py --extra-targets 10 --threads 4 ) &
GPU=$!
wait $SYN; S1=$?; wait $GPU; S2=$?
[ $S1 -eq 0 ] && [ $S2 -eq 0 ] || { step "step 1 FAILED (synth $S1, clone/vc $S2)"; exit 1; }
fi

if [ "$START" -le 2 ]; then
step "2 assemble the cloned-voice clips"
run $PY generate_phrasing_variants.py assemble-clone || exit 2
fi

if [ "$START" -le 3 ]; then
step "3 Whisper checks: other wordings, then the new voice-converted clips (GPU 2)"
run $PY generate_phrasing_variants.py check || exit 3
run $PY check_converted.py || exit 3
fi

[ "$STOP_AFTER" -lt 4 ] && { step "PAUSED after step 3 (STOP_AFTER=$STOP_AFTER); resume: START=4 STOP_AFTER=99 bash run_improve.sh"; exit 0; }
if [ "$START" -le 4 ]; then
step "4 combined clip list + normalisation statistics (train) + background-talk preview"
run $PY build_commands_all.py || exit 4
run $PY preprocessing/compute_normalization_stats.py --draws 4000 || exit 4
run $PY - <<'EOF' || exit 4
# 12 examples of the new TV/radio background, for listening: data/commands_aug_preview/talk/
import sys, numpy as np, soundfile as sf, pandas as pd
from pathlib import Path
sys.path.insert(0, "dataloading"); import command_data as cd
cfg = cd.load_config(); ds = cd.TrainDraws(cfg); aug = ds.augmenter()
out = Path(cd.ROOT) / "data/commands_aug_preview/talk"; out.mkdir(parents=True, exist_ok=True)
rows = ds.rows[ds.rows.dataset == "owner_recordings"].sample(12, random_state=1)
for i, r in enumerate(rows.itertuples(index=False)):
    y = cd.read_wave(r.path)
    for k in range(50):
        z, ap = aug.background_talk(y, np.random.default_rng(1000 * i + k), 80000)
        if ap: break
    sf.write(str(out / f"{i:02d}_{r.label}_{ap['kind']}_{ap['speech_over_bed_db']}dB{'_full' if 'full_window' in ap else ''}.wav"), z, 16000, subtype="PCM_16")
print(f"preview: {out}")
EOF
fi

if [ "$START" -le 5 ]; then
step "5 train $RUN (tau 6; new data + background talk)"
( cd model && $PY train_commands.py --tau 6 --out runs/$RUN > runs/$RUN.log 2>&1 ) || { tail -n 20 model/runs/$RUN.log | tee -a "$LOG"; exit 5; }
tail -n 2 model/runs/$RUN.log | tee -a "$LOG"
fi

if [ "$START" -le 6 ]; then
step "6 tests: held-out test set, phrasing test, streaming, room-noise gap; export for the Pi"
( cd evaluation && run $PY evaluate_commands.py ../model/runs/bcresnet3 ../model/runs/bcresnet6 ../model/runs/bcresnet6_ownernoise ../model/runs/$RUN ) || exit 6
( cd evaluation && run $PY phrasing_variants_test.py score ../model/runs/bcresnet6 ../model/runs/bcresnet6_ownernoise ../model/runs/$RUN ) || exit 6
( cd evaluation && run $PY streaming_commands.py ../model/runs/$RUN --streams 40 ) || exit 6
run $PY evaluation/streaming_gap_test.py $RUN || exit 6
( cd model && run $PY export_commands_onnx.py runs/$RUN --out export/$RUN ) || exit 6
fi

if [ "$START" -le 7 ]; then
step "7 re-score the Pi A/B captures (never trained on) with all three models"
( cd evaluation && OMP_NUM_THREADS=1 run $PY ab_replay_models.py bcresnet6 bcresnet6_ownernoise $RUN ) || exit 7
fi
step "ALL DONE"
