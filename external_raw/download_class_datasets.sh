#!/bin/bash
# Datasets from the class list (data/AI231 ME2 - Datasets.csv + github markandrian30/AI231 MEX2/Data) that were not yet
# in this project (owner request 2026-10-01). Raw downloads only, one after another; nothing is integrated here.
# Each step skips what is already there, so after a disconnect just run it again:
#   tmux new -s classdl "bash download_class_datasets.sh"        (log: download_class_datasets.log)
set -o pipefail
cd "$(dirname "$0")"
LOG="$PWD/download_class_datasets.log"
PY=/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python
say() { echo "=== $1 $(date '+%m-%d %H:%M:%S')" | tee -a "$LOG"; }
get() { [ -s "$2" ] && [ ! -e "$2.part" ] && { echo "have $2" | tee -a "$LOG"; return 0; }
        touch "$2.part"; curl -sSL --retry 5 -C - -o "$2" "$1" 2>&1 | tee -a "$LOG" && rm -f "$2.part"; }
FAIL=0

say "1 Classmate A's Option B synthetic set (github markandrian30/AI231, MEX2/Data only)"
mkdir -p class_optionb && cd class_optionb
if [ ! -d AI231/.git ]; then
  git clone -q --filter=blob:none --no-checkout https://github.com/markandrian30/AI231.git 2>&1 | tee -a "$LOG"
fi
( cd AI231 && git sparse-checkout set MEX2/Data && git checkout -q main && git log -1 --format='commit %H %cd' ) 2>&1 | tee -a "$LOG" || FAIL=1
echo "wav files: $(find AI231/MEX2/Data -name '*.wav' | wc -l)" | tee -a "$LOG"
cd ..

say "2 Google Speech Commands v0.02 (CC-BY 4.0)"
mkdir -p google_speech_commands && cd google_speech_commands
get http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz speech_commands_v0.02.tar.gz || FAIL=1
if [ ! -e extracted.ok ] && [ -s speech_commands_v0.02.tar.gz ]; then
  mkdir -p v0.02 && tar -xzf speech_commands_v0.02.tar.gz -C v0.02 && touch extracted.ok || FAIL=1
fi
echo "wav files: $(find v0.02 -name '*.wav' 2>/dev/null | wc -l)" | tee -a "$LOG"
cd ..

say "3 Snips SLU v1.0 (huggingface MWilinski/snips_slu_v1.0, parquet)"
mkdir -p snips_slu && cd snips_slu
HF=https://huggingface.co/datasets/MWilinski/snips_slu_v1.0/resolve/main
get $HF/README.md README.md
for f in train-00000-of-00002-8a79c77a8d003471.parquet train-00001-of-00002-30a6f2d916e23e78.parquet; do
  get $HF/data/$f $f || FAIL=1
done
cd ..

say "4 SynTTS-Commands (huggingface lugan/Syntts-Commands-Media-Dataset, MIT): English only"
mkdir -p syntts_commands && cd syntts_commands
HF=https://huggingface.co/datasets/lugan/Syntts-Commands-Media-Dataset/resolve/main
get $HF/README.md README.md
get $HF/comprehensive_metadata.csv comprehensive_metadata.csv || FAIL=1
mkdir -p Free_ST_English VoxCeleb12_English
for z in $(curl -s "https://huggingface.co/api/datasets/lugan/Syntts-Commands-Media-Dataset/tree/main/Free_ST_English" \
           | $PY -c "import json,sys;print(' '.join(x['path'].split('/')[-1] for x in json.load(sys.stdin)))"); do
  get "$HF/Free_ST_English/$z" "Free_ST_English/$z" || FAIL=1
done
# the larger VoxCeleb subset (13.5 GB): only the commands that map to schema-B intents
for z in Pause.zip Next_track.zip Skip_song.zip Volume_up.zip Volume_down.zip; do
  get "$HF/VoxCeleb12_English/$z" "VoxCeleb12_English/$z" || FAIL=1
done
cd ..

say "5 Classmate F's Google Drive file (VCM.zip)"
mkdir -p class_vcm && cd class_vcm
[ -s VCM.zip ] || $PY -m gdown -q 1miLmuozdojOylqD0xlfOwXAdNsbGYwfN -O VCM.zip 2>&1 | tee -a "$LOG" || FAIL=1
ls -l VCM.zip 2>&1 | tee -a "$LOG"
cd ..

say "sizes"
du -sh class_optionb google_speech_commands snips_slu syntts_commands class_vcm 2>&1 | tee -a "$LOG"
[ $FAIL -eq 0 ] && say "ALL DONE" || say "FINISHED WITH ERRORS (see above); re-run to resume"
