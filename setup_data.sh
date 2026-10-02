#!/bin/bash
# Step 1 of reproducing this project on the AI231 HPC server (see REPRODUCE.md).
#   bash setup_data.sh --server [SHARED_DIR]      default SHARED_DIR: /home/arvir.jane.redondo/AI231_ME2_reproduce
# Links the shared, read-only audio folder into this clone (nothing is copied), rebuilds the clips of any
# non-shareable dataset you obtained yourself (Hey Snips, Fluent Speech Commands), and prints how to obtain the
# ones that are missing. Missing datasets do NOT stop the reproduction: reproduce.sh runs without them and reports it.
# Re-run it after adding a dataset.
set -o pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/arvir.jane.redondo/.conda/envs/conda_arvir/bin/python}
[ "$1" = "--server" ] || { echo "usage: bash setup_data.sh --server [SHARED_DIR]"; exit 1; }
SHARED=${2:-/home/arvir.jane.redondo/AI231_ME2_reproduce}
"$PY" -c "import torch, torchaudio, onnxruntime, pandas, soundfile" 2>/dev/null || {
  echo "Python environment not usable: $PY"
  echo "Use the owner's environment (default, read-only) or your own: pip install -r repro/requirements-frozen.txt,"
  echo "then: PY=/path/to/python bash setup_data.sh --server"; exit 1; }
echo "== linking shared data from $SHARED"
"$PY" repro/link_data.py "$SHARED" || exit 1
echo "== rebuilding clips of datasets you obtained yourself (if any)"
"$PY" repro/optional_data.py restore || exit 1
echo "== datasets that cannot be shared"
"$PY" repro/optional_data.py check || exit 1
echo
echo "Setup done. Next:  bash reproduce.sh            (everything, ~6 h on one GPU)"
echo "                   bash reproduce.sh wakeword   |  commands  |  hf     (one part)"
