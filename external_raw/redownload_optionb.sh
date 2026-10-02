#!/bin/bash
# Re-download Classmate A's Option-B set (github markandrian30/AI231, MEX2/Data only), owner request 2026-10-01: the repo was
# updated. Fresh sparse clone into class_optionb_new; the old copy is replaced only after the new one checks out.
set -o pipefail
cd "$(dirname "$0")"
rm -rf class_optionb_new && mkdir class_optionb_new && cd class_optionb_new
git clone -q --filter=blob:none --no-checkout https://github.com/markandrian30/AI231.git || exit 1
cd AI231 && git sparse-checkout set MEX2/Data && git checkout -q main || exit 1
git log -1 --format='commit %H %cd %s'
echo "wav files: $(find MEX2/Data -name '*.wav' | wc -l)"
find MEX2/Data -maxdepth 2 -not -name '*.wav' | head -40
du -sh MEX2/Data
echo NEW_OK
