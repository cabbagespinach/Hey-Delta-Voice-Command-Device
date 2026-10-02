#!/usr/bin/env python3
"""
Whisper check of the voice-converted owner recordings (convert_owner_voices.py output). Training uses only rows with kept = True (strict).

    CUDA_VISIBLE_DEVICES=<one gpu> python check_converted.py

Each converted clip is transcribed with Whisper large-v3 and compared with the transcript of its SOURCE recording
(recordings/_check/check.csv): the conversion must keep the words, whatever Whisper made of them. A leading
"(hey) delta" is ignored on both sides. Kept = normalised character edit ratio <= 0.25.
Writes data/commands_vc/check.csv and prints the kept rate per target voice and per label.
"""
from pathlib import Path
import re, sys

import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
VC = ROOT / "data/commands_vc"
REC = HERE.parent / "model/deploy/recordings"
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import norm, edit_ratio                 # noqa: E402

strip = lambda t: re.sub(r"^(hey\s+)?(delta|dealt|del)\b[\s,.!?]*", "", str(t).strip(), flags=re.I)


def main():
    idx = pd.read_csv(VC / "index.csv")
    src = pd.read_csv(REC / "_check/check.csv").set_index("file")
    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", device_index=0, compute_type="float16")
    prev = pd.read_csv(VC / "check.csv") if (VC / "check.csv").exists() else None
    rows = [] if prev is None else prev.to_dict("records")
    done = set() if prev is None else set(prev.file)                     # (2026-10-01) only check new clips
    for r in idx.itertuples(index=False):
        if r.file in done:
            continue
        y, _ = sf.read(str(VC / r.file), dtype="float32")
        seg, _ = wm.transcribe(y, language="en", beam_size=5)
        heard = " ".join(s.text.strip() for s in seg).strip()
        ref = src.loc[r.source_file, "heard"]
        er = round(edit_ratio(norm(strip(heard)), norm(strip(ref))), 3)
        rows.append(dict(r._asdict(), heard=heard, source_heard=ref, edit_ratio=er, kept=er <= 0.25))
    d = pd.DataFrame(rows)
    d.to_csv(VC / "check.csv", index=False)
    print(f"kept {d.kept.sum()} of {len(d)} ({d.kept.mean():.0%})")
    print(d.groupby("target_voice").kept.mean().round(2).to_string())
    print(d.groupby("label").kept.mean().round(2).sort_values().to_string())


if __name__ == "__main__":
    main()
