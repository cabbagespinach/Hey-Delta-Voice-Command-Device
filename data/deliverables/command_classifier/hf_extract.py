#!/usr/bin/env python3
"""
Download the class's collated dataset (Hugging Face airimonda/ai231-me2-voice-commands, pinned revision) and write its
clips as WAV files (owner, 2026-10-02).

    python hf_extract.py     # -> external_raw/hf_class_dataset/*.parquet (raw, kept for provenance)
                             #    data/commands_hf/<split>/<name>.wav + data/commands_hf/manifest.csv

Splits train / test / holdout (the `numerals` set is not a split and is not used). Audio is 16 kHz mono 16-bit PCM
WAV in the parquet already (checked per clip); bytes are written as-is, anything else is converted.
"""
from pathlib import Path
import io, subprocess, sys, wave

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "external_raw/hf_class_dataset"
OUT = ROOT / "data/commands_hf"
REPO = "airimonda/ai231-me2-voice-commands"
REV = "a90b8d106349b02c5570a1a258503386043f63b2"
FILES = ["data/holdout-00000-of-00001.parquet", "data/test-00000-of-00001.parquet",
         "data/train-00000-of-00002.parquet", "data/train-00001-of-00002.parquet", "README.md", "variations.csv"]
sys.path.insert(0, str(HERE))
from generate_synthetic_commands import to16k                              # noqa: E402


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        dst = RAW / Path(f).name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{f}"
        subprocess.run(["curl", "-sSL", "--fail", "--retry", "5", "-o", str(dst) + ".part", url], check=True)
        Path(str(dst) + ".part").rename(dst)
        print(f"downloaded {f} ({dst.stat().st_size / 1e6:.0f} MB)", flush=True)
    (RAW / "SOURCE.txt").write_text(f"https://huggingface.co/datasets/{REPO} revision {REV}\n")


def write_clip(data: bytes, dst: Path):
    i = sf.info(io.BytesIO(data))
    if i.samplerate == 16000 and i.channels == 1 and i.subtype == "PCM_16" and i.format == "WAV":
        dst.write_bytes(data)
        return False
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    y = to16k(y.mean(1), sr)
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return True


def main():
    fetch()
    rows, conv = [], 0
    for f in sorted(RAW.glob("*.parquet")):
        split = f.name.split("-")[0]
        pf = pq.ParquetFile(f)
        for b in pf.iter_batches(batch_size=500):
            for r in b.to_pylist():
                a = r.pop("audio")
                dst = OUT / split / Path(r["file"]).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                conv += write_clip(a["bytes"], dst)
                rows.append(dict(path=str(dst.relative_to(ROOT)), split=split, **r))
        print(f"{f.name}: {sum(x['split'] == split for x in rows)} clips so far in {split}", flush=True)
    m = pd.DataFrame(rows)
    assert m.path.is_unique
    m.to_csv(OUT / "manifest.csv", index=False)
    print(m.split.value_counts().to_string())
    print(f"{len(m)} clips -> {OUT / 'manifest.csv'} ({conv} converted, the rest written as-is)")


if __name__ == "__main__":
    main()
