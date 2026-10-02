#!/usr/bin/env python3
"""
Build the curated RIR bank for rir_reverberation from the MIT IR Survey
(Traer & McDermott 2016, PNAS; CC-BY 4.0; 271 real impulse responses of everyday spaces).

Source: external_raw/mit_ir/16khz/*.wav (Hugging Face mirror
davidscripka/MIT_environmental_impulse_responses, the 16 kHz copy of
https://mcdermottlab.mit.edu/Reverb/IR_Survey.html).

For each response, RT60 is estimated from the Schroeder backward-integrated energy decay
(T20: linear fit from -5 to -25 dB, extrapolated to -60 dB; T30 used when T20 fails).
Only responses whose RT60 lies in augmentation_config.json rir_reverberation.parameters.rt60_s
are kept: the config restricts reverberation to deployment-plausible rooms. Kept files are
copied to data/rir/ and listed in data/rir/rir_bank.csv (id, filepath, rt60_s, room, source).

Usage: python build_rir_bank.py
"""
from pathlib import Path
import json, re, shutil

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "external_raw/mit_ir/16khz"
OUT = ROOT / "data/rir"
CFG = json.loads((HERE / "augmentation_config.json").read_text())


def rt60(ir, sr):
    e = ir.astype(np.float64) ** 2
    e = e[int(np.argmax(np.abs(ir))):]                      # from the direct path onward
    edc = np.cumsum(e[::-1])[::-1]
    edc_db = 10 * np.log10(edc / edc[0] + 1e-300)
    t = np.arange(len(edc_db)) / sr
    for lo, hi in ((-5, -25), (-5, -35)):                    # T20, then T30
        m = (edc_db <= lo) & (edc_db >= hi)
        if m.sum() > 10:
            slope = np.polyfit(t[m], edc_db[m], 1)[0]       # dB per second
            if slope < 0:
                return float(-60 / slope)
    return float("nan")


def main():
    spec = next(t for t in CFG["enabled_transforms"] if t["name"] == "rir_reverberation")["parameters"]["rt60_s"]
    rows = []
    for p in sorted(SRC.glob("*.wav")):
        ir, sr = sf.read(p)
        if ir.ndim > 1:
            ir = ir[:, 0]
        r = rt60(ir, sr)
        room = re.sub(r"^h\d+_|_\d+txts$", "", p.stem)
        rows.append(dict(id=f"mit_{p.stem.split('_')[0]}", src=p, rt60_s=round(r, 3), room=room, sample_rate=sr))
    df = pd.DataFrame(rows)
    keep = df[(df.rt60_s >= spec["min"]) & (df.rt60_s <= spec["max"])].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    for r in keep.itertuples(index=False):
        shutil.copy(r.src, OUT / r.src.name)
    keep["filepath"] = [f"data/rir/{Path(s).name}" for s in keep.src]
    keep["source"] = "MIT IR Survey (Traer & McDermott 2016), CC-BY 4.0"
    keep[["id", "filepath", "rt60_s", "room", "sample_rate", "source"]].to_csv(OUT / "rir_bank.csv", index=False)
    print(f"{len(df)} MIT responses; RT60 median {df.rt60_s.median():.2f} s; "
          f"kept {len(keep)} in [{spec['min']}, {spec['max']}] s -> {OUT / 'rir_bank.csv'}")


if __name__ == "__main__":
    main()
