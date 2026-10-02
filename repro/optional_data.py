#!/usr/bin/env python3
"""
Datasets that cannot be shared (Hey Snips, Qualcomm, Fluent Speech Commands, classmates' recordings).

    python repro/optional_data.py check      # report which are present; print how to obtain the missing ones
    python repro/optional_data.py restore    # rebuild, from a copy you obtained, exactly the clips we used
    python repro/optional_data.py filter     # make the clip lists match the audio present (run by reproduce.sh)

`filter` keeps the full lists as <name>.full.csv, drops rows whose audio is missing and writes
repro_outputs/missing_data.md (what was left out, per dataset). When wakeword windows are dropped, the frozen
isolated validation/test manifests are set aside so they re-freeze on the reduced data, and the normalisation
statistics are recomputed (reproduce.sh). With every dataset present, nothing changes.
"""
from pathlib import Path
import json, shutil, sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro_common import REPO, OPTIONAL   # noqa: E402

import pandas as pd

OUT = REPO / "repro_outputs"
STATUS = OUT / "optional_data.json"
LISTS = {  # list -> path column
    "data/deliverables/segmentation_windowing/outputs/windows.csv": "filepath",
    "data/commands_schema_b_all.csv": "path",
    "data/commands_hf_only.csv": "path",
    "data/commands_hf_plus.csv": "path",
}
ISOLATED = REPO / "data/deliverables/evaluation/isolated_set"


def present(key):
    o = OPTIONAL[key]
    if key == "classmate_recordings":
        return (REPO / "data/commands_other_speakers").is_dir()
    return (REPO / o["raw"]).exists() if o["raw"] else False


def referenced(key):
    o = OPTIONAL[key]
    if o["files"] is None:
        return []
    refs = set()
    for f, col in LISTS.items():
        src = REPO / (f[:-4] + ".full.csv") if (REPO / (f[:-4] + ".full.csv")).exists() else REPO / f
        if src.exists():
            refs |= {p for p in pd.read_csv(src, usecols=[col])[col].dropna() if o["files"].search(p)}
    return sorted(refs)


def check():
    OUT.mkdir(exist_ok=True)
    status = {}
    for k, o in OPTIONAL.items():
        refs = referenced(k)
        have = sum((REPO / p).exists() for p in refs)
        ok = present(k) and (have == len(refs) or not refs)
        status[k] = dict(present=ok, raw_present=present(k), clips_needed=len(refs), clips_present=have)
        if ok:
            print(f"[ok]      {o['name']}: present" + (f" ({have} clips)" if refs else ""))
            continue
        bar = "-" * 100
        print(f"\n{bar}\n[MISSING] {o['name']}\n  used for : {o['used_for']}\n  licence  : {o['licence']}"
              f"\n  without it: {o['without']}\n  to obtain it:")
        for line in o["how"]:
            print(f"    {line}")
        if present(k) and have < len(refs):
            print(f"  (your copy is there; {len(refs) - have} clips still need rebuilding: run setup_data.sh again)")
        print(bar)
    STATUS.write_text(json.dumps(status, indent=2) + "\n")
    print(f"\nstatus -> {STATUS.relative_to(REPO)}. The pipeline runs either way; missing data is reported in the results.")


def restore():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "data/deliverables/command_classifier"))
    if present("hey_snips"):
        import integrate_external_datasets as ied
        sel = pd.read_csv(REPO / "external_raw/hey_snips/selection.csv")
        base = REPO / OPTIONAL["hey_snips"]["raw"]
        n = 0
        for r in sel.itertuples(index=False):         # identical to integrate_external_datasets.heysnips()
            cat = "negatives_confusable" if r.is_hotword == 1 else "negatives_general"
            if not (REPO / f"data/{cat}/Ext-HeySnips-{r.id}.wav").exists():
                ied.write(ied.load16(base.parent / r.audio_file_path), cat, f"Ext-HeySnips-{r.id}")
                n += 1
        print(f"Hey Snips: {n} clips rebuilt")
    if present("fsc"):
        import soundfile as sf
        from generate_synthetic_commands import to16k, trim
        from prepare_real_commands import write
        base = REPO / OPTIONAL["fsc"]["raw"]
        wav = {Path(p).stem: base / p for s in ("train", "valid", "test")
               for p in pd.read_csv(base / f"data/{s}_data.csv").path}
        n = 0
        for rel in referenced("fsc"):                   # identical to prepare_real_commands.py (fsc)
            if (REPO / rel).exists():
                continue
            y, sr = sf.read(str(wav[Path(rel).stem.split("__", 1)[1]]), dtype="float32")
            write(REPO / rel, trim(to16k(y, sr)))
            n += 1
        print(f"Fluent Speech Commands: {n} clips rebuilt")


def filt():
    OUT.mkdir(exist_ok=True)
    L = ["# Data left out of this reproduction", "",
         "Rows whose audio is not present (a dataset that cannot be shared and was not obtained). "
         "The full lists are kept next to each list as `<name>.full.csv`.", ""]
    dropped_windows = 0
    for f, col in LISTS.items():
        p, full = REPO / f, REPO / (f[:-4] + ".full.csv")
        if not p.exists():
            continue
        if not full.exists():
            shutil.copy2(p, full)
        d = pd.read_csv(full, low_memory=False)
        ok = d[col].map(lambda x: (REPO / x).exists())
        d[ok].to_csv(p, index=False)
        if (~ok).any():
            by = {}
            for k, o in OPTIONAL.items():
                if o["files"] is not None:
                    by[o["name"]] = int(d.loc[~ok, col].map(lambda x: bool(o["files"].search(x))).sum())
            other = int((~ok).sum()) - sum(by.values())
            L += [f"## {f}", "", f"{int((~ok).sum())} of {len(d)} rows left out: " +
                  ", ".join(f"{k} {v}" for k, v in by.items() if v) + (f", other {other}" if other else ""), ""]
            if "split" in d:
                L += [d[~ok].groupby("split").size().rename("rows left out").to_markdown(), ""]
        if f.endswith("windows.csv"):
            dropped_windows = int((~ok).sum())
    if dropped_windows:
        keep = ISOLATED / "frozen_full_data"
        keep.mkdir(exist_ok=True)
        for x in ISOLATED.glob("isolated_*_manifest.*"):
            shutil.move(str(x), keep / x.name)
        L += [f"Wakeword: {dropped_windows} windows left out, so the frozen isolated validation/test manifests were "
              "moved to `isolated_set/frozen_full_data/` and re-freeze on the reduced data (normalisation statistics "
              "recomputed). Wakeword numbers are therefore close to, not identical with, the reported ones.", ""]
    else:
        L += ["Nothing was left out: this is an exact-data reproduction.", ""]
    (OUT / "missing_data.md").write_text("\n".join(L) + "\n")
    (OUT / "wakeword_reduced.flag").write_text("1\n" if dropped_windows else "0\n")
    print("\n".join(L))


if __name__ == "__main__":
    {"check": check, "restore": restore, "filter": filt}[sys.argv[1]]()
