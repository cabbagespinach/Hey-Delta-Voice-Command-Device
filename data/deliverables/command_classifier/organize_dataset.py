#!/usr/bin/env python3
"""
Gather the owner's schema-B command data into ONE folder, one subfolder per class (owner, 2026-10-01).

    python organize_dataset.py      # -> data/command_dataset/<class>/<class>_[source]_[description]_NNNNN.wav
                                    #    + data/command_dataset/manifest.csv (new name -> original path, text, split)

Input: data/commands_schema_b.csv (every clip already relabelled to the schema-B classes by schema_b.py).
Only data the classmates do NOT have: rows whose source is on the class list (data/AI231 ME2 - Datasets.csv) are
left out (SLURP, Fluent Speech Commands, Timers and Such, and cut-off fragments made from them), plus MSWC (its
audio is Common Voice, which is on the list) and the classmates' own recordings (other_speakers).
Source / description come only from existing metadata (dataset column, old label); nothing is inferred by a model.
Files are hard links (no extra disk); falls back to a copy if a hard link is not possible.
"""
from pathlib import Path
import os, re, shutil

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
D = ROOT / "data"
OUT = D / "command_dataset"

EXCLUDE = {   # dataset -> reason
    "web_fsc": "Fluent Speech Commands (on class list)",
    "web_slurp": "SLURP (on class list)",
    "web_tas": "Timers and Such (on class list)",
    "fragments_web_slurp": "cut-offs of SLURP (on class list)",
    "fragments_web_tas": "cut-offs of Timers and Such (on class list)",
    "reuse_mswc": "MSWC = Common Voice audio (on class list)",
    "other_speakers": "classmates' own recordings (they have them)",
}
SOURCE = {    # dataset -> (source, description) ; no underscores inside a field
    "owner_recordings": ("owner", ""),
    "converted_owner": ("owner-vc", ""),
    "reuse_owner": ("owner", "room"),
    "synthetic_piper": ("tts-piper", ""),
    "synthetic_ph": ("tts-edge", ""),
    "synthetic_clone": ("tts-chatterbox", ""),
    "variants_piper": ("tts-piper", "variant"),
    "variants_ph": ("tts-edge", "variant"),
    "variants_clone": ("tts-chatterbox", "variant"),
    "fragments_owner_recordings": ("owner", "fragment"),
    "fragments_synthetic_piper": ("tts-piper", "fragment"),
    "fragments_synthetic_ph": ("tts-edge", "fragment"),
    "fragments_synthetic_clone": ("tts-chatterbox", "fragment"),
    "reuse_fleurs": ("fleurs", "speech"),
    "reuse_musan": ("musan", ""),
}


def tok(s):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-").lower()


def describe(r):
    src, desc = SOURCE[r["dataset"]]
    parts = [desc] if desc else []
    if r["class"] == "unknown":
        old = r["label"]
        if r["dataset"] == "reuse_musan":                   # Ext-MUSAN-<music|noise|speech>-...
            m = re.search(r"Ext-MUSAN-([a-z]+)-", r["path"])
            sub = m.group(1) if m else ""
        elif old.startswith("unknown_"):
            sub = old[len("unknown_"):]
        else:                                              # old command with no schema class -> near-miss
            sub = "nearmiss-" + tok(old)
        if sub and sub not in parts and not (desc == "fragment" and sub == "partial"):
            parts.append(sub)
    return src, "-".join(tok(p) for p in parts)


def place(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    df = pd.read_csv(D / "commands_schema_b.csv")
    unknown_ds = set(df.dataset) - set(EXCLUDE) - set(SOURCE)
    assert not unknown_ds, f"unmapped datasets: {unknown_ds}"
    keep = df[~df.dataset.isin(EXCLUDE)].copy()
    print("excluded:", df[df.dataset.isin(EXCLUDE)].groupby("dataset").size().to_dict())

    if OUT.exists():
        shutil.rmtree(OUT)
    rows = []
    for cls, g in keep.groupby("class", sort=True):
        folder = OUT / cls.lower()
        folder.mkdir(parents=True)
        for i, r in enumerate(g.sort_values(["dataset", "path"]).to_dict("records"), 1):
            src, desc = describe(r)
            name = "_".join(p for p in (cls.lower(), src, desc, f"{i:05d}") if p) + Path(r["path"]).suffix
            place(ROOT / r["path"], folder / name)
            rows.append(dict(file=f"{cls.lower()}/{name}", label=cls.lower(), source=src, description=desc,
                             **{k: r[k] for k in ("dataset", "speaker", "split", "real", "license", "text")},
                             original_path=r["path"]))
    man = pd.DataFrame(rows)
    man.to_csv(OUT / "manifest.csv", index=False)
    print(man.groupby("label").size().to_string())
    print("total", len(man))
    print(man.groupby(["source", "description"]).size().to_string())


if __name__ == "__main__":
    main()
