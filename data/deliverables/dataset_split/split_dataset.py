#!/usr/bin/env python3
"""
Reproduce the deterministic taxonomy + split from manifestfile(1).csv.

Usage:
  python split_dataset.py manifestfile.csv output_dir

The script uses only pandas/numpy plus the Python standard library.
"""
from pathlib import Path
import hashlib, json, re, sys
import numpy as np
import pandas as pd

SEED = 20260925
RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}
CATEGORY_MAP = {
    "positives": "positive_wakeword",
    "negatives_general": "negative_general_speech",
    "negatives_media": "negative_media",
    "negatives_confusable": "negative_confusable",
    "negatives_partial": "negative_partial_wakeword",
    "negatives_silence": "negative_silence_noise",
}
NOISE_SOURCES = {"synth_white", "synth_pink", "synth_hum", "synth_fan", "synth_clicks"}

def stable_hash(value):
    return int.from_bytes(
        hashlib.sha256(f"{SEED}|{value}".encode("utf-8")).digest()[:8], "big"
    )

def atomic_speakers(series):
    out = set()
    for value in series.dropna():
        out.update(str(value).split("|"))
    return out

def build(input_csv, output_dir):
    input_csv, output_dir = Path(input_csv), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_csv)

    required = {
        "filepath","label","category","source","source_id","speaker_id",
        "recording_id","duration_sec","sample_rate","voice","speed","pitch",
        "transcription","edit_distance","parent_filepath","created_at"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.filepath.duplicated().any():
        raise AssertionError("Duplicate filepath")

    parent_recording = df.set_index("filepath")["recording_id"].to_dict()
    df["source_group"] = df["parent_filepath"].map(parent_recording).fillna(df["recording_id"])
    df["canonical_label"] = df["label"].astype(str).str.strip().str.lower()
    df["canonical_category"] = df["category"].map(CATEGORY_MAP)
    if df["canonical_category"].isna().any():
        raise ValueError("Unmapped category present")
    df["provenance_type"] = np.select(
        [df.parent_filepath.notna(), df.source.isin(NOISE_SOURCES)],
        ["derived_from_positive", "synthetic_noise"],
        default="synthetic_tts",
    )
    df["session_id"] = pd.NA
    df["speaker_id_normalized"] = df.speaker_id.apply(
        lambda x: x.strip() if isinstance(x, str) else pd.NA
    )
    df["transcription_normalized"] = df.transcription.apply(
        lambda x: re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()
        if isinstance(x, str) else pd.NA
    )
    df["complete_hey_delta"] = df.transcription_normalized.eq("hey delta")
    df.loc[df.canonical_label.eq("positive"), "complete_hey_delta"] = True
    df["positive_rule_status"] = np.where(
        df.canonical_label.eq("positive"), "complete_wakeword",
        np.where(df.canonical_label.eq("negative"), "not_positive", "unknown")
    )
    df["is_derived"] = df.parent_filepath.notna()

    categories = sorted(df.canonical_category.unique())
    rows = []
    for gid, g in df.groupby("source_group", sort=False):
        r = {"source_group": gid, "n": len(g)}
        for c in categories:
            r[c] = int((g.canonical_category == c).sum())
        rows.append(r)
    groups = pd.DataFrame(rows)

    positive_groups = set(groups.loc[groups.positive_wakeword > 0, "source_group"])
    ranked = groups[groups.source_group.isin(positive_groups)].sort_values(
        ["positive_wakeword","n","source_group"], ascending=[False,False,True]
    )
    if len(ranked) < 2:
        raise ValueError("Need >=2 positive source groups")
    assignment = {gid: "train" for gid in positive_groups}
    assignment[ranked.iloc[0].source_group] = "test"
    assignment[ranked.iloc[1].source_group] = "validation"

    target_n = {s: len(df)*RATIOS[s] for s in RATIOS}
    target_cat = {
        s: {c: int((df.canonical_category == c).sum())*RATIOS[s] for c in categories}
        for s in RATIOS
    }
    stats_n = {s: 0 for s in RATIOS}
    stats_cat = {s: {c: 0 for c in categories} for s in RATIOS}
    for _, r in groups[groups.source_group.isin(positive_groups)].iterrows():
        s = assignment[r.source_group]
        stats_n[s] += int(r.n)
        for c in categories: stats_cat[s][c] += int(r[c])

    remaining = groups[~groups.source_group.isin(positive_groups)].copy()
    remaining["hash"] = remaining.source_group.map(stable_hash)
    remaining = remaining.sort_values(["n","hash"], ascending=[False,True])

    for _, r in remaining.iterrows():
        choices = []
        for s in RATIOS:
            n2 = dict(stats_n)
            c2 = {x: dict(stats_cat[x]) for x in RATIOS}
            n2[s] += int(r.n)
            for c in categories: c2[s][c] += int(r[c])
            obj = sum(abs(n2[x]-target_n[x])/max(target_n[x],1.0) for x in RATIOS)
            obj += sum(
                abs(c2[x][c]-target_cat[x][c])/max(target_cat[x][c],1.0)
                for x in RATIOS for c in categories
            )
            choices.append((obj, stable_hash(f"{r.source_group}|{s}"), s))
        s = min(choices)[2]
        assignment[r.source_group] = s
        stats_n[s] += int(r.n)
        for c in categories: stats_cat[s][c] += int(r[c])

    df["split"] = df.source_group.map(assignment)
    if df.split.isna().any(): raise AssertionError("Unassigned rows")

    # Leakage checks.
    sets = {s:set(df.loc[df.split==s,"source_group"]) for s in RATIOS}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not (sets[a] & sets[b]), f"source_group overlap {a}/{b}"
    recs = {s:set(df.loc[df.split==s,"recording_id"]) for s in RATIOS}
    for a,b in [("train","validation"),("train","test"),("validation","test")]:
        assert not (recs[a] & recs[b]), f"recording_id overlap {a}/{b}"

    path_to_split = df.set_index("filepath").split.to_dict()
    for _, r in df[df.parent_filepath.notna()].iterrows():
        assert path_to_split[r.parent_filepath] == r.split, f"Parent/child leakage: {r.filepath}"

    assert (df.groupby("source_group").split.nunique() == 1).all()

    columns = list(required) + [
        "canonical_label","canonical_category","provenance_type","source_group",
        "session_id","speaker_id_normalized","transcription_normalized",
        "complete_hey_delta","positive_rule_status","is_derived","split"
    ]
    normalized = df[columns]
    normalized.to_csv(output_dir/"normalized_manifest.csv", index=False)
    for s in RATIOS:
        normalized[normalized.split==s].to_csv(output_dir/f"{s}_manifest.csv", index=False)
    pd.DataFrame({"source_group":list(assignment), "split":[assignment[x] for x in assignment]}).to_csv(
        output_dir/"split_assignments.csv", index=False
    )
    return df

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python split_dataset.py INPUT_CSV OUTPUT_DIR")
    build(sys.argv[1], sys.argv[2])
